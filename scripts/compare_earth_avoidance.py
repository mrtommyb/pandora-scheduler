#!/usr/bin/env python3
"""Run scheduler twice with different visibility_earth_deg and compare outputs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RunSummary:
    earth_deg: float
    schedule_csv: Path
    tracker_csv: Path | None
    rows: int
    unique_targets: int
    non_empty_comments: int
    total_hours_by_target: dict[str, float]
    transit_acquired_by_planet: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare scheduler outputs across two visibility_earth_deg values"
    )
    parser.add_argument("--start", required=True, help="Schedule start date/time")
    parser.add_argument("--end", required=True, help="Schedule end date/time")
    parser.add_argument("--config", type=Path, required=True, help="Base JSON config")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output_directory/earth_compare"),
        help="Root output directory for comparison runs",
    )
    parser.add_argument("--earth-a", type=float, required=True, help="First earth angle")
    parser.add_argument("--earth-b", type=float, required=True, help="Second earth angle")
    parser.add_argument(
        "--target-definitions",
        type=Path,
        default=None,
        help="Override --target-definitions path",
    )
    parser.add_argument(
        "--gmat-ephemeris",
        type=Path,
        default=None,
        help="Override --gmat-ephemeris path",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra arg to pass through to run_scheduler.py (repeatable)",
    )
    parser.add_argument(
        "--skip-xml",
        action="store_true",
        help="Pass --skip-xml to reduce runtime",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Pass --show-progress to scheduler",
    )
    return parser.parse_args()


def parse_datetime(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported datetime format: {value!r}") from exc


def latest_schedule_csv(output_dir: Path) -> Path:
    files = sorted(output_dir.glob("Pandora_Schedule_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No Pandora_Schedule_*.csv found in {output_dir}")
    return files[-1]


def read_schedule_summary(earth_deg: float, output_dir: Path) -> RunSummary:
    schedule_csv = latest_schedule_csv(output_dir)
    tracker_csv = output_dir / "tracker.csv"

    total_hours_by_target: dict[str, float] = defaultdict(float)
    unique_targets: set[str] = set()
    non_empty_comments = 0
    rows = 0

    with schedule_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            target = str(row.get("Target", "")).strip()
            if target:
                unique_targets.add(target)

            comments = str(row.get("Comments", "")).strip()
            if comments and comments.lower() not in {"nan", "none"}:
                non_empty_comments += 1

            start_raw = row.get("Observation Start")
            stop_raw = row.get("Observation Stop")
            if not target or not start_raw or not stop_raw:
                continue

            try:
                start_dt = parse_datetime(start_raw)
                stop_dt = parse_datetime(stop_raw)
            except ValueError:
                continue

            duration_h = (stop_dt - start_dt).total_seconds() / 3600.0
            total_hours_by_target[target] += duration_h

    transit_acquired_by_planet: dict[str, float] = {}
    if tracker_csv.exists():
        with tracker_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                planet = str(row.get("Planet Name", "")).strip()
                val = row.get("Transits Acquired")
                if not planet or val is None or val == "":
                    continue
                try:
                    transit_acquired_by_planet[planet] = float(val)
                except ValueError:
                    continue

    return RunSummary(
        earth_deg=earth_deg,
        schedule_csv=schedule_csv,
        tracker_csv=tracker_csv if tracker_csv.exists() else None,
        rows=rows,
        unique_targets=len(unique_targets),
        non_empty_comments=non_empty_comments,
        total_hours_by_target=dict(total_hours_by_target),
        transit_acquired_by_planet=transit_acquired_by_planet,
    )


def run_scheduler_once(
    *,
    base_config: dict[str, Any],
    earth_deg: float,
    output_dir: Path,
    start: str,
    end: str,
    target_definitions: Path | None,
    gmat_ephemeris: Path | None,
    skip_xml: bool,
    show_progress: bool,
    extra_args: list[str],
) -> RunSummary:
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(base_config)
    cfg["visibility_earth_deg"] = float(earth_deg)

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    cmd = [
        sys.executable,
        "run_scheduler.py",
        "--start",
        start,
        "--end",
        end,
        "--output",
        str(output_dir),
        "--config",
        str(config_path),
    ]

    if target_definitions is not None:
        cmd.extend(["--target-definitions", str(target_definitions)])
    if gmat_ephemeris is not None:
        cmd.extend(["--gmat-ephemeris", str(gmat_ephemeris)])
    if skip_xml:
        cmd.append("--skip-xml")
    if show_progress:
        cmd.append("--show-progress")
    cmd.extend(extra_args)

    print(f"\\n=== Running with visibility_earth_deg={earth_deg} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    return read_schedule_summary(float(earth_deg), output_dir)


def top_hour_deltas(a: RunSummary, b: RunSummary, limit: int = 15) -> list[tuple[str, float]]:
    names = set(a.total_hours_by_target) | set(b.total_hours_by_target)
    deltas: list[tuple[str, float]] = []
    for name in names:
        delta = b.total_hours_by_target.get(name, 0.0) - a.total_hours_by_target.get(name, 0.0)
        if abs(delta) > 1e-9:
            deltas.append((name, delta))
    deltas.sort(key=lambda item: abs(item[1]), reverse=True)
    return deltas[:limit]


def tracker_deltas(a: RunSummary, b: RunSummary, limit: int = 15) -> list[tuple[str, float]]:
    names = set(a.transit_acquired_by_planet) | set(b.transit_acquired_by_planet)
    deltas: list[tuple[str, float]] = []
    for name in names:
        delta = b.transit_acquired_by_planet.get(name, 0.0) - a.transit_acquired_by_planet.get(name, 0.0)
        if abs(delta) > 1e-9:
            deltas.append((name, delta))
    deltas.sort(key=lambda item: abs(item[1]), reverse=True)
    return deltas[:limit]


def main() -> int:
    args = parse_args()
    with args.config.open(encoding="utf-8") as handle:
        base_config = json.load(handle)

    run_a_dir = args.output_root / f"earth_{args.earth_a:g}"
    run_b_dir = args.output_root / f"earth_{args.earth_b:g}"

    summary_a = run_scheduler_once(
        base_config=base_config,
        earth_deg=args.earth_a,
        output_dir=run_a_dir,
        start=args.start,
        end=args.end,
        target_definitions=args.target_definitions,
        gmat_ephemeris=args.gmat_ephemeris,
        skip_xml=args.skip_xml,
        show_progress=args.show_progress,
        extra_args=args.extra_arg,
    )
    summary_b = run_scheduler_once(
        base_config=base_config,
        earth_deg=args.earth_b,
        output_dir=run_b_dir,
        start=args.start,
        end=args.end,
        target_definitions=args.target_definitions,
        gmat_ephemeris=args.gmat_ephemeris,
        skip_xml=args.skip_xml,
        show_progress=args.show_progress,
        extra_args=args.extra_arg,
    )

    print("\n=== Comparison Summary ===")
    print(
        f"Schedule rows: {summary_a.earth_deg:g}deg={summary_a.rows}, "
        f"{summary_b.earth_deg:g}deg={summary_b.rows}, "
        f"delta={summary_b.rows - summary_a.rows:+d}"
    )
    print(
        f"Unique targets: {summary_a.earth_deg:g}deg={summary_a.unique_targets}, "
        f"{summary_b.earth_deg:g}deg={summary_b.unique_targets}, "
        f"delta={summary_b.unique_targets - summary_a.unique_targets:+d}"
    )
    print(
        f"Non-empty comments: {summary_a.earth_deg:g}deg={summary_a.non_empty_comments}, "
        f"{summary_b.earth_deg:g}deg={summary_b.non_empty_comments}, "
        f"delta={summary_b.non_empty_comments - summary_a.non_empty_comments:+d}"
    )

    print("\nTop target-hour deltas (B - A):")
    hour_deltas = top_hour_deltas(summary_a, summary_b)
    if not hour_deltas:
        print("  No target hour changes detected.")
    else:
        for target, delta in hour_deltas:
            print(f"  {target}: {delta:+.2f} h")

    print("\nTop transits-acquired deltas (B - A):")
    t_deltas = tracker_deltas(summary_a, summary_b)
    if not t_deltas:
        print("  No tracker transit changes detected.")
    else:
        for target, delta in t_deltas:
            print(f"  {target}: {delta:+.1f}")

    print("\nOutputs:")
    print(f"  A ({summary_a.earth_deg:g} deg): {summary_a.schedule_csv}")
    print(f"  B ({summary_b.earth_deg:g} deg): {summary_b.schedule_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
