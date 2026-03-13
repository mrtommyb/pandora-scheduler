#!/usr/bin/env python3
"""Debug occultation scheduling for a single visit.

Usage examples::

    # By visit index (0-based row in the schedule CSV)
    poetry run python scripts/debug_occultation_visit.py \\
        --schedule output_standalone/Pandora_Schedule_*.csv \\
        --data-dir output_standalone/data \\
        --visit 3

    # By target name (first matching visit)
    poetry run python scripts/debug_occultation_visit.py \\
        --schedule output_standalone/Pandora_Schedule_*.csv \\
        --data-dir output_standalone/data \\
        --target "GJ_367b"

    # List all visits
    poetry run python scripts/debug_occultation_visit.py \\
        --schedule output_standalone/Pandora_Schedule_*.csv \\
        --list-visits

Reuses production helpers from ``observation_utils`` and ``utils.io``
so the debug output reflects the actual scheduling logic.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

from pandorascheduler_rework.utils.io import (
    load_visibility_arrays_cached,
    resolve_star_visibility_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_schedule(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Observation Start"] = pd.to_datetime(df["Observation Start"])
    df["Observation Stop"] = pd.to_datetime(df["Observation Stop"])
    return df


def _find_occultation_candidates(data_dir: Path) -> list[str]:
    """Return star names with visibility parquet files under aux_targets/."""
    aux_dir = data_dir / "aux_targets"
    if not aux_dir.is_dir():
        return []
    candidates = []
    for child in sorted(aux_dir.iterdir()):
        if child.is_dir():
            vis_file = child / f"Visibility for {child.name}.parquet"
            if vis_file.is_file():
                candidates.append(child.name)
    return candidates


def _load_occultation_list(data_dir: Path) -> pd.DataFrame | None:
    """Load optional occultation-standard_targets.csv."""
    for name in ("occultation-standard_targets.csv", "all_targets.csv"):
        p = data_dir / name
        if p.is_file():
            return pd.read_csv(p)
    return None


def _format_mjd_utc(mjd: float) -> str:
    try:
        dt = Time(mjd, format="mjd", scale="utc").to_datetime()
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return f"MJD {mjd:.6f}"


# ---------------------------------------------------------------------------
# Core debug routine
# ---------------------------------------------------------------------------


def debug_visit(
    schedule: pd.DataFrame,
    visit_idx: int,
    data_dir: Path,
    *,
    occ_list: pd.DataFrame | None = None,
) -> None:
    """Print detailed occultation debug information for one visit."""
    row = schedule.iloc[visit_idx]
    target = row["Target"]
    obs_start = row["Observation Start"]
    obs_stop = row["Observation Stop"]
    ra = row.get("RA", float("nan"))
    dec = row.get("DEC", float("nan"))

    print("=" * 72)
    print(f"Visit {visit_idx}: {target}")
    print(f"  Start : {obs_start}")
    print(f"  Stop  : {obs_stop}")
    print(f"  RA/DEC: {ra}, {dec}")
    duration_min = (obs_stop - obs_start).total_seconds() / 60.0
    print(f"  Duration: {duration_min:.1f} min")
    if "Comments" in row.index and pd.notna(row["Comments"]):
        print(f"  Comments: {row['Comments']}")
    print()

    # Convert to MJD for visibility lookup
    start_mjd = Time(obs_start).mjd
    stop_mjd = Time(obs_stop).mjd

    # Find candidate occultation targets
    aux_dir = data_dir / "aux_targets"
    candidates = _find_occultation_candidates(data_dir)
    if not candidates:
        print("  [!] No auxiliary target visibility files found in", aux_dir)
        return

    print(f"Candidate occultation targets: {len(candidates)}")
    print()

    # Treat the visit as a single occultation interval
    starts = np.array([start_mjd])
    stops = np.array([stop_mjd])

    # Evaluate each candidate
    coverage_results: list[dict] = []

    for name in candidates:
        vis_file = resolve_star_visibility_file(aux_dir, name)
        if vis_file is None:
            continue
        data = load_visibility_arrays_cached(vis_file)
        if data is None:
            continue
        vis_times, vis_flags = data

        # Restrict to visit interval
        mask = (vis_times >= start_mjd) & (vis_times <= stop_mjd)
        n_samples = int(mask.sum())
        if n_samples == 0:
            coverage_results.append(
                {"name": name, "samples": 0, "visible": 0, "coverage": 0.0, "fully_visible": False}
            )
            continue
        n_visible = int((vis_flags[mask] == 1).sum())
        coverage = n_visible / n_samples if n_samples > 0 else 0.0
        fully_visible = bool(np.all(vis_flags[mask] == 1))

        coverage_results.append(
            {
                "name": name,
                "samples": n_samples,
                "visible": n_visible,
                "coverage": coverage,
                "fully_visible": fully_visible,
            }
        )

    # Sort by coverage descending
    coverage_results.sort(key=lambda r: r["coverage"], reverse=True)

    # Print pass-by-pass analysis
    fully_visible = [r for r in coverage_results if r["fully_visible"]]
    partially_visible = [r for r in coverage_results if 0 < r["coverage"] < 1.0]
    no_coverage = [r for r in coverage_results if r["coverage"] == 0.0]

    print("--- Pass 1 Analysis (single target for all intervals) ---")
    if fully_visible:
        print(f"  {len(fully_visible)} candidate(s) with FULL visibility:")
        for r in fully_visible[:10]:
            print(f"    {r['name']:30s}  {r['visible']}/{r['samples']} samples  100%")
        if len(fully_visible) > 10:
            print(f"    ... and {len(fully_visible) - 10} more")
        print(f"  -> Pass 1 would select: {fully_visible[0]['name']}")
    else:
        print("  No candidate covers the entire interval.")
        print("  -> Pass 1 fails; escalating to Pass 2.")
    print()

    print("--- Pass 2 Analysis (greedy multi-target fill) ---")
    print("  (Single-interval visit: identical to Pass 1 outcome)")
    print()

    print("--- Pass 3 Analysis (best-effort partial coverage) ---")
    if partially_visible:
        print(f"  {len(partially_visible)} candidate(s) with partial visibility:")
        for r in partially_visible[:15]:
            print(
                f"    {r['name']:30s}  {r['visible']}/{r['samples']} samples  "
                f"{r['coverage']*100:.1f}%"
            )
        if len(partially_visible) > 15:
            print(f"    ... and {len(partially_visible) - 15} more")
        best = coverage_results[0]
        if not best["fully_visible"] and best["coverage"] > 0:
            print(f"  -> Pass 3 would select: {best['name']} ({best['coverage']*100:.1f}%)")
    else:
        print("  No candidates with partial visibility.")
    print()

    # Pass 4: minute-resolution analysis for top candidates
    print("--- Pass 4 Analysis (minute-resolution greedy) ---")
    minute_scale = 1440.0
    start_idx = int(np.floor(start_mjd * minute_scale))
    stop_idx = int(np.ceil(stop_mjd * minute_scale))
    total_minutes = stop_idx - start_idx
    print(f"  Interval: {total_minutes} minutes")

    if total_minutes <= 0:
        print("  [!] Zero-length interval — nothing to schedule")
        return

    minutes_idx = np.arange(start_idx, stop_idx)
    top_candidates = coverage_results[:5]

    for r in top_candidates:
        if r["samples"] == 0:
            continue
        vis_file = resolve_star_visibility_file(aux_dir, r["name"])
        if vis_file is None:
            continue
        data = load_visibility_arrays_cached(vis_file)
        if data is None:
            continue
        vis_times, vis_flags = data
        vis_min_idx = np.round(vis_times * minute_scale).astype(int)
        visible_set = set(vis_min_idx[vis_flags == 1])
        covered = np.isin(minutes_idx, np.fromiter(visible_set, dtype=int))
        n_covered = int(covered.sum())

        # Find contiguous runs
        runs = []
        i = 0
        while i < len(covered):
            if covered[i]:
                run_start = i
                while i < len(covered) and covered[i]:
                    i += 1
                runs.append((run_start, i - run_start))
            else:
                i += 1

        run_desc = ", ".join(f"{length}min@+{start}m" for start, length in runs[:5])
        if len(runs) > 5:
            run_desc += f" ... (+{len(runs) - 5} more runs)"
        print(
            f"  {r['name']:30s}  {n_covered}/{total_minutes} min  "
            f"runs: {run_desc or 'none'}"
        )

    print()

    # Occultation list info
    if occ_list is not None and not occ_list.empty:
        print("--- Occultation Target List ---")
        if "Star Name" in occ_list.columns:
            occ_names = set(occ_list["Star Name"])
            in_list = [r for r in coverage_results if r["name"] in occ_names]
            not_in_list = [r for r in coverage_results if r["name"] not in occ_names and r["coverage"] > 0]
            print(f"  {len(occ_names)} targets in occultation list")
            print(f"  {len(in_list)} visible candidate(s) from occultation list")
            print(f"  {len(not_in_list)} visible candidate(s) NOT in occultation list")
        print()

    # Summary
    print("--- Summary ---")
    if fully_visible:
        print(f"  OUTCOME: Would schedule {fully_visible[0]['name']} (Pass 1, full visibility)")
    elif coverage_results and coverage_results[0]["coverage"] > 0:
        best = coverage_results[0]
        print(
            f"  OUTCOME: Would schedule {best['name']} "
            f"(Pass 3, {best['coverage']*100:.1f}% coverage)"
        )
    else:
        print("  OUTCOME: No occultation target available for this visit")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug occultation scheduling for a single visit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--schedule", type=Path, required=True, help="Path to schedule CSV"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to data directory containing aux_targets/ (default: inferred from schedule path)",
    )
    parser.add_argument(
        "--visit", type=int, default=None, help="Visit index (0-based row number)"
    )
    parser.add_argument(
        "--target", type=str, default=None, help="Target name (uses first matching visit)"
    )
    parser.add_argument(
        "--list-visits",
        action="store_true",
        help="List all visits and exit",
    )
    args = parser.parse_args()

    schedule = _load_schedule(args.schedule)

    if args.data_dir is None:
        # Infer data dir from schedule path
        args.data_dir = args.schedule.parent / "data"

    if args.list_visits:
        print(f"{'Idx':>4}  {'Target':30s}  {'Start':20s}  {'Stop':20s}  {'Duration':>8}")
        print("-" * 90)
        for i, row in schedule.iterrows():
            dur = (row["Observation Stop"] - row["Observation Start"]).total_seconds() / 60
            print(
                f"{i:4d}  {str(row['Target']):30s}  "
                f"{row['Observation Start']!s:20s}  "
                f"{row['Observation Stop']!s:20s}  "
                f"{dur:7.1f}m"
            )
        sys.exit(0)

    visit_idx: int | None = args.visit

    if visit_idx is None and args.target is not None:
        matches = schedule[schedule["Target"] == args.target]
        if matches.empty:
            # Try partial match
            matches = schedule[schedule["Target"].str.contains(args.target, na=False)]
        if matches.empty:
            print(f"No visit found for target '{args.target}'", file=sys.stderr)
            sys.exit(1)
        visit_idx = int(matches.index[0])
        print(f"Found target '{args.target}' at visit index {visit_idx}")
        print()

    if visit_idx is None:
        parser.error("Specify --visit or --target")

    if visit_idx < 0 or visit_idx >= len(schedule):
        print(f"Visit index {visit_idx} out of range [0, {len(schedule)-1}]", file=sys.stderr)
        sys.exit(1)

    occ_list = _load_occultation_list(args.data_dir)

    debug_visit(schedule, visit_idx, args.data_dir, occ_list=occ_list)


if __name__ == "__main__":
    main()
