#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time
from tqdm import tqdm

from pandorascheduler_rework.science_calendar import (
    _extract_visibility_segment,
    _occultation_windows,
    _read_visibility,
    _visibility_change_indices,
)
from pandorascheduler_rework.utils.io import (
    load_visibility_arrays_cached,
    resolve_star_visibility_file,
)


def split_occ_windows(
    occ_starts: list, occ_stops: list, occ_sequence_limit_min: int
) -> list[tuple]:
    step = timedelta(minutes=occ_sequence_limit_min + 1)
    out: list[tuple] = []
    for start, end in zip(occ_starts, occ_stops):
        current = start
        while current < end:
            nxt = min(current + step, end)
            out.append((current, nxt))
            current = nxt
    return out


def build_candidate_cache(
    csv_path: Path, vis_root: Path
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    df = pd.read_csv(csv_path)
    names = df.get("Star Name", pd.Series(dtype=object)).dropna().astype(str).tolist()
    cache: list[tuple[str, np.ndarray, np.ndarray]] = []
    iterator = tqdm(names, desc=f"Loading {csv_path.name}", leave=False)
    for name in iterator:
        vis_file = resolve_star_visibility_file(vis_root, name)
        if vis_file is None:
            continue
        arrays = load_visibility_arrays_cached(vis_file)
        if arrays is None:
            continue
        times_mjd, visible = arrays
        cache.append((name, times_mjd, visible))
    return cache


def best_coverage_for_interval(
    start, stop, cache: list[tuple[str, np.ndarray, np.ndarray]]
) -> tuple[float, str | None]:
    start_mjd = Time(start, format="datetime", scale="utc").to_value("mjd")
    stop_mjd = Time(stop, format="datetime", scale="utc").to_value("mjd")

    best_frac = 0.0
    best_name: str | None = None
    for name, times_mjd, visible in cache:
        interval_mask = (times_mjd >= start_mjd) & (times_mjd <= stop_mjd)
        if interval_mask.sum() == 0:
            continue
        frac = float((visible[interval_mask] == 1).sum()) / float(interval_mask.sum())
        if frac > best_frac:
            best_frac = frac
            best_name = name
            if best_frac >= 1.0:
                break
    return best_frac, best_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug occultation chunk coverage")
    parser.add_argument(
        "--schedule-csv",
        type=Path,
        default=Path(
            "output_directory/Pandora_Schedule_0.8_0.0_0.2_2026-03-13_to_2026-06-12.csv"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("output_directory/data_91_25_96"))
    parser.add_argument("--visit-start", type=str, required=True)
    parser.add_argument(
        "--visit-stop",
        type=str,
        default=None,
        help="Optional explicit visit stop (YYYY-MM-DD HH:MM:SS). If omitted, read from schedule CSV row.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional explicit target name. If omitted, read from schedule CSV row.",
    )
    parser.add_argument("--occ-sequence-limit-min", type=int, default=20)
    parser.add_argument("--min-sequence-minutes", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = pd.to_datetime(args.visit_start).to_pydatetime()
    if args.visit_stop is None or args.target is None:
        sched = pd.read_csv(args.schedule_csv)
        row_match = sched.loc[sched["Observation Start"] == args.visit_start]
        if row_match.empty:
            print(f"No schedule row with Observation Start={args.visit_start}")
            return 1
        row = row_match.iloc[0]
        target = str(args.target) if args.target is not None else str(row["Target"])
        stop = (
            pd.to_datetime(args.visit_stop).to_pydatetime()
            if args.visit_stop is not None
            else pd.to_datetime(row["Observation Stop"]).to_pydatetime()
        )
    else:
        target = str(args.target)
        stop = pd.to_datetime(args.visit_stop).to_pydatetime()

    star = target[:-1] if target.endswith(tuple("bcdef")) and target != "EV_Lac" else target

    print(f"Visit: {target}  {start} -> {stop}")
    vis = _read_visibility(args.data_dir / "targets" / star, star)
    if vis is None:
        print(f"Primary star visibility missing for {star}")
        return 1

    visit_times, vis_flags = _extract_visibility_segment(
        vis, start, stop, args.min_sequence_minutes
    )
    changes = _visibility_change_indices(vis_flags)
    occ_starts, occ_stops, _ = _occultation_windows(visit_times, vis_flags, changes)
    chunks = split_occ_windows(occ_starts, occ_stops, args.occ_sequence_limit_min)
    print(f"Occultation chunks: {len(chunks)}")

    occ_cache = build_candidate_cache(
        args.data_dir / "occultation-standard_targets.csv",
        args.data_dir / "aux_targets",
    )
    target_cache = build_candidate_cache(
        args.data_dir / "exoplanet_targets.csv",
        args.data_dir / "targets",
    )
    print(f"occ-list candidates loaded: {len(occ_cache)}")
    print(f"target-list candidates loaded: {len(target_cache)}")

    zero_both: list[int] = []
    iterator = tqdm(enumerate(chunks), total=len(chunks), desc="Evaluating chunks")
    for i, (chunk_start, chunk_stop) in iterator:
        occ_frac, occ_name = best_coverage_for_interval(chunk_start, chunk_stop, occ_cache)
        tgt_frac, tgt_name = best_coverage_for_interval(
            chunk_start, chunk_stop, target_cache
        )
        best_any = max(occ_frac, tgt_frac)
        print(
            f"{i:02d} {chunk_start} -> {chunk_stop}  "
            f"occ={occ_frac*100:5.1f}% ({occ_name})  "
            f"target={tgt_frac*100:5.1f}% ({tgt_name})  "
            f"best_any={best_any*100:5.1f}%"
        )
        if best_any <= 0.0:
            zero_both.append(i)

    if zero_both:
        print(f"Problematic chunk indices (0% in both lists): {zero_both}")
    else:
        print("No fully-zero chunks; each chunk has some candidate coverage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
