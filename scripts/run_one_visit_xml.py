#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from pandorascheduler_rework.config import PandoraSchedulerConfig
from pandorascheduler_rework.science_calendar import (
    ScienceCalendarInputs,
    generate_science_calendar,
)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "":
            return default
        return normalized in {"1", "true", "yes", "y", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _parse_datetime(value: str) -> datetime:
    for pattern in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            continue
    raise ValueError(f"Unsupported datetime format: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate science calendar XML for one schedule row."
    )
    parser.add_argument(
        "--schedule-csv",
        type=Path,
        required=True,
        help="Path to Pandora_Schedule_*.csv",
    )
    parser.add_argument(
        "--visit-start",
        type=str,
        required=True,
        help="Observation Start to select (YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to example_scheduler_config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_directory"),
        help="Base output directory containing data_<sun>_<moon>_<earth>",
    )
    parser.add_argument(
        "--out-xml",
        type=Path,
        default=Path("output_directory/debug_one_visit.xml"),
        help="Output XML path",
    )
    parser.add_argument(
        "--tmp-schedule",
        type=Path,
        default=Path("output_directory/debug_one_visit.csv"),
        help="Temporary one-row schedule CSV path",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()

    cfg_json = json.loads(args.config.read_text(encoding="utf-8"))

    schedule = pd.read_csv(args.schedule_csv)
    row = schedule.loc[schedule["Observation Start"] == args.visit_start]
    if row.empty:
        raise ValueError(f"No row found with Observation Start={args.visit_start!r}")
    one_row = row.head(1).copy()

    args.tmp_schedule.parent.mkdir(parents=True, exist_ok=True)
    one_row.to_csv(args.tmp_schedule, index=False)

    sun = float(cfg_json.get("visibility_sun_deg", cfg_json.get("sun_avoidance_deg", 91.0)))
    moon = float(cfg_json.get("visibility_moon_deg", cfg_json.get("moon_avoidance_deg", 25.0)))
    earth = float(cfg_json.get("visibility_earth_deg", cfg_json.get("earth_avoidance_deg", 86.0)))
    data_dir = args.output_dir / f"data_{int(sun)}_{int(moon)}_{int(earth)}"

    # Keep config aligned with run_scheduler parsing semantics.
    enable_occ_xml = _as_bool(
        cfg_json.get("generate_occultation_xml", cfg_json.get("enable_occultation_xml", True)),
        True,
    )
    enable_pass1 = _as_bool(
        cfg_json.get("one_occultation_target", cfg_json.get("enable_occultation_pass1", True)),
        True,
    )
    strict_limits = _as_bool(cfg_json.get("strict_occultation_time_limits", True), True)

    # Window values are only metadata for this script; one-row schedule drives the XML content.
    row_start = _parse_datetime(str(one_row.iloc[0]["Observation Start"]))
    row_stop = _parse_datetime(str(one_row.iloc[0]["Observation Stop"]))

    config = PandoraSchedulerConfig(
        window_start=row_start,
        window_end=row_stop,
        schedule_step=timedelta(hours=float(cfg_json.get("schedule_step_hours", 24.0))),
        output_dir=args.output_dir,
        sun_avoidance_deg=sun,
        moon_avoidance_deg=moon,
        earth_avoidance_deg=earth,
        obs_sequence_duration_min=int(cfg_json.get("obs_sequence_duration_min", 90)),
        occ_sequence_limit_min=int(cfg_json.get("occ_sequence_limit_min", 50)),
        min_sequence_minutes=int(cfg_json.get("min_sequence_minutes", 5)),
        break_occultation_sequences=bool(cfg_json.get("break_occultation_sequences", True)),
        use_target_list_for_occultations=bool(cfg_json.get("use_target_list_for_occultations", False)),
        prioritise_occultations_by_slew=bool(cfg_json.get("prioritise_occultations_by_slew", False)),
        enable_occultation_xml=enable_occ_xml,
        enable_occultation_pass1=enable_pass1,
        strict_occultation_time_limits=strict_limits,
        primary_only_mode=_as_bool(cfg_json.get("primary_only_mode", False), False),
        show_progress=True,
    )

    inputs = ScienceCalendarInputs(schedule_csv=args.tmp_schedule, data_dir=data_dir)
    args.out_xml.parent.mkdir(parents=True, exist_ok=True)
    xml_path = generate_science_calendar(inputs=inputs, config=config, output_path=args.out_xml)

    print(f"One-row schedule: {args.tmp_schedule}")
    print(f"Data dir: {data_dir}")
    print(f"XML: {xml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
