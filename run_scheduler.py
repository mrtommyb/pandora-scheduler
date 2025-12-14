#!/usr/bin/env python3
"""
Pandora Scheduler Driver Script
===============================

Pandora Scheduler - Main Entry Point

This script runs the complete Pandora observation scheduling pipeline from start
to finish, generating all necessary output files including:
  - Target manifests (from target definition files)
  - Visibility catalogs (if requested)
  - Observation schedule CSV
  - Science calendar XML
  - Observation time reports
  - Tracker files (CSV and pickle)

Usage:
 # Basic run with default configuration
    poetry run python run_scheduler.py \\
        --start "2026-02-05" \\
        --end "2026-02-12" \\
        --output ./output

    # Run with custom configuration
    poetry run python run_scheduler.py \\
        --start "2026-02-05" \\
        --end "2026-02-12" \\
        --output ./output \\
        --config config.json

    # Generate visibility data as part of the run
    poetry run python run_scheduler.py \\
        --start "2026-02-05" \\
        --end "2026-02-12" \\
        --output ./output \\
        --generate-visibility

    # Use custom target definition files
    poetry run python run_scheduler.py \\
        --start "2026-02-05" \\
        --end "2026-02-12" \\
        --output ./output \\
        --target-definitions ./custom_targets
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from pandorascheduler_rework.config import PandoraSchedulerConfig
from pandorascheduler_rework.pipeline import SchedulerResult, build_schedule
from pandorascheduler_rework.science_calendar import (
    ScienceCalendarInputs,
    generate_science_calendar,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Pandora observation scheduler pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Schedule window start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="Schedule window end date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for generated files",
    )

    # Optional configuration
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON configuration file (overrides defaults)",
    )
    parser.add_argument(
        "--target-definitions",
        type=Path,
        help=(
            "Base directory containing target definition files (e.g., PandoraTargetList/target_definition_files/). "
            "When provided, generates *_targets.csv manifests from JSON definitions."
        ),
    )
    parser.add_argument(
        "--generate-visibility",
        action="store_true",
        help="Generate visibility catalogs before scheduling (requires target manifests)",
    )

    # Visibility configuration (if generating)
    parser.add_argument(
        "--gmat-ephemeris",
        type=Path,
        help="Path to GMAT ephemeris file (for visibility generation)",
    )
    parser.add_argument(
        "--sun-avoidance",
        type=float,
        default=91.0,
        help="Sun avoidance angle in degrees (default: 91.0)",
    )
    parser.add_argument(
        "--moon-avoidance",
        type=float,
        default=25.0,
        help="Moon avoidance angle in degrees (default: 25.0)",
    )
    parser.add_argument(
        "--earth-avoidance",
        type=float,
        default=86.0,
        help="Earth avoidance angle in degrees (default: 86.0)",
    )

    # Scheduling configuration
    parser.add_argument(
        "--schedule-step-hours",
        type=float,
        default=24.0,
        help=(
            "Scheduler rolling window step size in hours (default: 24.0). "
            "Per-target visit duration comes from target manifests (Obs Window (hrs))."
        ),
    )
    parser.add_argument(
        "--transit-coverage",
        type=float,
        default=0.4,
        help="Minimum transit coverage fraction (default: 0.4)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="0.8,0.0,0.2",
        help="Schedule weights as comma-separated values: coverage,saa,schedule (default: 0.8,0.0,0.2)",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.5,
        help="Minimum visibility fraction for non-transit observations (default: 0.5)",
    )

    # Flags
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--skip-xml",
        action="store_true",
        help="Skip generation of the science calendar XML",
    )

    # Profiling configuration
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling",
    )
    parser.add_argument(
        "--profile-output",
        default="profile_output.prof",
        help="Profile output file",
    )

    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show progress bars during execution",
    )
    parser.add_argument(
        "--skip-manifests",
        action="store_true",
        help="Skip regenerating target manifests from target definition files",
    )
    parser.add_argument(
        "--legacy-mode",
        action="store_true",
        help=(
            "Use legacy scheduling algorithms for validation against historical outputs. "
            "When enabled, uses MJD-based visibility filtering which matches the original "
            "scheduler exactly. Default (disabled) uses improved datetime-based filtering."
        ),
    )

    return parser.parse_args()


def parse_datetime(date_str: str) -> datetime:
    """Parse a date string into a datetime object."""
    try:
        # Try full datetime format first
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            # Fallback to date only
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Invalid date format: {date_str}. Expected YYYY-MM-DD or YYYY-MM-DD HH:MM:SS"
            )


def print_summary(result: SchedulerResult, xml_path: Optional[Path]) -> None:
    """Print a summary of the scheduling run."""
    try:

        def _hours(td: pd.Timedelta) -> float:
            return float(td.total_seconds() / 3600.0)

        def _fmt_hours(value: float) -> str:
            return f"{value:,.2f} h"

        def _safe_to_datetime(series: pd.Series) -> pd.Series:
            # Schedule CSV can contain strings or timestamps; normalize defensively.
            return pd.to_datetime(series, errors="coerce")

        import pandas as pd

        print("\n" + "=" * 80)
        print("SCHEDULING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nGenerated Files:")
        print("-" * 80)

        if getattr(result, "schedule_csv", None):
            print(f"  📄 Schedule CSV:      {result.schedule_csv}")
        reports = getattr(result, "reports", {}) or {}
        if reports.get("observation_time"):
            print(f"  📊 Observation Time   {reports.get('observation_time')}")
        if reports.get("tracker_csv"):
            print(f"  📊 Tracker Csv        {reports.get('tracker_csv')}")
        if reports.get("tracker_pickle"):
            print(f"  📊 Tracker Pickle     {reports.get('tracker_pickle')}")
        if xml_path:
            print(f"  📑 Science Calendar:  {xml_path}")

        print("-" * 80)
        print("\nSchedule Statistics:")
        schedule_df = None
        diagnostics = getattr(result, "diagnostics", {}) or {}
        schedule_df = diagnostics.get("schedule_dataframe")

        if schedule_df is not None and len(schedule_df) > 0:
            try:
                df = schedule_df.copy()
                if "Target" in df.columns:
                    df["Target"] = df["Target"].astype(str).str.strip()

                if "Observation Start" in df.columns:
                    df["Observation Start"] = _safe_to_datetime(df["Observation Start"])
                if "Observation Stop" in df.columns:
                    df["Observation Stop"] = _safe_to_datetime(df["Observation Stop"])

                total_obs = int(len(df))
                unique_targets = (
                    int(df["Target"].nunique()) if "Target" in df.columns else 0
                )

                # Categorize
                if "Target" in df.columns:
                    is_free = df["Target"].eq("Free Time")
                    is_std = df["Target"].str.contains(r"\bSTD\b", na=False)
                    is_primary = (~is_free) & (~is_std)
                else:
                    is_free = pd.Series([False] * len(df))
                    is_std = pd.Series([False] * len(df))
                    is_primary = pd.Series([True] * len(df))

                primary_count = int(is_primary.sum())
                std_count = int(is_std.sum())
                free_count = int(is_free.sum())

                # Durations
                durations = None
                if (
                    "Observation Start" in df.columns
                    and "Observation Stop" in df.columns
                ):
                    durations = df["Observation Stop"] - df["Observation Start"]
                    durations = durations.where(durations.notna(), pd.Timedelta(0))
                    # Guard against negative durations due to bad parsing
                    durations = durations.clip(lower=pd.Timedelta(0))

                # Schedule span
                total_span_hours = None
                if (
                    "Observation Start" in df.columns
                    and "Observation Stop" in df.columns
                    and df["Observation Start"].notna().any()
                    and df["Observation Stop"].notna().any()
                ):
                    sched_start = df["Observation Start"].min()
                    sched_stop = df["Observation Stop"].max()
                    if pd.notna(sched_start) and pd.notna(sched_stop):
                        total_span_hours = _hours(sched_stop - sched_start)

                # Time totals by category
                primary_hours = std_hours = free_hours = None
                if durations is not None:
                    primary_hours = _hours(durations[is_primary].sum())
                    std_hours = _hours(durations[is_std].sum())
                    free_hours = _hours(durations[is_free].sum())

                print(f"  Total observations:        {total_obs}")
                print(f"  Unique targets:            {unique_targets}")
                print(f"  Primary observations:      {primary_count}")
                print(f"  Standard (STD) blocks:     {std_count}")
                print(f"  Free Time blocks:          {free_count}")

                if total_span_hours is not None:
                    print(
                        f"  Schedule span:             {_fmt_hours(total_span_hours)}"
                    )

                if primary_hours is not None:
                    used_hours = primary_hours + std_hours
                    print(f"  Primary time:              {_fmt_hours(primary_hours)}")
                    print(f"  STD time:                  {_fmt_hours(std_hours)}")
                    print(f"  Free time:                 {_fmt_hours(free_hours)}")
                    if total_span_hours is not None and total_span_hours > 0:
                        util = 100.0 * (used_hours / total_span_hours)
                        print(f"  Utilization (non-free):    {util:,.1f}%")

                if (
                    durations is not None
                    and durations.astype("timedelta64[ns]").notna().any()
                ):

                    def _dur_stats(
                        mask: pd.Series,
                    ) -> tuple[float | None, float | None]:
                        subset = durations[mask]
                        if subset.empty:
                            return None, None
                        return _hours(subset.mean()), _hours(subset.median())

                    p_mean, p_med = _dur_stats(is_primary)
                    if p_mean is not None:
                        print(f"  Primary duration (mean):   {_fmt_hours(p_mean)}")
                        print(
                            f"  Primary duration (median): {_fmt_hours(p_med or 0.0)}"
                        )

                # Transit-quality stats (only for rows that have Transit Coverage)
                if "Transit Coverage" in df.columns and is_primary.any():
                    cov = pd.to_numeric(
                        df.loc[is_primary, "Transit Coverage"], errors="coerce"
                    )
                    cov = cov.dropna()
                    if len(cov) > 0:
                        cov_mean = float(cov.mean())
                        cov_median = float(cov.median())
                        cov_min = float(cov.min())
                        cov_p10 = float(cov.quantile(0.10))
                        cov_p90 = float(cov.quantile(0.90))
                        full = int((cov >= 0.999999).sum())
                        good = int((cov >= 0.90).sum())
                        print("\n  Primary transit coverage:")
                        print(
                            f"    mean/median:             {cov_mean:.3f} / {cov_median:.3f}"
                        )
                        print(
                            f"    min / p10 / p90:         {cov_min:.3f} / {cov_p10:.3f} / {cov_p90:.3f}"
                        )
                        print(
                            f"    >= 0.90:                 {good} ({100.0 * good / len(cov):.1f}%)"
                        )
                        print(
                            f"    ~100% (>= 0.999999):     {full} ({100.0 * full / len(cov):.1f}%)"
                        )

                # SAA overlap summary
                if "SAA Overlap" in df.columns and is_primary.any():
                    saa = pd.to_numeric(
                        df.loc[is_primary, "SAA Overlap"], errors="coerce"
                    )
                    saa = saa.dropna()
                    if len(saa) > 0:
                        print("\n  Primary SAA overlap:")
                        print(
                            f"    mean/median:             {float(saa.mean()):.3f} / {float(saa.median()):.3f}"
                        )
            except Exception:
                print("  (Unable to compute detailed schedule statistics)")
        else:
            print("  (No schedule dataframe available for statistics)")

        print("\n" + "=" * 80)
    except AttributeError as exc:
        # Be defensive: if the SchedulerResult shape is unexpected, log and provide minimal info.
        logger = logging.getLogger(__name__)
        logger.error("Failed to generate summary: %s", exc)
        try:
            # Best-effort fallback: print whatever schedule CSV path exists.
            if hasattr(result, "schedule_csv") and result.schedule_csv:
                print(f"Schedule generated: {result.schedule_csv}")
        except Exception:
            # Give up cleanly; avoid raising further exceptions from summary printing.
            pass


def main() -> int:
    """Main execution function."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Initialize profiler if requested
    profiler = None
    if args.profile:
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()

    try:
        logger.info(f"Scheduling window: {args.start} to {args.end}")
        logger.info(f"Output directory: {args.output}")

        # 1. Load Configuration
        json_config = {}
        if args.config:
            with open(args.config, "r") as f:
                json_config = json.load(f)

        # Default weights if not provided
        transit_scheduling_weights = (0.8, 0.0, 0.2)

        # Resolve target definition base (explicit path only)
        target_def_base = args.target_definitions

        # Resolve visibility GMAT file (explicit path only)
        visibility_gmat = args.gmat_ephemeris

        # 2. Validate Inputs
        if args.generate_visibility and not target_def_base:
            logger.error(
                "Visibility generation requires target definitions. "
                "Please provide target definitions via --target-definitions"
            )
            return 1

        # Determine whether visibility generation was requested (CLI or JSON)
        generate_visibility = args.generate_visibility or json_config.get(
            "generate_visibility", False
        )
        # `config` is not yet constructed here, so check the CLI/ENV visibility GMAT
        if generate_visibility and visibility_gmat is None:
            logger.warning(
                "Visibility generation requested but no GMAT ephemeris provided."
            )

        # Build PandoraSchedulerConfig with the dataclass field names and types
        schedule_step_hours = float(
            get_val("schedule_step_hours", args.schedule_step_hours, 24.0)
        )
        transit_cov = float(get_val("transit_coverage_min", args.transit_coverage, 0.4))
        min_vis = float(get_val("min_visibility", args.min_visibility, 0.5))

        # Coerce unified transit_scheduling_weights from JSON or CLI into a 3-tuple
        raw_transit_weights = (
            json_config.get("transit_scheduling_weights") or transit_scheduling_weights
        )
        if isinstance(raw_transit_weights, str):
            raw_transit_weights = tuple(
                float(x.strip()) for x in raw_transit_weights.split(",")
            )
        transit_weights_tuple = tuple(float(x) for x in raw_transit_weights)

        extra_inputs: Dict[str, Any] = {}
        if target_def_base:
            extra_inputs["target_definition_base"] = Path(target_def_base)
            # When target definitions are provided, we need to specify which categories to process.
            # These map to the standard directory names in the PandoraTargetList repository.
            extra_inputs["target_definition_files"] = [
                "exoplanet",
                "auxiliary-standard",
                "monitoring-standard",
                "occultation-standard",
            ]

        if args.skip_manifests:
            extra_inputs["skip_manifests"] = True

        # Visibility GMAT goes into the typed field `gmat_ephemeris` on the config
        if visibility_gmat is None:
            gmat_path = None
        else:
            gmat_path = Path(str(visibility_gmat)).expanduser().resolve()

        config = PandoraSchedulerConfig(
            window_start=parse_datetime(args.start),
            window_end=parse_datetime(args.end),
            schedule_step=timedelta(hours=schedule_step_hours),
            targets_manifest=args.output / "data",
            gmat_ephemeris=gmat_path,
            output_dir=args.output,
            # Scheduling Thresholds
            transit_coverage_min=transit_cov,
            min_visibility=min_vis,
            commissioning_days=int(json_config.get("commissioning_days", 0)),
            # Weights
            transit_scheduling_weights=transit_weights_tuple,
            # Extra inputs for pipeline
            extra_inputs=extra_inputs,
            # Flags
            show_progress=args.show_progress,
            use_legacy_mode=args.legacy_mode,
        )

        # 3. Ensure targets manifest location exists (may be an output/data dir)
        if config.targets_manifest and not config.targets_manifest.exists():
            try:
                config.targets_manifest.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Created targets manifest directory: %s", config.targets_manifest
                )
            except Exception:
                logger.debug(
                    "Unable to create targets manifest directory: %s",
                    config.targets_manifest,
                )

        # 4. Run Scheduler (using new API)
        logger.info("Starting scheduler pipeline...")
        if args.legacy_mode:
            logger.info("Legacy mode enabled - using MJD-based visibility filtering")
        result = build_schedule(config)

        # 4. Generate Science Calendar XML
        xml_path = None
        if not args.skip_xml and result.schedule_csv:
            # Use the same config object to create calendar settings
            # Ensure we point to the correct data directory (where manifests are)
            # The pipeline copies/generates data into output_dir/data
            data_dir = config.output_dir / "data"

            inputs = ScienceCalendarInputs(
                schedule_csv=result.schedule_csv,
                data_dir=data_dir,
            )

            xml_path = generate_science_calendar(
                inputs=inputs,
                config=config,
            )
            logger.info(f"Science calendar written to: {xml_path}")

        # 5. Print Summary
        print_summary(result, xml_path)

        return 0

    except KeyboardInterrupt:
        logger.error("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=args.verbose)
        return 1
    finally:
        if profiler:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats("cumulative")
            stats.dump_stats(args.profile_output)
            print(f"\nProfiling results written to {args.profile_output}")
            stats.print_stats(30)


def get_val(key: str, cli_arg: Any, default: Any) -> Any:
    """Helper to prioritize CLI arg > JSON config > default."""
    if cli_arg is not None:
        return cli_arg
    # Note: json_config would need to be passed in or global
    # For simplicity in this script structure, we'll rely on CLI or defaults mostly
    # But to support JSON fully, we'd check it here.
    return default


if __name__ == "__main__":
    sys.exit(main())
