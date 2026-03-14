"""Tests for the occultation config toggles added in Phase 1.

Covers:
  - Config defaults and explicit construction
  - strict_occultation_time_limits=False (relaxed mode)
  - enable_occultation_pass1=False (Pass 1 is skipped)
  - use_pass1 parameter in schedule_occultation_targets
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from astropy.time import Time

from pandorascheduler_rework import observation_utils, science_calendar
from pandorascheduler_rework.config import PandoraSchedulerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> PandoraSchedulerConfig:
    defaults = dict(
        window_start=datetime(2026, 1, 1),
        window_end=datetime(2026, 1, 2),
    )
    defaults.update(overrides)
    return PandoraSchedulerConfig(**defaults)


def _write_star_visibility(
    directory: Path,
    name: str,
    times: list[datetime],
    flags: list[int],
) -> None:
    """Write visibility parquet in the expected directory layout:
    directory / name / 'Visibility for {name}.parquet'
    """
    target_dir = directory / name
    target_dir.mkdir(parents=True, exist_ok=True)
    mjd_times = Time(times, scale="utc").to_value("mjd")
    pd.DataFrame({"Time(MJD_UTC)": mjd_times, "Visible": flags}).to_parquet(
        target_dir / f"Visibility for {name}.parquet", index=False,
    )


def _make_builder(tmp_path, config=None):
    """Return a _ScienceCalendarBuilder with minimal filesystem scaffolding."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"Star Name": [], "RA": [], "DEC": []}).to_csv(
        data_dir / "exoplanet_targets.csv", index=False,
    )
    pd.DataFrame({"Star Name": [], "RA": [], "DEC": []}).to_csv(
        data_dir / "all_targets.csv", index=False,
    )
    pd.DataFrame({
        "Star Name": ["OccA"],
        "RA": [10.0],
        "DEC": [20.0],
        "Number of Hours Requested": [600],
    }).to_csv(data_dir / "occultation-standard_targets.csv", index=False)

    schedule_df = pd.DataFrame([{
        "Target": "TestPlanet",
        "Observation Start": "2026-01-01 00:00:00",
        "Observation Stop": "2026-01-01 01:00:00",
        "Transit Coverage": 0.5,
        "SAA Overlap": 0.0,
        "Schedule Factor": 0.9,
        "Quality Factor": 0.8,
        "Comments": "",
    }])
    schedule_path = tmp_path / "schedule.csv"
    schedule_df.to_csv(schedule_path, index=False)

    inputs = science_calendar.ScienceCalendarInputs(
        schedule_csv=schedule_path,
        data_dir=data_dir,
    )
    if config is None:
        config = _make_config()
    return science_calendar._ScienceCalendarBuilder(inputs, config)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_defaults_are_true(self):
        cfg = _make_config()
        assert cfg.enable_occultation_xml is True
        assert cfg.enable_occultation_pass1 is True
        assert cfg.strict_occultation_time_limits is True

    def test_explicit_false(self):
        cfg = _make_config(
            enable_occultation_xml=False,
            enable_occultation_pass1=False,
            strict_occultation_time_limits=False,
        )
        assert cfg.enable_occultation_xml is False
        assert cfg.enable_occultation_pass1 is False
        assert cfg.strict_occultation_time_limits is False


# ---------------------------------------------------------------------------
# strict_occultation_time_limits
# ---------------------------------------------------------------------------

class TestRelaxedTimeLimits:
    """When strict_occultation_time_limits=False, errors become warnings."""

    def test_relaxed_returns_large_fallback_for_missing_target(self, tmp_path):
        config = _make_config(strict_occultation_time_limits=False)
        builder = _make_builder(tmp_path, config)
        result = builder._get_occultation_time_limit("UnknownTarget")
        # Should return a very large timedelta instead of raising
        assert result >= timedelta(hours=999_999)

    def test_strict_raises_for_missing_target(self, tmp_path):
        config = _make_config(strict_occultation_time_limits=True)
        builder = _make_builder(tmp_path, config)
        with pytest.raises(ValueError, match="not found in catalog"):
            builder._get_occultation_time_limit("UnknownTarget")

    def test_relaxed_returns_large_fallback_for_empty_catalog(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Star Name": [], "RA": [], "DEC": []}).to_csv(
            data_dir / "exoplanet_targets.csv", index=False,
        )
        pd.DataFrame({"Star Name": [], "RA": [], "DEC": []}).to_csv(
            data_dir / "all_targets.csv", index=False,
        )
        pd.DataFrame({
            "Star Name": [],
            "RA": [],
            "DEC": [],
            "Number of Hours Requested": [],
        }).to_csv(data_dir / "occultation-standard_targets.csv", index=False)

        schedule_df = pd.DataFrame([{
            "Target": "TestPlanet",
            "Observation Start": "2026-01-01 00:00:00",
            "Observation Stop": "2026-01-01 01:00:00",
            "Transit Coverage": 0.5,
            "SAA Overlap": 0.0,
            "Schedule Factor": 0.9,
            "Quality Factor": 0.8,
            "Comments": "",
        }])
        schedule_path = tmp_path / "schedule.csv"
        schedule_df.to_csv(schedule_path, index=False)

        inputs = science_calendar.ScienceCalendarInputs(
            schedule_csv=schedule_path,
            data_dir=data_dir,
        )
        config = _make_config(strict_occultation_time_limits=False)
        builder = science_calendar._ScienceCalendarBuilder(inputs, config)
        result = builder._get_occultation_time_limit("AnyTarget")
        assert result >= timedelta(hours=999_999)


# ---------------------------------------------------------------------------
# enable_occultation_pass1 → use_pass1 in schedule_occultation_targets
# ---------------------------------------------------------------------------

class TestUsePass1:
    """schedule_occultation_targets with use_pass1=False skips Pass 1."""

    def _run_schedule(self, tmp_path, use_pass1: bool):
        """Set up a simple case where a single star covers all intervals."""
        vis_dir = tmp_path / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # A star visible across the entire window
        t0 = datetime(2026, 1, 1)
        times = [t0 + timedelta(minutes=m) for m in range(120)]
        flags = [1] * 120
        _write_star_visibility(vis_dir, "StarA", times, flags)

        v_names = np.array(["StarA"])
        start_mjd = Time([t0], scale="utc").to_value("mjd")
        stop_mjd = Time([t0 + timedelta(hours=1)], scale="utc").to_value("mjd")

        o_df = pd.DataFrame({
            "Target": [None],
            "start": [t0.strftime("%Y-%m-%dT%H:%M:%SZ")],
            "stop": [(t0 + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")],
            "RA": [np.nan],
            "DEC": [np.nan],
        })
        o_list = pd.DataFrame({
            "Star Name": ["StarA"],
            "RA": [10.0],
            "DEC": [20.0],
        })

        result_df, flag = observation_utils.schedule_occultation_targets(
            v_names, start_mjd, stop_mjd,
            t0, t0 + timedelta(hours=1),
            str(vis_dir), o_df, o_list, "test",
            use_pass1=use_pass1,
        )
        return result_df, flag

    def test_pass1_true_assigns(self, tmp_path):
        result_df, flag = self._run_schedule(tmp_path, use_pass1=True)
        assert flag is True
        assert result_df["Target"].notna().any()

    def test_pass1_false_still_assigns_via_later_passes(self, tmp_path):
        """Even with Pass 1 skipped, Pass 2 should still assign the target."""
        result_df, flag = self._run_schedule(tmp_path, use_pass1=False)
        assert flag is True
        assert result_df["Target"].notna().any()

    def test_pass1_does_not_assign_when_interval_has_no_samples(self, tmp_path):
        """An empty interval must not be treated as fully visible by Pass 1."""
        vis_dir = tmp_path / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

        t0 = datetime(2026, 1, 1)
        times = [t0 + timedelta(minutes=m) for m in range(60)]
        flags = [1] * 60
        _write_star_visibility(vis_dir, "StarA", times, flags)

        # Interval one day later -> zero overlapping visibility samples
        start = t0 + timedelta(days=1)
        stop = start + timedelta(hours=1)

        v_names = np.array(["StarA"])
        start_mjd = Time([start], scale="utc").to_value("mjd")
        stop_mjd = Time([stop], scale="utc").to_value("mjd")

        o_df = pd.DataFrame({
            "Target": [None],
            "start": [start.strftime("%Y-%m-%dT%H:%M:%SZ")],
            "stop": [stop.strftime("%Y-%m-%dT%H:%M:%SZ")],
            "RA": [np.nan],
            "DEC": [np.nan],
        })
        o_list = pd.DataFrame({
            "Star Name": ["StarA"],
            "RA": [10.0],
            "DEC": [20.0],
        })

        result_df, flag = observation_utils.schedule_occultation_targets(
            v_names,
            start_mjd,
            stop_mjd,
            start,
            stop,
            str(vis_dir),
            o_df,
            o_list,
            "test",
            use_pass1=True,
        )

        assert flag is False
        assert (result_df["Target"] == "No target").all()
