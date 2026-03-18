"""Tests for science_calendar._ScienceCalendarBuilder internals."""

from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from pandorascheduler_rework.config import PandoraSchedulerConfig
from pandorascheduler_rework.science_calendar import (
    ScienceCalendarInputs,
    _ScienceCalendarBuilder,
    _extract_visibility_segment,
    _is_transit_entry,
    _lookup_auxiliary_row,
    _lookup_planet_row,
    _normalise_target_name,
    _parse_datetime,
    _read_catalog,
    _visibility_change_indices,
    generate_science_calendar,
)


def _seed_catalogs(data_dir: Path) -> None:
    """Create minimal catalog CSVs so _ScienceCalendarBuilder.__init__ succeeds."""
    data_dir.mkdir(exist_ok=True)
    pd.DataFrame({"Planet Name": [], "Star Name": []}).to_csv(
        data_dir / "exoplanet_targets.csv", index=False
    )
    pd.DataFrame({"Star Name": []}).to_csv(
        data_dir / "all_targets.csv", index=False
    )
    pd.DataFrame({"Star Name": []}).to_csv(
        data_dir / "occultation-standard_targets.csv", index=False
    )


# ---------------------------------------------------------------------------
# _ScienceCalendarBuilder.__init__ / build_calendar
# ---------------------------------------------------------------------------


class TestBuilderInit:
    """Tests for _ScienceCalendarBuilder initialization."""

    def _make_schedule(self, tmp_path, rows=None):
        csv = tmp_path / "sched.csv"
        if rows is None:
            rows = [
                {
                    "Target": "Free Time",
                    "Observation Start": "2026-03-01T00:00:00Z",
                    "Observation Stop": "2026-03-01T06:00:00Z",
                }
            ]
        pd.DataFrame(rows).to_csv(csv, index=False)
        return csv

    def test_raises_on_missing_csv(self, tmp_path):
        _seed_catalogs(tmp_path)
        inputs = ScienceCalendarInputs(
            schedule_csv=tmp_path / "nope.csv",
            data_dir=tmp_path,
        )
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 3, 1),
            window_end=datetime(2026, 4, 1),
        )
        with pytest.raises(FileNotFoundError, match="Schedule CSV missing"):
            _ScienceCalendarBuilder(inputs, config)

    def test_raises_on_empty_csv(self, tmp_path):
        _seed_catalogs(tmp_path)
        csv = tmp_path / "empty.csv"
        pd.DataFrame(columns=["Target"]).to_csv(csv, index=False)
        inputs = ScienceCalendarInputs(schedule_csv=csv, data_dir=tmp_path)
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 3, 1),
            window_end=datetime(2026, 4, 1),
        )
        with pytest.raises(ValueError, match="empty"):
            _ScienceCalendarBuilder(inputs, config)

    def test_init_loads_schedule(self, tmp_path):
        _seed_catalogs(tmp_path)
        csv = self._make_schedule(tmp_path)
        inputs = ScienceCalendarInputs(schedule_csv=csv, data_dir=tmp_path)
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 3, 1),
            window_end=datetime(2026, 4, 1),
        )
        builder = _ScienceCalendarBuilder(inputs, config)
        assert builder.schedule is not None
        assert len(builder.schedule) == 1


# ---------------------------------------------------------------------------
# _add_meta
# ---------------------------------------------------------------------------


class TestAddMeta:
    """Tests for XML Meta element construction."""

    def test_meta_contains_expected_attributes(self, tmp_path):
        _seed_catalogs(tmp_path)
        csv = tmp_path / "sched.csv"
        pd.DataFrame(
            [
                {
                    "Target": "Star b",
                    "Observation Start": "2026-03-01T00:00:00Z",
                    "Observation Stop": "2026-03-02T00:00:00Z",
                }
            ]
        ).to_csv(csv, index=False)

        inputs = ScienceCalendarInputs(schedule_csv=csv, data_dir=tmp_path)
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 3, 1),
            window_end=datetime(2026, 4, 1),
            created_timestamp="2026-03-01T00:00:00Z",
            author="test",
        )
        builder = _ScienceCalendarBuilder(inputs, config)

        root = ET.Element("ScienceCalendar")
        builder._add_meta(root)

        meta = root.find("Meta")
        assert meta is not None
        assert meta.get("Valid_From") == "2026-03-01T00:00:00Z"
        assert meta.get("Created") == "2026-03-01T00:00:00Z"
        assert meta.get("Author") == "test"
        assert "Calendar_Weights" in meta.attrib


# ---------------------------------------------------------------------------
# _read_catalog
# ---------------------------------------------------------------------------


class TestReadCatalog:
    """Tests for catalog reading."""

    def test_raises_for_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Required target catalog missing"):
            _read_catalog(tmp_path / "nope.csv")

    def test_reads_existing_csv(self, tmp_path):
        path = tmp_path / "cat.csv"
        pd.DataFrame({"Star Name": ["Alpha"]}).to_csv(path, index=False)
        result = _read_catalog(path)
        assert result is not None
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_normalise_target_name_planet(self):
        name, star = _normalise_target_name("TOI-700 b")
        assert name == "TOI-700 b"
        assert star == "TOI-700"

    def test_normalise_target_name_no_suffix(self):
        name, star = _normalise_target_name("Vega")
        assert name == "Vega"
        assert star == "Vega"

    def test_parse_datetime_iso(self):
        dt = _parse_datetime("2026-03-01T12:30:00Z")
        assert dt is not None
        assert dt.hour == 12

    def test_parse_datetime_none(self):
        assert _parse_datetime(None) is None

    def test_parse_datetime_already_datetime(self):
        dt = datetime(2026, 3, 1, 12, 30)
        assert _parse_datetime(dt) == dt

    def test_is_transit_entry_true(self):
        row = pd.Series({"Transit Coverage": 0.85})
        assert _is_transit_entry(row) == True

    def test_is_transit_entry_false(self):
        row = pd.Series({"Transit Coverage": None})
        assert _is_transit_entry(row) == False

    def test_is_transit_entry_numeric_1(self):
        row = pd.Series({"Transit Coverage": 1.0})
        assert _is_transit_entry(row) == True

    def test_lookup_planet_row_found(self):
        cat = pd.DataFrame({"Planet Name": ["TOI-700 b"], "RA": [120.0]})
        result = _lookup_planet_row(cat, "TOI-700 b")
        assert result is not None
        assert result["RA"].iloc[0] == 120.0

    def test_lookup_planet_row_not_found(self):
        cat = pd.DataFrame({"Planet Name": ["TOI-700 b"], "RA": [120.0]})
        assert _lookup_planet_row(cat, "NOPE") is None

    def test_lookup_planet_row_empty_catalog(self):
        cat = pd.DataFrame({"Planet Name": [], "RA": []})
        assert _lookup_planet_row(cat, "anything") is None

    def test_lookup_auxiliary_row_found(self):
        cat = pd.DataFrame({"Star Name": ["Vega"], "DEC": [38.7]})
        result = _lookup_auxiliary_row(cat, "Vega")
        assert result is not None

    def test_lookup_auxiliary_row_not_found(self):
        cat = pd.DataFrame({"Star Name": ["Vega"]})
        assert _lookup_auxiliary_row(cat, "Sirius") is None


# ---------------------------------------------------------------------------
# Visibility segment extraction
# ---------------------------------------------------------------------------


class TestExtractVisibilitySegment:
    """Tests for _extract_visibility_segment."""

    def test_basic_split(self):
        times = pd.to_datetime(
            ["2026-03-01 00:00", "2026-03-01 00:01", "2026-03-01 00:02",
             "2026-03-01 00:03", "2026-03-01 00:04"]
        )
        vis = pd.DataFrame({
            "Time_UTC": times,
            "Time(MJD_UTC)": [61370.0 + i / 1440 for i in range(5)],
            "Visible": [1, 1, 0, 1, 1],
        })
        start = datetime(2026, 3, 1, 0, 0)
        stop = datetime(2026, 3, 1, 0, 4)
        visit_times, flags = _extract_visibility_segment(vis, start, stop, min_sequence_minutes=1)
        assert len(visit_times) >= 1
        assert len(visit_times) == len(flags)

    def test_empty_when_no_visible(self):
        times = pd.to_datetime(["2026-03-01 00:00", "2026-03-01 00:01"])
        vis = pd.DataFrame({
            "Time_UTC": times,
            "Time(MJD_UTC)": [61370.0, 61370.0 + 1 / 1440],
            "Visible": [0, 0],
        })
        visit_times, flags = _extract_visibility_segment(
            vis, datetime(2026, 3, 1), datetime(2026, 3, 1, 0, 1),
            min_sequence_minutes=1,
        )
        # Times are returned but all flags should be 0 (not visible)
        assert all(f == 0 for f in flags)


class TestVisibilityChangeIndices:
    """Tests for _visibility_change_indices."""

    def test_all_visible(self):
        indices = _visibility_change_indices(np.array([1, 1, 1, 1]))
        assert isinstance(indices, list)

    def test_alternating(self):
        indices = _visibility_change_indices(np.array([1, 0, 1, 0]))
        # Should detect each transition
        assert len(indices) >= 2


# ---------------------------------------------------------------------------
# Full calendar generation (smoke test)
# ---------------------------------------------------------------------------


class TestGenerateScienceCalendar:
    """Smoke test: full generate_science_calendar with free-time-only schedule."""

    def test_free_time_only_schedule(self, tmp_path):
        """Schedule with only 'Free Time' produces a valid but visit-less XML."""
        csv = tmp_path / "sched.csv"
        pd.DataFrame(
            [
                {
                    "Target": "Free Time",
                    "Observation Start": "2026-03-01T00:00:00Z",
                    "Observation Stop": "2026-03-01T06:00:00Z",
                    "Primary Target": False,
                }
            ]
        ).to_csv(csv, index=False)

        data_dir = tmp_path / "data"
        _seed_catalogs(data_dir)

        inputs = ScienceCalendarInputs(schedule_csv=csv, data_dir=data_dir)
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 3, 1),
            window_end=datetime(2026, 4, 1),
        )

        out = generate_science_calendar(inputs, config)
        assert out.exists()
        tree = ET.parse(out)
        root = tree.getroot()
        # The XML uses a namespace; use wildcard to find elements
        ns = {"ns": "/pandora/calendar/"}
        meta = root.find("ns:Meta", ns)
        assert meta is not None


# ---------------------------------------------------------------------------
# _merged_segments
# ---------------------------------------------------------------------------


class TestMergedSegments:
    """Tests for _ScienceCalendarBuilder._merged_segments."""

    def _make_builder(self, tmp_path, min_seq_min=10):
        _seed_catalogs(tmp_path)
        csv = tmp_path / "sched.csv"
        pd.DataFrame(
            [
                {
                    "Target": "Star b",
                    "Observation Start": "2026-03-01T00:00:00Z",
                    "Observation Stop": "2026-03-02T00:00:00Z",
                }
            ]
        ).to_csv(csv, index=False)
        inputs = ScienceCalendarInputs(
            schedule_csv=csv, data_dir=tmp_path
        )
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 3, 1),
            window_end=datetime(2026, 4, 1),
            min_sequence_minutes=min_seq_min,
        )
        return _ScienceCalendarBuilder(inputs, config)

    def test_short_segment_absorbed(self, tmp_path):
        """A 2-minute segment between two long segments is absorbed."""
        builder = self._make_builder(tmp_path, min_seq_min=10)
        t0 = datetime(2026, 3, 1, 0, 0)
        # 24 vis, 2 novis, 16 vis, 57 novis → 100 minute-resolution
        flags = [1] * 24 + [0] * 2 + [1] * 16 + [0] * 58
        times = [t0 + timedelta(minutes=i) for i in range(100)]
        changes = _visibility_change_indices(flags)

        raw = list(
            _ScienceCalendarBuilder._iterate_segments(
                changes, times, flags, t0, times[-1],
            )
        )
        merged = list(
            builder._merged_segments(
                changes, times, flags, t0, times[-1],
            )
        )
        # Raw has a short 2-min non-visible segment
        durations_raw = [
            (e - s).total_seconds() / 60 for s, e, v in raw
        ]
        assert any(d < 10 for d in durations_raw)
        # Merged has no segments shorter than threshold
        durations_merged = [
            (e - s).total_seconds() / 60 for s, e, v in merged
        ]
        assert all(d >= 10 for d in durations_merged)

    def test_no_merge_when_disabled(self, tmp_path):
        """min_sequence_minutes=0 disables merging."""
        builder = self._make_builder(tmp_path, min_seq_min=0)
        t0 = datetime(2026, 3, 1, 0, 0)
        flags = [1] * 5 + [0] * 2 + [1] * 5
        times = [t0 + timedelta(minutes=i) for i in range(12)]
        changes = _visibility_change_indices(flags)
        merged = list(
            builder._merged_segments(
                changes, times, flags, t0, times[-1],
            )
        )
        raw = list(
            _ScienceCalendarBuilder._iterate_segments(
                changes, times, flags, t0, times[-1],
            )
        )
        assert len(merged) == len(raw)
