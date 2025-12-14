"""Build science calendar XML from observation schedules.

This module generates Pandora science calendar XML files from CSV schedules.
It handles:
- Creating visit and observation sequence XML structure
- Managing occultation and transit observations
- Splitting long observations into sequences
- Integrating target parameters from manifest files
- Outputting properly formatted XML calendars

This is the refactored version of xml_builder.py with clearer naming
and improved documentation.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from numbers import Number
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from xml.dom import minidom

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
from tqdm import tqdm

from pandorascheduler_rework import observation_utils
from pandorascheduler_rework.config import PandoraSchedulerConfig
from pandorascheduler_rework.utils.array_ops import (
    break_long_sequences,
    remove_short_sequences,
)
from pandorascheduler_rework.utils.io import read_csv_cached, read_parquet_cached
from pandorascheduler_rework.xml import observation_sequence

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScienceCalendarInputs:
    """Filesystem pointers consumed by the XML builder."""

    schedule_csv: Path
    data_dir: Path


def generate_science_calendar(
    inputs: ScienceCalendarInputs,
    config: PandoraSchedulerConfig,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate the science calendar XML, matching the legacy behaviour."""

    builder = _ScienceCalendarBuilder(inputs, config)
    calendar_element = builder.build_calendar()
    xml_string = _serialise_calendar(calendar_element)

    destination = output_path or (inputs.data_dir / "Pandora_science_calendar.xml")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(xml_string, encoding="utf-8")
    return destination


class _ScienceCalendarBuilder:
    """Encapsulates the translation from CSV schedules to XML."""

    def __init__(
        self, inputs: ScienceCalendarInputs, config: PandoraSchedulerConfig
    ) -> None:
        self.inputs = inputs
        self.config = config
        self.schedule = read_csv_cached(str(inputs.schedule_csv))
        if self.schedule is None:
            raise FileNotFoundError(f"Schedule CSV missing: {inputs.schedule_csv}")

        if self.schedule.empty:
            raise ValueError("Schedule CSV is empty; nothing to convert into XML")

        self.data_dir = inputs.data_dir
        self.target_catalog = _read_catalog(self.data_dir / "exoplanet_targets.csv")
        self.aux_catalog = _read_catalog(self.data_dir / "all_targets.csv")
        self.occ_catalog = _read_catalog(
            self.data_dir / "occultation-standard_targets.csv"
        )

        obs_minutes, occ_minutes = observation_utils.general_parameters(
            config.obs_sequence_duration_min,
            config.occ_sequence_limit_min,
        )
        self.sequence_duration = timedelta(minutes=obs_minutes)
        self.occultation_limit = timedelta(minutes=occ_minutes + 1)

        # Track cumulative observation time for each occultation target
        self.occultation_obs_time: Dict[str, timedelta] = {}

    def _get_occultation_time_limit(self, target_name: str) -> timedelta:
        """Get the time limit for an occultation target.

        Looks up 'Number of Hours Requested' from the occultation manifest.
        Raises ValueError if the catalog is missing, the target is not found,
        or the required column is missing.
        """
        if self.occ_catalog is None or self.occ_catalog.empty:
            raise ValueError(
                f"Cannot get time limit for occultation target '{target_name}': "
                "occultation catalog is not loaded"
            )
        match = self.occ_catalog[self.occ_catalog["Star Name"] == target_name]
        if match.empty:
            raise ValueError(f"Occultation target '{target_name}' not found in catalog")
        if "Number of Hours Requested" not in match.columns:
            raise ValueError(
                "Occultation catalog is missing required 'Number of Hours Requested' "
                "column"
            )
        hours_req = match.iloc[0]["Number of Hours Requested"]
        if pd.isna(hours_req):
            raise ValueError(
                f"Occultation target '{target_name}' has missing "
                "'Number of Hours Requested' value"
            )
        return timedelta(hours=float(hours_req))

    def build_calendar(self) -> ET.Element:
        root = ET.Element("ScienceCalendar", xmlns="/pandora/calendar/")
        self._add_meta(root)

        visits = self.schedule
        if self.config.visit_limit is not None:
            visits = visits.head(self.config.visit_limit)

        iterator = tqdm(
            visits.iterrows(),
            total=len(visits),
            desc="Building science calendar",
            disable=not self.config.show_progress,
        )

        for visit_counter, (_, row) in enumerate(iterator, start=1):
            self._add_visit(root, visit_counter, row)

        return root

    def _add_meta(self, root: ET.Element) -> None:
        weights = ", ".join(
            f"{value:.1f}" for value in self.config.transit_scheduling_weights
        )
        keepout = ", ".join(
            f"{value:.1f}"
            for value in (
                self.config.sun_avoidance_deg,
                self.config.moon_avoidance_deg,
                self.config.earth_avoidance_deg,
            )
        )

        valid_from = str(self.schedule.iloc[0]["Observation Start"])
        expires = str(self.schedule.iloc[self.schedule.index[-1]]["Observation Stop"])

        raw_created = self.config.created_timestamp
        if isinstance(raw_created, str):
            created_value = raw_created
        else:
            timestamp = raw_created or datetime.now()
            # Round to nearest second
            created_value = str(
                (timestamp + timedelta(microseconds=500_000)).replace(microsecond=0)
            )

        attrs = {
            "Valid_From": valid_from,
            "Expires": expires,
            "Calendar_Weights": weights,
            "Keepout_Angles": keepout,
            "Observation_Sequence_Duration_hrs_max": str(self.sequence_duration),
            "Removed_Sequences_Shorter_Than_min": str(self.config.min_sequence_minutes),
            "Created": created_value,
            "Delivery_Id": "",
        }
        if self.config.author:
            attrs["Author"] = self.config.author

        ET.SubElement(root, "Meta", attrib=attrs)

    def _add_visit(self, root: ET.Element, visit_counter: int, row: pd.Series) -> None:
        id_padding = 4 - len(str(visit_counter))
        target_label = str(row.get("Target", ""))

        if not target_label or target_label == "Free Time":
            return

        if target_label.startswith("WARNING"):
            LOGGER.warning("Need visible STD during %s", target_label)
            return

        target_name, star_name = _normalise_target_name(target_label)

        visit_element = ET.SubElement(root, "Visit")
        ET.SubElement(visit_element, "ID").text = f"{'0' * id_padding}{visit_counter}"

        start = _parse_datetime(row.get("Observation Start"))
        stop = _parse_datetime(row.get("Observation Stop"))
        if start is None or stop is None:
            raise ValueError(f"Unable to parse observation window for {target_label}")

        planet_row = _lookup_planet_row(self.target_catalog, target_name)
        has_transit = _is_transit_entry(row)

        if planet_row is not None and has_transit:
            visibility_df = _read_visibility(
                self.data_dir / "targets" / star_name, star_name
            )
            transit_df = _read_planet_visibility(
                self.data_dir / "targets" / star_name / target_name, target_name
            )
            target_info = planet_row
            transit_windows = _transit_windows(transit_df)
            transit_start, transit_stop = (
                transit_windows if transit_windows else ([], [])
            )
            priority_flag = True
        else:
            visibility_df = _read_visibility(
                self.data_dir / "aux_targets" / star_name, target_name
            )
            target_info = _lookup_auxiliary_row(self.aux_catalog, target_name)
            transit_start, transit_stop = ([], [])
            priority_flag = False

        if visibility_df is None or visibility_df.empty:
            LOGGER.error(
                "No visibility data for %s. Aborting schedule build.", target_name
            )
            return

        try:
            ra_value = (
                float(target_info.iloc[0]["RA"])
                if target_info is not None
                else float("nan")
            )
            dec_value = (
                float(target_info.iloc[0]["DEC"])
                if target_info is not None
                else float("nan")
            )
        except (KeyError, ValueError, TypeError, AttributeError):
            ra_value, dec_value = _resolve_coordinates(star_name)

        visit_times, visibility_flags = _extract_visibility_segment(
            visibility_df,
            start,
            stop,
            self.config.min_sequence_minutes,
        )
        # If all samples were filtered out (e.g. sequences too short), attempt
        # to fall back to the transit window for transit entries so that
        # scheduled transits can still be emitted.
        if not visit_times or not any(bool(f) for f in visibility_flags):
            if has_transit and transit_start and transit_stop:
                try:
                    visit_times = [transit_start[0], transit_stop[0]]
                    visibility_flags = [1, 1]
                except Exception:
                    source, time_min, time_max, in_window = _visibility_diagnostics(
                        visibility_df, start, stop
                    )
                    LOGGER.warning(
                        "No visibility samples within visit for %s (visit=%s..%s, vis_file=%s, vis_range=%s..%s, samples_in_window=%s)",
                        target_name,
                        start,
                        stop,
                        source,
                        time_min,
                        time_max,
                        in_window,
                    )
                    return
            else:
                source, time_min, time_max, in_window = _visibility_diagnostics(
                    visibility_df, start, stop
                )
                LOGGER.warning(
                    "No visibility samples within visit for %s (visit=%s..%s, vis_file=%s, vis_range=%s..%s, samples_in_window=%s)",
                    target_name,
                    start,
                    stop,
                    source,
                    time_min,
                    time_max,
                    in_window,
                )
                return

        visibility_changes = _visibility_change_indices(visibility_flags)
        final_time = visit_times[-1]
        seq_counter = 1

        if not visibility_changes:
            self._emit_full_visibility(
                visit_element,
                target_name,
                start,
                final_time,
                ra_value,
                dec_value,
                target_info,
                priority_flag,
                transit_start,
                transit_stop,
            )
            return

        oc_starts, oc_stops, augmented_changes = _occultation_windows(
            visit_times,
            visibility_flags,
            visibility_changes,
        )

        occultation_info = self._find_occultation_target(
            oc_starts,
            oc_stops,
            start,
            final_time,
            ra_value,
            dec_value,
        )
        if occultation_info is None:
            LOGGER.warning(
                "Unable to schedule occultation target for %s between %s and %s",
                target_name,
                start,
                final_time,
            )
            return

        occ_df, scheduled = occultation_info
        if not scheduled or occ_df is None:
            LOGGER.warning(
                "Unable to schedule occultation target for %s between %s and %s",
                target_name,
                start,
                final_time,
            )
            return

        oc_index = 0
        for change_position, change_idx in enumerate(augmented_changes):
            if change_position == 0:
                segment_start = visit_times[0]
            else:
                segment_start = visit_times[augmented_changes[change_position - 1] + 1]

            segment_stop = visit_times[change_idx]
            is_visible = bool(visibility_flags[change_idx])

            if is_visible:
                current = segment_start
                while current < segment_stop:
                    next_value = min(current + self.sequence_duration, segment_stop)
                    priority = _target_priority(
                        priority_flag,
                        transit_start,
                        transit_stop,
                        current,
                        next_value,
                    )
                    observation_sequence(
                        visit_element,
                        f"{seq_counter:03d}",
                        target_name,
                        priority,
                        current.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        next_value.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        ra_value,
                        dec_value,
                        target_info if target_info is not None else pd.DataFrame(),
                    )
                    seq_counter += 1
                    current = next_value
            else:
                current = segment_start
                while current < segment_stop:
                    next_value = (
                        min(current + self.occultation_limit, segment_stop)
                        if self.config.break_occultation_sequences
                        else segment_stop
                    )
                    if occ_df is None or oc_index >= len(occ_df):
                        LOGGER.warning(
                            "Ran out of occultation targets for %s between %s and %s",
                            target_name,
                            current,
                            next_value,
                        )
                        break

                    occ_target = str(occ_df.iloc[oc_index]["Target"])

                    # Check if this occultation target has exceeded its time limit
                    current_occ_time = self.occultation_obs_time.get(
                        occ_target, timedelta()
                    )
                    target_time_limit = self._get_occultation_time_limit(occ_target)
                    if current_occ_time >= target_time_limit:
                        LOGGER.info(
                            "Skipping %s: exceeded occultation time limit (%.1f/%.1f hrs)",
                            occ_target,
                            current_occ_time.total_seconds() / 3600,
                            target_time_limit.total_seconds() / 3600,
                        )
                        oc_index += 1
                        continue

                    occ_info = _lookup_occultation_info(
                        occ_target,
                        self.target_catalog,
                        self.aux_catalog,
                        self.occ_catalog,
                    )
                    ra_occ = _fallback_float(
                        occ_df.iloc[oc_index].get("RA"), occ_info, "RA"
                    )
                    dec_occ = _fallback_float(
                        occ_df.iloc[oc_index].get("DEC"), occ_info, "DEC"
                    )

                    observation_sequence(
                        visit_element,
                        f"{seq_counter:03d}",
                        occ_target,
                        "0",
                        current.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        next_value.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        ra_occ,
                        dec_occ,
                        occ_info if occ_info is not None else pd.DataFrame(),
                    )

                    # Track the observation time for this occultation target
                    sequence_duration = next_value - current
                    self.occultation_obs_time[occ_target] = (
                        self.occultation_obs_time.get(occ_target, timedelta())
                        + sequence_duration
                    )

                    seq_counter += 1
                    oc_index += 1
                    current = next_value

    def _emit_full_visibility(
        self,
        visit_element: ET.Element,
        target_name: str,
        start: datetime,
        stop: datetime,
        ra_value: float,
        dec_value: float,
        target_info: pd.DataFrame | None,
        priority_flag: bool,
        transit_start: Sequence[datetime],
        transit_stop: Sequence[datetime],
    ) -> None:
        segments = break_long_sequences(start, stop, self.sequence_duration)
        seq_counter = 1
        for seg_start, seg_stop in segments:
            priority = _target_priority(
                priority_flag,
                transit_start,
                transit_stop,
                seg_start,
                seg_stop,
            )
            observation_sequence(
                visit_element,
                f"{seq_counter:03d}",
                target_name,
                priority,
                seg_start,
                seg_stop,
                ra_value,
                dec_value,
                target_info if target_info is not None else pd.DataFrame(),
            )
            seq_counter += 1

    def _find_occultation_target(
        self,
        starts: Sequence[datetime],
        stops: Sequence[datetime],
        visit_start: datetime,
        visit_stop: datetime,
        reference_ra: float,
        reference_dec: float,
    ) -> Optional[tuple[pd.DataFrame, bool]]:
        if not starts or not stops:
            return None

        if self.config.break_occultation_sequences:
            expanded_starts: list[datetime] = []
            expanded_stops: list[datetime] = []
            for start, stop in zip(starts, stops):
                segments = break_long_sequences(start, stop, self.occultation_limit)
                if not segments:
                    expanded_starts.append(start)
                    expanded_stops.append(stop)
                    continue

                for segment_start, segment_stop in segments:
                    expanded_starts.append(segment_start)
                    expanded_stops.append(segment_stop)
        else:
            expanded_starts = list(starts)
            expanded_stops = list(stops)

        candidates: List[Tuple[Path, str, Path]] = [
            (
                self.data_dir / "occultation-standard_targets.csv",
                "occ list",
                self.data_dir / "aux_targets",
            ),
        ]
        if not self.config.use_target_list_for_occultations:
            candidates.reverse()

        # Build set of targets that have exceeded their time limit
        excluded_targets = {
            name
            for name, obs_time in self.occultation_obs_time.items()
            if obs_time >= self._get_occultation_time_limit(name)
        }

        for csv_path, label, vis_root in candidates:
            result_df, flag = _build_occultation_schedule(
                expanded_starts,
                expanded_stops,
                visit_start,
                visit_stop,
                csv_path,
                vis_root,
                label,
                reference_ra,
                reference_dec,
                self.config.prioritise_occultations_by_slew,
                excluded_targets,
            )
            if flag and result_df is not None:
                return result_df, True
        return None


def _read_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required target catalog missing: {path}")
    df = read_csv_cached(str(path))
    if df is None:
        raise FileNotFoundError(f"Unable to read catalog: {path}")
    return df


def _normalise_target_name(target: str) -> tuple[str, str]:
    if target.endswith("STD"):
        stripped = target[:-4]
        return stripped, stripped
    if target.endswith(tuple("bcdef")) and target != "EV_Lac":
        return target, target[:-1].strip()
    return target, target


def _parse_datetime(value: object) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        for pattern in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d",  # Date only (time assumed to be 00:00:00)
        ):
            try:
                return datetime.strptime(value, pattern)
            except ValueError:
                continue
    return None


def _is_transit_entry(row: pd.Series) -> bool:
    value = row.get("Transit Coverage")
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(numeric)


def _lookup_planet_row(
    catalog: pd.DataFrame, target_name: str
) -> Optional[pd.DataFrame]:
    if catalog.empty or "Planet Name" not in catalog.columns:
        return None
    match = catalog.loc[catalog["Planet Name"] == target_name]
    if match.empty:
        return None
    return match.head(1)


def _lookup_auxiliary_row(
    catalog: pd.DataFrame, target_name: str
) -> Optional[pd.DataFrame]:
    if catalog.empty or "Star Name" not in catalog.columns:
        return None
    match = catalog.loc[catalog["Star Name"] == target_name]
    if match.empty:
        return None
    # Filter out exoplanet rows (those with a Planet Name) so we return
    # the auxiliary row when both planet and auxiliary entries exist.
    if "Planet Name" in match.columns:
        aux_only = match.loc[match["Planet Name"].isna() | (match["Planet Name"] == "")]
        if not aux_only.empty:
            return aux_only.head(1)
    return match.head(1)


def _lookup_occultation_info(
    target_name: str,
    planet_catalog: pd.DataFrame,
    aux_catalog: pd.DataFrame,
    occ_catalog: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    if "Star Name" in planet_catalog.columns:
        planet_match = planet_catalog.loc[planet_catalog["Star Name"] == target_name]
        if not planet_match.empty:
            return planet_match.head(1)
    if "Planet Name" in planet_catalog.columns:
        planet_match = planet_catalog.loc[planet_catalog["Planet Name"] == target_name]
        if not planet_match.empty:
            return planet_match.head(1)

    aux_match = aux_catalog.loc[
        aux_catalog.get("Star Name", pd.Series(dtype=object)) == target_name
    ]
    if not aux_match.empty:
        return aux_match.head(1)

    occ_match = occ_catalog.loc[
        occ_catalog.get("Star Name", pd.Series(dtype=object)) == target_name
    ]
    if not occ_match.empty:
        return occ_match.head(1)

    return None


def _read_visibility(directory: Path, name: str) -> Optional[pd.DataFrame]:
    """Read star visibility file with caching."""
    path = directory / f"Visibility for {name}.parquet"
    df = read_parquet_cached(
        str(path),
        columns=["Time(MJD_UTC)", "Time_UTC", "Visible"],
    )
    if df is None:
        # Older visibility parquet fixtures (and some historical outputs) may not
        # include Time_UTC. Fall back to MJD-only visibility timeline.
        df = read_parquet_cached(
            str(path),
            columns=["Time(MJD_UTC)", "Visible"],
        )
    if df is None:
        LOGGER.debug("Visibility file missing for %s", name)
        return None
    # Keep the source path for later diagnostics/logging.
    df.attrs["_source_path"] = str(path)
    if df.empty:
        LOGGER.debug("DF is empty for %s", name)
    return df


def _visibility_diagnostics(
    visibility_df: pd.DataFrame,
    start: datetime,
    stop: datetime,
) -> tuple[str, str, str, str]:
    """Return (source_path, time_min, time_max, samples_in_window) for warnings."""
    source = str(visibility_df.attrs.get("_source_path", "<unknown>"))
    if "Time_UTC" in visibility_df.columns and pd.api.types.is_datetime64_any_dtype(
        visibility_df["Time_UTC"]
    ):
        times = visibility_df["Time_UTC"].to_numpy(dtype="datetime64[ns]")
        if times.size == 0:
            return source, "<empty>", "<empty>", "0"
        time_min = pd.to_datetime(times.min()).to_pydatetime().isoformat(sep=" ")
        time_max = pd.to_datetime(times.max()).to_pydatetime().isoformat(sep=" ")
        mask = (times >= np.datetime64(start)) & (times <= np.datetime64(stop))
        in_window = int(mask.sum())
        return source, time_min, time_max, str(in_window)

    if "Time(MJD_UTC)" in visibility_df.columns:
        mjd = visibility_df["Time(MJD_UTC)"].to_numpy(dtype=float)
        if mjd.size == 0:
            return source, "<empty>", "<empty>", "0"
        mjd_min = float(np.nanmin(mjd))
        mjd_max = float(np.nanmax(mjd))
        time_min = (
            Time(mjd_min, format="mjd", scale="utc").to_datetime().isoformat(sep=" ")
        )
        time_max = (
            Time(mjd_max, format="mjd", scale="utc").to_datetime().isoformat(sep=" ")
        )
        start_mjd = Time(start).mjd
        stop_mjd = Time(stop).mjd
        in_window = int(((mjd >= start_mjd) & (mjd <= stop_mjd)).sum())
        return source, time_min, time_max, str(in_window)

    return source, "<unknown>", "<unknown>", "<unknown>"


def _read_planet_visibility(directory: Path, name: str) -> Optional[pd.DataFrame]:
    """Read planet transit-visibility file with caching.

    Planet visibility parquet files contain transit windows (MJD start/stop), not a
    per-timestep visibility timeline. Therefore we load the transit window columns.
    """
    path = directory / f"Visibility for {name}.parquet"
    df = read_parquet_cached(
        str(path),
        columns=["Transit_Start", "Transit_Stop"],
    )
    if df is None:
        LOGGER.debug("Planet visibility file missing for %s", name)
    return df


def _extract_visibility_segment(
    visibility_df: pd.DataFrame,
    start: datetime,
    stop: datetime,
    min_sequence_minutes: int,
) -> tuple[List[datetime], List[int]]:
    if "Time_UTC" in visibility_df.columns and pd.api.types.is_datetime64_any_dtype(
        visibility_df["Time_UTC"]
    ):
        times_dt64 = visibility_df["Time_UTC"].to_numpy(dtype="datetime64[ns]")
        start_dt64 = np.datetime64(start)
        stop_dt64 = np.datetime64(stop)
        mask = (times_dt64 >= start_dt64) & (times_dt64 <= stop_dt64)
        if not bool(mask.any()):
            return [], []

        window_indices = np.flatnonzero(mask)

        # Round to nearest second with legacy half-up semantics.
        # (pandas .round('S') uses bankers rounding; we need >=0.5s to round up.)
        ns = times_dt64[window_indices].astype("datetime64[ns]").view("int64")
        rounded_ns = ((ns + 500_000_000) // 1_000_000_000) * 1_000_000_000
        rounded_dt64 = rounded_ns.view("datetime64[ns]")
        visit_times = pd.to_datetime(rounded_dt64).to_pydatetime().tolist()
    else:
        raw_times = Time(
            visibility_df["Time(MJD_UTC)"].to_numpy(),
            format="mjd",
            scale="utc",
        ).to_datetime()
        # Normalise astropy datetimes to naive Python datetimes (UTC) so comparisons
        # with schedule start/stop (which are naive datetimes) behave predictably.
        raw_times = [
            (rt.replace(tzinfo=None) if getattr(rt, "tzinfo", None) is not None else rt)
            for rt in raw_times
        ]

        mask = [start <= value <= stop for value in raw_times]
        if not any(mask):
            return [], []

        window_indices = [idx for idx, include in enumerate(mask) if include]
        filtered_times = [raw_times[idx] for idx in window_indices]
        # Round each time to nearest second
        visit_times = [
            (t + timedelta(microseconds=500_000)).replace(microsecond=0)
            for t in filtered_times
        ]

    flags = [float(visibility_df.iloc[int(idx)]["Visible"]) for idx in window_indices]
    filtered_flags, _ = remove_short_sequences(
        np.asarray(flags, dtype=float),
        min_sequence_minutes,
    )
    return visit_times, [int(value) for value in filtered_flags]


def _visibility_change_indices(flags: Sequence[int]) -> List[int]:
    return [idx for idx in range(len(flags) - 1) if flags[idx] != flags[idx + 1]]


def _occultation_windows(
    visit_times: Sequence[datetime],
    visibility_flags: Sequence[int],
    visibility_changes: Sequence[int],
) -> tuple[List[datetime], List[datetime], List[int]]:
    changes = list(visibility_changes)
    flags = list(visibility_flags)
    times = list(visit_times)

    if not flags:
        return [], [], []

    if flags[-1] == 0 and len(times) >= 2:
        changes.append(len(times) - 2)

    occ_starts: List[datetime] = []
    occ_stops: List[datetime] = []

    if flags[0] == 0 and changes:
        occ_starts.append(times[0])
        occ_stops.append(times[changes[0]])

    for idx in range(len(changes) - 1):
        change_idx = changes[idx]
        next_idx = changes[idx + 1]
        if flags[next_idx] == 0:
            occ_starts.append(times[change_idx + 1])
            occ_stops.append(times[next_idx])

    if flags[-1] == 1 and len(times) >= 2:
        changes.append(len(times) - 2)

    if not occ_starts:
        return [], [], changes

    if len(occ_starts) != len(occ_stops):
        raise ValueError("Occultation start/stop lists are mismatched")

    return occ_starts, occ_stops, changes


def _prioritise_occultation_targets(
    occ_list: pd.DataFrame,
    reference_ra: float,
    reference_dec: float,
) -> pd.DataFrame:
    if occ_list.empty:
        return occ_list

    if not (np.isfinite(reference_ra) and np.isfinite(reference_dec)):
        return occ_list

    if "RA" not in occ_list.columns or "DEC" not in occ_list.columns:
        return occ_list

    ra_values = pd.to_numeric(occ_list["RA"], errors="coerce")
    dec_values = pd.to_numeric(occ_list["DEC"], errors="coerce")
    if ra_values.isna().all() or dec_values.isna().all():
        return occ_list

    try:
        origin = SkyCoord(ra=reference_ra, dec=reference_dec, unit="deg")
        target_coords = SkyCoord(
            ra=ra_values.to_numpy(),
            dec=dec_values.to_numpy(),
            unit="deg",
        )
        separations = origin.separation(target_coords).deg
    except Exception as exc:  # SkyCoord failure should not abort scheduling
        LOGGER.debug("Unable to rank occultation targets by slew distance: %s", exc)
        return occ_list

    priorities = np.where(np.isfinite(separations), separations, np.inf)
    reordered = (
        occ_list.assign(_separation=priorities)
        .sort_values(by="_separation", kind="mergesort")
        .drop(columns="_separation")
        .reset_index(drop=True)
    )
    return reordered


def _build_occultation_schedule(
    starts: Sequence[datetime],
    stops: Sequence[datetime],
    visit_start: datetime,
    visit_stop: datetime,
    list_path: Path,
    vis_root: Path,
    label: str,
    reference_ra: float,
    reference_dec: float,
    prioritise_by_slew: bool,
    excluded_targets: Optional[set] = None,
) -> tuple[Optional[pd.DataFrame], bool]:
    if not starts or not stops:
        return None, False

    schedule_rows = [
        [
            "",
            start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            stop.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "",
            "",
        ]
        for start, stop in zip(starts, stops)
    ]
    occ_df = pd.DataFrame(
        schedule_rows, columns=["Target", "start", "stop", "RA", "DEC"]
    )

    try:
        occ_list = read_csv_cached(str(list_path))
    except FileNotFoundError:
        LOGGER.warning("Occultation list missing: %s", list_path)
        return None, False

    # Filter out excluded targets (those that have exceeded their time limit)
    if excluded_targets and "Star Name" in occ_list.columns:
        before_count = len(occ_list)
        occ_list = occ_list[~occ_list["Star Name"].isin(excluded_targets)].reset_index(
            drop=True
        )
        if len(occ_list) < before_count:
            LOGGER.debug(
                "Excluded %d occultation targets that exceeded time limit",
                before_count - len(occ_list),
            )

    # If no targets remain after exclusion, fail early
    if occ_list.empty:
        LOGGER.warning("No occultation targets available after exclusion filter")
        return None, False

    if prioritise_by_slew:
        occ_list = _prioritise_occultation_targets(
            occ_list,
            reference_ra,
            reference_dec,
        )

    target_names = occ_list.get("Star Name", pd.Series(dtype=object)).to_numpy()
    starts_mjd = Time(list(starts), format="datetime", scale="utc").to_value("mjd")
    stops_mjd = Time(list(stops), format="datetime", scale="utc").to_value("mjd")

    occ_df, flag = observation_utils.schedule_occultation_targets(
        target_names,
        starts_mjd,
        stops_mjd,
        visit_start,
        visit_stop,
        str(vis_root),
        occ_df,
        occ_list,
        label,
    )
    return occ_df, flag


def _transit_windows(
    transit_df: Optional[pd.DataFrame],
) -> Optional[tuple[List[datetime], List[datetime]]]:
    if transit_df is None or transit_df.empty:
        return None

    start_times = Time(
        transit_df["Transit_Start"].to_numpy(),
        format="mjd",
        scale="utc",
    ).to_datetime()
    stop_times = Time(
        transit_df["Transit_Stop"].to_numpy(),
        format="mjd",
        scale="utc",
    ).to_datetime()

    # Round each time to nearest second to match legacy
    # `round_to_nearest_second` behaviour.
    start = [
        (t + timedelta(microseconds=500_000)).replace(microsecond=0)
        for t in start_times
    ]
    stop = [
        (t + timedelta(microseconds=500_000)).replace(microsecond=0) for t in stop_times
    ]
    return start, stop


def _target_priority(
    priority_flag: bool,
    transit_start: Sequence[datetime],
    transit_stop: Sequence[datetime],
    sequence_start: datetime,
    sequence_stop: datetime,
) -> str:
    if not priority_flag or not transit_start or not transit_stop:
        return "0"

    for start, stop in zip(transit_start, transit_stop):
        if start <= sequence_stop and stop >= sequence_start:
            return "2"
    return "1"


def _fallback_float(value: object, info: Optional[pd.DataFrame], column: str) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass

    if info is not None and column in info.columns:
        try:
            candidate = info.iloc[0][column]
            if isinstance(candidate, (int, float, np.integer, np.floating)):
                return float(candidate)
            if isinstance(candidate, str):
                return float(candidate)
        except (KeyError, ValueError, TypeError):
            pass

    return float("nan")


def _resolve_coordinates(star_name: str) -> tuple[float, float]:
    """Raise error if coordinates not in catalog - no Simbad lookups allowed."""
    raise RuntimeError(f"No coordinates found in catalog for {star_name}")


def _serialise_calendar(root: ET.Element) -> str:
    _convert_numeric_content(root)
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    xml_doc = minidom.parseString(xml_bytes)
    return xml_doc.toprettyxml(indent="\t")


def _convert_numeric_content(element: ET.Element) -> None:
    for child in element.iter():
        for key, value in list(child.attrib.items()):
            if isinstance(value, Number):
                child.set(key, str(value))
        if child.text is not None and isinstance(child.text, Number):
            child.text = str(child.text)
