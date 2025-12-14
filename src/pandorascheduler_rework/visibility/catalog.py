"""Public API for generating visibility artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from pandorascheduler_rework.config import PandoraSchedulerConfig
from pandorascheduler_rework.utils.io import read_csv_cached, read_parquet_cached

from .geometry import (
    MinuteCadence,
    build_minute_cadence,
    compute_saa_crossings,
    interpolate_gmat_ephemeris,
)

LOGGER = logging.getLogger(__name__)


def build_visibility_catalog(
    config: PandoraSchedulerConfig,
    target_list: Path,
    partner_list: Path | None = None,
    output_subpath: str = "targets",
) -> None:
    """Generate visibility outputs for the requested targets."""

    if not config.output_dir:
        raise ValueError("config.output_dir is required for visibility generation")

    output_root = config.output_dir / "data" / output_subpath
    output_root.mkdir(parents=True, exist_ok=True)

    target_path = target_list if target_list.is_absolute() else target_list.resolve()
    gmat_path = (
        config.gmat_ephemeris
        if config.gmat_ephemeris.is_absolute()
        else config.gmat_ephemeris.resolve()
    )

    target_manifest = _load_target_manifest(target_path, config.target_filters)
    if target_manifest.empty:
        LOGGER.info("No targets matched visibility configuration; skipping build.")
        return

    # Check if any star visibility files need to be generated
    stars_to_generate = []
    for _, row in target_manifest.iterrows():
        star_name = str(row.get("Star Name", ""))
        output_path = output_root / star_name / f"Visibility for {star_name}.parquet"
        if not output_path.exists() or config.force_regenerate:
            stars_to_generate.append((star_name, row))

    # Only compute expensive ephemeris/payload if we need to generate files
    if stars_to_generate:
        cadence = build_minute_cadence(config.window_start, config.window_end)
        ephemeris = interpolate_gmat_ephemeris(gmat_path, cadence)
        base_payload = _build_base_payload(ephemeris, cadence)
        star_metadata = _build_star_metadata(target_manifest)

        for star_name, row in stars_to_generate:
            output_dir = output_root / star_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"Visibility for {star_name}.parquet"

            star_coord = _resolve_star_coord(row, star_metadata)
            visibility_df = _build_star_visibility(base_payload, star_coord, config)

            if "exoplanet" not in target_path.name.lower():
                visibility_df["Time(MJD_UTC)"] = np.round(
                    visibility_df["Time(MJD_UTC)"], 6
                )

            visibility_df.to_parquet(
                output_path,
                index=False,
                engine="pyarrow",
                compression="snappy",
                write_statistics=False,
                use_dictionary=False,
            )
    else:
        # Still need star_metadata for planet transits
        star_metadata = _build_star_metadata(target_manifest)

    planet_manifests: list[tuple[pd.DataFrame, Path]] = [(target_manifest, target_path)]

    if partner_list is not None:
        partner_path = (
            partner_list if partner_list.is_absolute() else partner_list.resolve()
        )
        partner_manifest = _load_target_manifest(partner_path, config.target_filters)
        if not partner_manifest.empty:
            star_metadata.update(_build_star_metadata(partner_manifest))
            planet_manifests.append((partner_manifest, partner_path))

    generated_planets: list[tuple[str, str]] = []
    for manifest, manifest_path in planet_manifests:
        generated_planets.extend(
            _build_planet_transits(
                manifest,
                manifest_path,
                output_root,
                star_metadata,
                config,
            )
        )

    _apply_transit_overlaps(generated_planets, output_root)


def _load_target_manifest(
    manifest_path: Path,
    filters: Iterable[str],
) -> pd.DataFrame:
    manifest = read_csv_cached(str(manifest_path))
    if manifest is None:
        raise FileNotFoundError(f"Target manifest missing: {manifest_path}")
    if filters:
        manifest = manifest[manifest["Star Name"].isin(filters)]
    required_columns = {"Star Name", "Star Simbad Name"}
    missing = required_columns.difference(manifest.columns)
    if missing:
        raise ValueError(f"Target manifest missing required columns: {sorted(missing)}")
    return manifest.reset_index(drop=True)


def _build_base_payload(ephemeris, cadence: MinuteCadence) -> dict[str, np.ndarray]:
    saa_crossing = compute_saa_crossings(
        ephemeris.spacecraft_lat_deg, ephemeris.spacecraft_lon_deg
    )

    # Pre-compute datetime conversion once for all stars.
    # Store as datetime64[ns] (timestamp) so parquet writing is faster than
    # writing variable-length strings, and downstream code avoids re-parsing.
    mjd_array = np.asarray(cadence.mjd_utc, dtype=float)
    time_utc = Time(mjd_array, format="mjd", scale="utc")
    datetime_utc = time_utc.datetime64

    return {
        "Time(MJD_UTC)": mjd_array,
        "Time_UTC": datetime_utc,
        "SAA_Crossing": np.round(saa_crossing, 1),
        "earth_pc": SkyCoord(
            ephemeris.earth_pc,
            unit=u.km,
            representation_type="cartesian",
        ),
        "sun_pc": SkyCoord(
            ephemeris.sun_pc,
            unit=u.km,
            representation_type="cartesian",
        ),
        "moon_pc": SkyCoord(
            ephemeris.moon_pc,
            unit=u.km,
            representation_type="cartesian",
        ),
    }


def _build_star_visibility(
    payload: dict[str, np.ndarray],
    star_coord: SkyCoord,
    config: PandoraSchedulerConfig,
) -> pd.DataFrame:
    sun_sep = payload["sun_pc"].separation(star_coord).deg
    moon_sep = payload["moon_pc"].separation(star_coord).deg
    earth_sep = payload["earth_pc"].separation(star_coord).deg

    sun_req = sun_sep > config.sun_avoidance_deg
    moon_req = moon_sep > config.moon_avoidance_deg
    earth_req = earth_sep > config.earth_avoidance_deg

    visible = (sun_req & moon_req & earth_req).astype(float)

    # Use pre-computed datetime array from payload (computed once for all stars)
    data = {
        "Time(MJD_UTC)": payload["Time(MJD_UTC)"],
        "Time_UTC": payload["Time_UTC"],
        "SAA_Crossing": payload["SAA_Crossing"],
        "Visible": np.round(visible, 1),
        "Earth_Sep": np.round(earth_sep, 3),
        "Moon_Sep": np.round(moon_sep, 3),
        "Sun_Sep": np.round(sun_sep, 3),
    }
    return pd.DataFrame(data)


def _resolve_star_coord(
    row: pd.Series,
    star_metadata: dict[str, tuple[float, float]],
) -> SkyCoord:
    """Resolve star coordinates from catalog data only (no Simbad lookups)."""
    star_name = str(row.get("Star Name", ""))

    ra_val = row.get("RA")
    dec_val = row.get("DEC")

    # Use star_metadata as fallback if RA/DEC missing
    if (pd.isna(ra_val) or pd.isna(dec_val)) and star_name in star_metadata:
        fallback_ra, fallback_dec = star_metadata[star_name]
        if pd.isna(ra_val):
            ra_val = fallback_ra
        if pd.isna(dec_val):
            dec_val = fallback_dec

    if pd.notna(ra_val) and pd.notna(dec_val):
        return SkyCoord(
            ra=float(ra_val) * u.deg, dec=float(dec_val) * u.deg, frame="icrs"
        )

    # No Simbad fallback - raise error if coordinates not in catalog
    raise RuntimeError(f"No coordinates found in catalog for {star_name}")


def _build_star_metadata(manifest: pd.DataFrame) -> dict[str, tuple[float, float]]:
    if "RA" not in manifest.columns or "DEC" not in manifest.columns:
        return {}

    metadata: dict[str, tuple[float, float]] = {}
    for _, row in manifest.iterrows():
        ra = row.get("RA")
        dec = row.get("DEC")
        if pd.notna(ra) and pd.notna(dec):
            star_name = str(row.get("Star Name", ""))
            metadata[star_name] = (float(ra), float(dec))
    return metadata


def _build_planet_transits(
    manifest: pd.DataFrame,
    manifest_path: Path,
    output_root: Path,
    star_metadata: dict[str, tuple[float, float]],
    config: PandoraSchedulerConfig,
) -> list[tuple[str, str]]:
    if manifest.empty:
        return []

    required_columns = {
        "Planet Name",
        "Star Name",
        "Transit Duration (hrs)",
        "Period (days)",
        "Transit Epoch (BJD_TDB-2400000.5)",
    }
    missing = required_columns.difference(manifest.columns)
    if missing:
        LOGGER.info(
            "Manifest %s missing planet columns; skipping transit generation",
            manifest_path.name,
        )
        return []

    observer_location = EarthLocation(
        lat=0.0 * u.deg, lon=0.0 * u.deg, height=600.0 * u.km
    )

    generated: list[tuple[str, str]] = []

    for _, row in manifest.iterrows():
        star_name = str(row.get("Star Name", ""))
        planet_name = str(row.get("Planet Name", ""))

        star_visibility_path = (
            output_root / star_name / f"Visibility for {star_name}.parquet"
        )
        if not star_visibility_path.exists():
            LOGGER.warning(
                "Star visibility missing for %s; skipping planet %s",
                star_name,
                planet_name,
            )
            continue

        planet_dir = output_root / star_name / planet_name
        planet_dir.mkdir(parents=True, exist_ok=True)
        planet_output = planet_dir / f"Visibility for {planet_name}.parquet"
        if planet_output.exists() and not config.force_regenerate:
            LOGGER.info(
                "Skipping %s/%s; planet visibility already exists",
                star_name,
                planet_name,
            )
            generated.append((star_name, planet_name))
            continue

        planet_df = _compute_planet_transits(
            star_visibility_path,
            row,
            star_metadata,
            observer_location,
        )
        planet_df.to_parquet(
            planet_output,
            index=False,
            engine="pyarrow",
            compression="snappy",
            write_statistics=False,
            use_dictionary=False,
        )
        if not planet_df.empty:
            generated.append((star_name, planet_name))

    return generated


def _compute_planet_transits(
    star_visibility_path: Path,
    planet_row: pd.Series,
    star_metadata: dict[str, tuple[float, float]],
    observer_location: EarthLocation,
) -> pd.DataFrame:
    star_visibility = read_parquet_cached(
        str(star_visibility_path),
        columns=["Time(MJD_UTC)", "Visible", "SAA_Crossing"],
    )
    if star_visibility is None or star_visibility.empty:
        raise FileNotFoundError(
            f"Star visibility missing or empty for {star_visibility_path}"
        )
    t_mjd = star_visibility["Time(MJD_UTC)"].to_numpy(dtype=float)
    visible_mask = star_visibility["Visible"].to_numpy(dtype=float)

    if t_mjd.size == 0:
        return pd.DataFrame(
            {
                col: np.array([], dtype=float)
                for col in [
                    "Transits",
                    "Transit_Start",
                    "Transit_Stop",
                    "Transit_Coverage",
                ]
            }
        )

    transit_duration = planet_row["Transit Duration (hrs)"]
    period_days = planet_row["Period (days)"]
    epoch_bjd_tdb = planet_row["Transit Epoch (BJD_TDB-2400000.5)"]
    planet_name = planet_row["Planet Name"]

    if np.isnan(transit_duration) or np.isnan(period_days) or np.isnan(epoch_bjd_tdb):
        LOGGER.warning(
            "Incomplete ephemeris for %s; skipping planet visibility",
            planet_name,
        )
        return pd.DataFrame(
            {
                col: np.array([], dtype=float)
                for col in [
                    "Transits",
                    "Transit_Start",
                    "Transit_Stop",
                    "Transit_Coverage",
                ]
            }
        )

    transit_duration = float(transit_duration) * u.hour
    period = float(period_days) * u.day

    star_coord = _resolve_star_coord(
        planet_row,
        star_metadata,
    )

    bjd_tdb = Time(
        float(epoch_bjd_tdb) + 2400000.5,
        format="jd",
        scale="tdb",
        location=observer_location,
    )
    light_time = bjd_tdb.light_travel_time(
        star_coord, kind="barycentric", location=observer_location
    )
    jd_tdb = bjd_tdb - light_time
    epoch_mjd_utc = Time(jd_tdb.mjd, format="mjd", scale="utc")

    half_obs_width = 0.75 * u.hour + np.maximum(
        1.0 * u.hour + transit_duration / 2.0, transit_duration
    )
    time_grid = Time(t_mjd, format="mjd", scale="utc")

    if period <= 0 * u.day:
        LOGGER.warning(
            "Non-positive period for %s; skipping planet visibility",
            planet_name,
        )
        return pd.DataFrame(
            {
                col: np.array([], dtype=float)
                for col in [
                    "Transits",
                    "Transit_Start",
                    "Transit_Stop",
                    "Transit_Coverage",
                ]
            }
        )

    min_start_epoch = epoch_mjd_utc - half_obs_width
    elapsed_days = (time_grid[0] - min_start_epoch).to(u.day)
    min_pers_start = np.ceil((elapsed_days / period).value)

    first_transit = epoch_mjd_utc + min_pers_start * period

    mid_transits_list: list[Time] = []
    current = first_transit
    last_time = time_grid[-1]
    while current < last_time:
        mid_transits_list.append(current)
        current = current + period

    if not mid_transits_list:
        return pd.DataFrame(
            {
                col: np.array([], dtype=float)
                for col in [
                    "Transits",
                    "Transit_Start",
                    "Transit_Stop",
                    "Transit_Coverage",
                ]
            }
        )

    mid_transits = Time(mid_transits_list)
    start_transits = mid_transits - transit_duration / 2.0
    end_transits = mid_transits + transit_duration / 2.0

    start_datetimes = start_transits.to_value("datetime")
    end_datetimes = end_transits.to_value("datetime")

    # Floor to minute precision using pandas vectorized operations
    start_datetimes = pd.to_datetime(start_datetimes).floor("min").to_pydatetime()
    end_datetimes = pd.to_datetime(end_datetimes).floor("min").to_pydatetime()

    saa_mask = star_visibility["SAA_Crossing"].to_numpy(dtype=float)
    T_mjd_utc = Time(t_mjd, format="mjd", scale="utc")
    T_iso_utc = Time(T_mjd_utc.iso, format="iso", scale="utc")
    dt_iso_utc = T_iso_utc.to_value("datetime")

    # Use boolean indexing for better performance
    dt_vis_times = dt_iso_utc[visible_mask == 1.0]
    dt_saa_times = dt_iso_utc[saa_mask == 1.0]

    coverage = np.zeros(len(start_datetimes), dtype=float)
    saa_overlap = np.zeros(len(start_datetimes), dtype=float)

    for idx, (start_dt, end_dt) in enumerate(zip(start_datetimes, end_datetimes)):
        tran_minutes = pd.date_range(start_dt, end_dt, freq="min").to_pydatetime()
        if len(tran_minutes) == 0:
            continue
        minute_set = set(tran_minutes)
        tran_vis = minute_set.intersection(dt_vis_times)
        if len(tran_vis) > 0:
            coverage[idx] = len(tran_vis) / len(tran_minutes)

        saa_candidates = []
        for dt_val in dt_saa_times:
            if tran_minutes[0] <= dt_val <= tran_minutes[-1]:
                saa_candidates.append(dt_val)
        if saa_candidates:
            overlap = set(saa_candidates).intersection(tran_minutes)
            if overlap:
                saa_overlap[idx] = len(overlap) / len(tran_minutes)

    transit_df = pd.DataFrame(
        {
            "Transits": np.arange(len(start_datetimes), dtype=int),
            "Transit_Start": start_transits.value,
            "Transit_Stop": end_transits.value,
            "Transit_Start_UTC": start_datetimes,
            "Transit_Stop_UTC": end_datetimes,
            "Transit_Coverage": coverage,
            "SAA_Overlap": saa_overlap,
        }
    )
    return transit_df


def _apply_transit_overlaps(
    generated_planets: Iterable[tuple[str, str]],
    output_root: Path,
) -> None:
    star_planets: dict[str, list[str]] = {}
    for star_name, planet_name in generated_planets:
        star_planets.setdefault(star_name, []).append(planet_name)

    for star_name, planets in star_planets.items():
        if len(planets) < 2:
            continue

        # Quick check: if all planet files already have Transit_Overlap, skip expensive recomputation
        all_have_overlap = True
        for planet in planets:
            planet_path = (
                output_root / star_name / planet / f"Visibility for {planet}.parquet"
            )
            if not planet_path.exists():
                all_have_overlap = False
                break
            # Quick check: just read the header line to see if Transit_Overlap exists
            try:
                with open(planet_path, "r") as f:
                    header = f.readline().strip()
                    if "Transit_Overlap" not in header:
                        all_have_overlap = False
                        break
            except Exception:
                all_have_overlap = False
                break

        if all_have_overlap:
            continue  # Skip this star system - all planets already have overlaps computed

        planet_data: dict[str, pd.DataFrame] = {}
        minute_sets: dict[str, list[tuple[set, int]]] = {}

        for planet in planets:
            planet_path = (
                output_root / star_name / planet / f"Visibility for {planet}.parquet"
            )
            if not planet_path.exists():
                raise FileNotFoundError(
                    f"Expected planet visibility missing: {planet_path}"
                )
            df = read_parquet_cached(
                str(planet_path),
                columns=[
                    "Transit_Start",
                    "Transit_Stop",
                    "Transit_Coverage",
                    "SAA_Overlap",
                ],
            )
            if df is None:
                raise FileNotFoundError(
                    f"Unable to read planet visibility: {planet_path}"
                )
            # Skip planets with no transits (empty DataFrame except for headers)
            if df.empty:
                continue
            planet_data[planet] = df
            sets: list[tuple[set, int]] = []

            # Vectorized datetime processing (much faster than iterrows)
            if (
                "Transit_Start_UTC" in df.columns
                and df["Transit_Start_UTC"].notna().any()
            ):
                # Use pre-existing datetime columns (fastest path)
                start_times = pd.to_datetime(df["Transit_Start_UTC"]).dt.floor("min")
                end_times = pd.to_datetime(df["Transit_Stop_UTC"]).dt.floor("min")
            else:
                # Fallback to MJD conversion (vectorized)
                start_mjd = df["Transit_Start"].to_numpy(dtype=float)
                end_mjd = df["Transit_Stop"].to_numpy(dtype=float)
                start_times = pd.Series(
                    Time(start_mjd, format="mjd", scale="utc").to_datetime()
                ).dt.floor("min")
                end_times = pd.Series(
                    Time(end_mjd, format="mjd", scale="utc").to_datetime()
                ).dt.floor("min")

            # Build minute sets for each transit
            for start_dt, end_dt in zip(start_times, end_times):
                minutes = list(
                    pd.date_range(start_dt, end_dt, freq="min").to_pydatetime()
                )
                if not minutes:
                    sets.append((set(), 0))
                else:
                    sets.append((set(minutes), len(minutes)))
            minute_sets[planet] = sets

        for planet, df in planet_data.items():
            overlaps = np.zeros(len(df), dtype=float)
            current_sets = minute_sets[planet]
            for idx, (minutes, total) in enumerate(current_sets):
                if total == 0:
                    continue
                best_overlap = 0.0
                for other_planet, other_sets in minute_sets.items():
                    if other_planet == planet:
                        continue
                    for other_minutes, other_total in other_sets:
                        if other_total == 0 or not other_minutes:
                            continue
                        shared = minutes.intersection(other_minutes)
                        if shared:
                            overlap_fraction = len(shared) / total
                            best_overlap = max(best_overlap, min(overlap_fraction, 1.0))
                overlaps[idx] = min(
                    best_overlap, 1.0
                )  # Ensure overlap never exceeds 1.0

            if "Transit_Overlap" in df.columns:
                df["Transit_Overlap"] = overlaps
            else:
                df["Transit_Overlap"] = overlaps

            planet_path = (
                output_root / star_name / planet / f"Visibility for {planet}.parquet"
            )
            df.to_parquet(
                planet_path,
                index=False,
                engine="pyarrow",
                compression="snappy",
                write_statistics=False,
                use_dictionary=False,
            )
