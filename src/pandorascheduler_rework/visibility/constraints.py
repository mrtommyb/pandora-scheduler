"""Geometric constraint primitives for visibility computation.

This module ports the key geometric calculations from ``pandoravisibility``
(day/night Earth-limb detection, star tracker keepout, roll attitude, and
per-orbit roll optimisation) so that they can operate directly on the GMAT
ephemeris vectors already available in the long-term scheduler pipeline.

All array inputs are expected to have shape ``(N, 3)`` in the **EarthMJ2000Eq**
frame (km), matching :class:`InterpolatedEphemeris`.  One-dimensional arrays
have shape ``(N,)``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from pandorascheduler_rework.config import PandoraSchedulerConfig

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_R_EARTH_KM: float = 6371.0  # mean Earth radius in km

# ---------------------------------------------------------------------------
# Star tracker body-frame boresight vectors (normalised)
# ---------------------------------------------------------------------------
_ST1_BODY_RAW = np.array([0.6804, -0.7071, -0.1923])
_ST2_BODY_RAW = np.array([0.6804, 0.7071, -0.1923])
ST1_BODY: np.ndarray = _ST1_BODY_RAW / np.linalg.norm(_ST1_BODY_RAW)
ST2_BODY: np.ndarray = _ST2_BODY_RAW / np.linalg.norm(_ST2_BODY_RAW)


# ============================================================================
# Low-level geometric primitives
# ============================================================================


def _normalise(v: np.ndarray) -> np.ndarray:
    """Row-wise normalise an ``(N, 3)`` array, returning unit vectors."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    return v / norms


def fast_sep_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angular separation between two sets of unit vectors (degrees).

    Parameters
    ----------
    a, b : ndarray, shape ``(N, 3)``
        Unit vectors in the same reference frame.

    Returns
    -------
    ndarray, shape ``(N,)``
    """
    dot = np.einsum("ij,ij->i", a, b)
    return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))


def fast_limb_deg(
    target_unit: np.ndarray,
    zenith_unit: np.ndarray,
    limb_angle_rad: np.ndarray,
) -> np.ndarray:
    """Angular distance from the Earth's limb to a target (degrees).

    This replicates ``pandoravisibility.Visibility._fast_limb_deg``.

    Parameters
    ----------
    target_unit : ndarray, shape ``(N, 3)``
        Unit direction from the observer toward the target.
    zenith_unit : ndarray, shape ``(N, 3)``
        Unit direction from Earth's centre to the observer (= −nadir).
    limb_angle_rad : ndarray, shape ``(N,)``
        ``arccos(R_earth / observer_distance)`` in radians.

    Returns
    -------
    ndarray, shape ``(N,)``
        Limb angle in degrees.
    """
    dot = np.einsum("ij,ij->i", target_unit, zenith_unit)
    elev = np.arcsin(np.clip(dot, -1.0, 1.0))
    return np.rad2deg(elev + limb_angle_rad)


def earthlimb_is_sunlit(
    target_unit: np.ndarray,
    nadir_unit: np.ndarray,
    sun_unit: np.ndarray,
) -> np.ndarray:
    """Determine whether the nearest Earth-limb point to the target is sunlit.

    The algorithm projects the target direction onto the plane perpendicular to
    the nadir direction to find the angular direction of the nearest limb point,
    then checks whether the Sun is on the same side.

    Parameters
    ----------
    target_unit : ndarray, shape ``(N, 3)``
        Unit direction from observer toward the target.
    nadir_unit : ndarray, shape ``(N, 3)``
        Unit direction from observer toward Earth centre.
    sun_unit : ndarray, shape ``(N, 3)``
        Unit direction from observer toward the Sun.

    Returns
    -------
    ndarray, shape ``(N,)`` of bool
        ``True`` where the nearest limb is sunlit.
    """
    # zenith = -nadir (from Earth centre to observer)
    zenith = -nadir_unit

    # Project target onto plane ⊥ zenith
    dot_tz = np.einsum("ij,ij->i", target_unit, zenith)[:, None]  # (N, 1)
    proj = target_unit - zenith * dot_tz  # (N, 3)
    proj_norm = np.linalg.norm(proj, axis=1, keepdims=True)
    # Where projection is degenerate (target aligned with zenith), pick
    # an arbitrary direction — the limb is equidistant in all directions.
    proj_norm = np.where(proj_norm == 0, 1.0, proj_norm)
    limb_dir = proj / proj_norm  # unit direction to nearest limb point (N, 3)

    # Check if the limb point faces the Sun
    return np.einsum("ij,ij->i", limb_dir, sun_unit) > 0


def effective_earth_threshold(
    target_unit: np.ndarray,
    nadir_unit: np.ndarray,
    sun_unit: np.ndarray,
    day_deg: Optional[float],
    night_deg: Optional[float],
    default_deg: float,
) -> np.ndarray | float:
    """Per-timestep Earth-avoidance threshold (degrees).

    When both *day_deg* and *night_deg* are ``None`` returns the scalar
    *default_deg* (no per-timestep branch needed).  Otherwise returns an
    ``(N,)`` array with the day value where the limb is sunlit and the night
    value where it is not.
    """
    if day_deg is None and night_deg is None:
        return default_deg
    day = day_deg if day_deg is not None else default_deg
    night = night_deg if night_deg is not None else default_deg
    sunlit = earthlimb_is_sunlit(target_unit, nadir_unit, sun_unit)
    return np.where(sunlit, day, night)


# ============================================================================
# Attitude construction
# ============================================================================


def sun_constrained_attitude(
    target_unit: np.ndarray,
    sun_unit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute payload X/Y/Z axes for the Sun-constrained (default) attitude.

    Parameters
    ----------
    target_unit, sun_unit : ndarray, shape ``(N, 3)``

    Returns
    -------
    x_payload, y_payload, z_payload : ndarray, shape ``(N, 3)``
        Payload body-frame axes in the ECI frame.
    """
    z_payload = target_unit  # boresight = target direction
    y_payload = np.cross(sun_unit, z_payload)
    y_norm = np.linalg.norm(y_payload, axis=1, keepdims=True)
    y_norm = np.where(y_norm == 0, 1.0, y_norm)
    y_payload = y_payload / y_norm
    x_payload = np.cross(y_payload, z_payload)
    x_norm = np.linalg.norm(x_payload, axis=1, keepdims=True)
    x_norm = np.where(x_norm == 0, 1.0, x_norm)
    x_payload = x_payload / x_norm
    return x_payload, y_payload, z_payload


def fixed_roll_attitude(
    target_unit: np.ndarray,
    roll_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute payload axes for a fixed roll angle about the boresight.

    Roll is measured from the projection of celestial north (GCRS +Z) onto the
    plane perpendicular to the boresight, rotating via the right-hand rule about
    the boresight (+Z payload = target direction).

    Parameters
    ----------
    target_unit : ndarray, shape ``(N, 3)``
    roll_rad : float
        Roll angle in radians.

    Returns
    -------
    x_payload, y_payload, z_payload : ndarray, shape ``(N, 3)``
    """
    N = target_unit.shape[0]
    z_payload = target_unit  # boresight

    # Project celestial north onto plane ⊥ boresight
    north = np.array([0.0, 0.0, 1.0])
    north_tiled = np.broadcast_to(north, (N, 3)).copy()
    dot_nz = np.einsum("ij,ij->i", north_tiled, z_payload)[:, None]
    north_proj = north_tiled - z_payload * dot_nz

    # Handle degenerate case (boresight ≈ pole)
    proj_norm = np.linalg.norm(north_proj, axis=1, keepdims=True)
    degenerate = (proj_norm < 1e-10).squeeze()
    if np.any(degenerate):
        fallback = np.array([1.0, 0.0, 0.0])
        fallback_tiled = np.broadcast_to(fallback, (N, 3)).copy()
        dot_fz = np.einsum("ij,ij->i", fallback_tiled, z_payload)[:, None]
        fallback_proj = fallback_tiled - z_payload * dot_fz
        north_proj[degenerate] = fallback_proj[degenerate]
        proj_norm = np.linalg.norm(north_proj, axis=1, keepdims=True)

    proj_norm = np.where(proj_norm == 0, 1.0, proj_norm)
    x_ref = north_proj / proj_norm
    y_ref = np.cross(z_payload, x_ref)
    y_ref_norm = np.linalg.norm(y_ref, axis=1, keepdims=True)
    y_ref_norm = np.where(y_ref_norm == 0, 1.0, y_ref_norm)
    y_ref = y_ref / y_ref_norm

    cos_r = np.cos(roll_rad)
    sin_r = np.sin(roll_rad)
    x_payload = cos_r * x_ref + sin_r * y_ref
    y_payload = -sin_r * x_ref + cos_r * y_ref
    return x_payload, y_payload, z_payload


# ============================================================================
# Star tracker evaluation
# ============================================================================


def star_tracker_eci(
    x_payload: np.ndarray,
    y_payload: np.ndarray,
    z_payload: np.ndarray,
    st_body: np.ndarray,
) -> np.ndarray:
    """Rotate a star-tracker body-frame vector into the ECI frame.

    Parameters
    ----------
    x_payload, y_payload, z_payload : ndarray, shape ``(N, 3)``
        Payload body-frame axes expressed in ECI.
    st_body : ndarray, shape ``(3,)``
        Tracker boresight in body frame.

    Returns
    -------
    ndarray, shape ``(N, 3)``
        Tracker boresight in ECI, normalised.
    """
    st_eci = (
        x_payload * st_body[0]
        + y_payload * st_body[1]
        + z_payload * st_body[2]
    )
    return _normalise(st_eci)


def evaluate_star_tracker(
    st_eci: np.ndarray,
    sun_unit: np.ndarray,
    moon_unit: np.ndarray,
    zenith_unit: np.ndarray,
    limb_angle_rad: np.ndarray,
    sun_min_deg: float,
    moon_min_deg: float,
    earthlimb_min_deg: float,
) -> np.ndarray:
    """Check all keepout constraints for one star tracker.

    Returns an ``(N,)`` boolean array — ``True`` where the tracker passes all
    enabled (> 0) constraints.
    """
    N = st_eci.shape[0]
    ok = np.ones(N, dtype=bool)
    if sun_min_deg > 0:
        ok &= fast_sep_deg(st_eci, sun_unit) >= sun_min_deg
    if moon_min_deg > 0:
        ok &= fast_sep_deg(st_eci, moon_unit) >= moon_min_deg
    if earthlimb_min_deg > 0:
        ok &= fast_limb_deg(st_eci, zenith_unit, limb_angle_rad) >= earthlimb_min_deg
    return ok


def solar_power_fraction(
    y_payload: np.ndarray,
    sun_unit: np.ndarray,
) -> np.ndarray:
    """Fraction of maximum solar power at each timestep.

    ``P = |ŷ_payload · ŝ|`` — unity when the panel normal is aligned with the Sun.
    """
    return np.abs(np.einsum("ij,ij->i", y_payload, sun_unit))


# ============================================================================
# Orbit boundary detection
# ============================================================================


def detect_orbit_boundaries(lat_deg: np.ndarray) -> np.ndarray:
    """Identify ascending-node crossings from sub-satellite latitude.

    Parameters
    ----------
    lat_deg : ndarray, shape ``(N,)``
        Sub-satellite latitude in degrees at minute cadence.

    Returns
    -------
    ndarray of int
        Indices of ascending-node crossings (latitude crosses 0° from
        negative to positive).  The first element is always 0 (start of
        data) and the last implicit orbit extends to the end of the array.
    """
    crossings = np.where((lat_deg[:-1] < 0) & (lat_deg[1:] >= 0))[0] + 1
    if len(crossings) == 0 or crossings[0] != 0:
        crossings = np.concatenate(([0], crossings))
    return crossings


def orbit_slices_from_boundaries(boundaries: np.ndarray, N: int) -> list[slice]:
    """Convert boundary indices into a list of slices, one per orbit."""
    slices: list[slice] = []
    for i in range(len(boundaries)):
        start = int(boundaries[i])
        stop = int(boundaries[i + 1]) if i + 1 < len(boundaries) else N
        slices.append(slice(start, stop))
    return slices


# ============================================================================
# Per-orbit best-roll sweep
# ============================================================================


def _st_thresholds_active(config: PandoraSchedulerConfig) -> bool:
    """Return ``True`` if any star tracker constraint is enabled."""
    return (
        config.st_required > 0
        and (
            config.st_sun_min_deg > 0
            or config.st_moon_min_deg > 0
            or config.st_earthlimb_min_deg > 0
        )
    )


def find_best_roll_per_orbit(
    target_unit: np.ndarray,
    zenith_unit: np.ndarray,
    sun_unit: np.ndarray,
    moon_unit: np.ndarray,
    limb_angle_rad: np.ndarray,
    orbit_slices: list[slice],
    boresight_visible: np.ndarray,
    config: PandoraSchedulerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep roll angles and pick the best per orbit.

    Parameters
    ----------
    target_unit, zenith_unit, sun_unit, moon_unit : ndarray ``(N, 3)``
    limb_angle_rad : ndarray ``(N,)``
    orbit_slices : list of slice
    boresight_visible : ndarray ``(N,)`` of bool
    config : PandoraSchedulerConfig

    Returns
    -------
    roll_deg : ndarray ``(N,)``
        Best roll in degrees per minute (NaN where not visible).
    st_visible : ndarray ``(N,)`` of bool
        ``True`` where at least ``st_required`` trackers pass at best roll.
    power_frac : ndarray ``(N,)``
        Solar power fraction at the chosen roll (0 where not applicable).
    """
    N = target_unit.shape[0]
    roll_deg_out = np.full(N, np.nan)
    st_visible_out = np.zeros(N, dtype=bool)
    power_frac_out = np.zeros(N, dtype=float)

    roll_angles = np.arange(0, 360, config.roll_step_deg)
    n_rolls = len(roll_angles)

    # Per-tracker Earth-limb overrides
    st1_el = (
        config.st1_earthlimb_min_deg
        if config.st1_earthlimb_min_deg is not None
        else config.st_earthlimb_min_deg
    )
    st2_el = (
        config.st2_earthlimb_min_deg
        if config.st2_earthlimb_min_deg is not None
        else config.st_earthlimb_min_deg
    )

    for orb_slice in orbit_slices:
        bvis = boresight_visible[orb_slice]
        if not np.any(bvis):
            continue

        tgt = target_unit[orb_slice]
        zen = zenith_unit[orb_slice]
        sun = sun_unit[orb_slice]
        moon = moon_unit[orb_slice]
        limb = limb_angle_rad[orb_slice]
        n_minutes = tgt.shape[0]

        best_roll_idx = -1
        best_vis_count = -1
        best_power_mean = -1.0
        best_st_ok = None
        best_pwr = None

        for ri in range(n_rolls):
            r_rad = np.deg2rad(roll_angles[ri])
            x_pay, y_pay, z_pay = fixed_roll_attitude(tgt, r_rad)

            # Solar power filter
            pwr = solar_power_fraction(y_pay, sun)
            mean_pwr = float(np.mean(pwr[bvis]))
            if mean_pwr < config.min_power_frac:
                continue

            # Star tracker evaluations
            st1_eci = star_tracker_eci(x_pay, y_pay, z_pay, ST1_BODY)
            st1_ok = evaluate_star_tracker(
                st1_eci, sun, moon, zen, limb,
                config.st_sun_min_deg,
                config.st_moon_min_deg,
                st1_el,
            )
            st2_eci = star_tracker_eci(x_pay, y_pay, z_pay, ST2_BODY)
            st2_ok = evaluate_star_tracker(
                st2_eci, sun, moon, zen, limb,
                config.st_sun_min_deg,
                config.st_moon_min_deg,
                st2_el,
            )

            if config.st_required == 1:
                combined = st1_ok | st2_ok
            else:
                combined = st1_ok & st2_ok

            vis_count = int(np.sum(bvis & combined))
            if vis_count > best_vis_count or (
                vis_count == best_vis_count and mean_pwr > best_power_mean
            ):
                best_vis_count = vis_count
                best_power_mean = mean_pwr
                best_roll_idx = ri
                best_st_ok = combined
                best_pwr = pwr

        if best_roll_idx >= 0 and best_st_ok is not None and best_pwr is not None:
            chosen_roll_deg = roll_angles[best_roll_idx]
            # Normalise to [-180, 180)
            chosen_roll_deg = ((chosen_roll_deg + 180) % 360) - 180
            active = bvis & best_st_ok
            roll_deg_out[orb_slice] = np.where(active, chosen_roll_deg, np.nan)
            st_visible_out[orb_slice] = best_st_ok
            power_frac_out[orb_slice] = best_pwr

    return roll_deg_out, st_visible_out, power_frac_out


# ============================================================================
# Public high-level API
# ============================================================================


def compute_visibility_with_constraints(
    target_unit: np.ndarray,
    nadir_unit: np.ndarray,
    sun_unit: np.ndarray,
    moon_unit: np.ndarray,
    observer_dist_km: np.ndarray,
    zenith_unit: np.ndarray,
    limb_angle_rad: np.ndarray,
    orbit_slices: list[slice],
    earth_center_sep_deg: np.ndarray,
    config: PandoraSchedulerConfig,
) -> dict[str, np.ndarray]:
    """Run boresight + optional ST/roll constraints and return result arrays.

    This is the main entry point called by ``_build_star_visibility`` in
    :mod:`catalog`.

    Parameters
    ----------
    target_unit : ndarray ``(N, 3)``
    nadir_unit, sun_unit, moon_unit, zenith_unit : ndarray ``(N, 3)``
    observer_dist_km : ndarray ``(N,)``
    limb_angle_rad : ndarray ``(N,)``
    orbit_slices : list of slice
    earth_center_sep_deg : ndarray ``(N,)``
        Pre-computed angular separation between target and Earth centre.
    config : PandoraSchedulerConfig

    Returns
    -------
    dict with keys:
        ``visible`` : ndarray ``(N,)`` bool
        ``roll_deg`` : ndarray ``(N,)`` float (NaN where N/A)
        ``n_st_pass`` : ndarray ``(N,)`` int (0, 1, or 2)
    """
    N = target_unit.shape[0]

    # --  Phase A: Boresight constraints  -----------------------------------
    sun_sep = fast_sep_deg(target_unit, sun_unit)
    moon_sep = fast_sep_deg(target_unit, moon_unit)

    sun_ok = sun_sep > config.sun_avoidance_deg
    moon_ok = moon_sep > config.moon_avoidance_deg

    # Day/night Earth-centre avoidance
    earth_threshold = effective_earth_threshold(
        target_unit,
        nadir_unit,
        sun_unit,
        config.earth_avoidance_day_deg,
        config.earth_avoidance_night_deg,
        config.earth_avoidance_deg,
    )
    earth_ok = earth_center_sep_deg > earth_threshold

    boresight_visible = sun_ok & moon_ok & earth_ok

    # --  Phase B: Star tracker + roll sweep  --------------------------------
    roll_deg = np.full(N, np.nan)
    n_st_pass = np.zeros(N, dtype=int)

    if _st_thresholds_active(config):
        roll_deg, st_visible, power_frac = find_best_roll_per_orbit(
            target_unit,
            zenith_unit,
            sun_unit,
            moon_unit,
            limb_angle_rad,
            orbit_slices,
            boresight_visible,
            config,
        )
        visible = boresight_visible & st_visible

        # Count passing trackers at the chosen roll (for diagnostics)
        # We'd need to re-evaluate, but for now set 1 where st_visible
        n_st_pass = st_visible.astype(int)
    else:
        visible = boresight_visible

    return {
        "visible": visible,
        "roll_deg": roll_deg,
        "n_st_pass": n_st_pass,
        "sun_sep": sun_sep,
        "moon_sep": moon_sep,
        "earth_center_sep": earth_center_sep_deg,
    }
