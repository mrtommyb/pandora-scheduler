"""Unit tests for visibility constraint primitives.

Tests cover the geometric functions ported from pandoravisibility:
day/night Earth detection, star tracker keepout, roll attitude, and
per-orbit roll sweep.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from pandorascheduler_rework.config import PandoraSchedulerConfig
from pandorascheduler_rework.visibility.constraints import (
    ST1_BODY,
    ST2_BODY,
    _normalise,
    detect_orbit_boundaries,
    earthlimb_is_sunlit,
    effective_earth_threshold,
    evaluate_star_tracker,
    fast_limb_deg,
    fast_sep_deg,
    find_best_roll_per_orbit,
    fixed_roll_attitude,
    orbit_slices_from_boundaries,
    solar_power_fraction,
    star_tracker_eci,
    sun_constrained_attitude,
    compute_visibility_with_constraints,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> PandoraSchedulerConfig:
    """Create a test config with sensible defaults; override any field."""
    defaults = dict(
        window_start=datetime(2026, 2, 5),
        window_end=datetime(2027, 2, 5),
    )
    defaults.update(overrides)
    return PandoraSchedulerConfig(**defaults)


def _unit(v):
    """Create a unit vector from v."""
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _tile(v, n):
    """Tile a 1-D vector into (n, 3)."""
    return np.broadcast_to(np.asarray(v, dtype=float), (n, 3)).copy()


# ============================================================================
# fast_sep_deg
# ============================================================================


class TestFastSepDeg:
    def test_zero_separation(self):
        a = _tile([1, 0, 0], 3)
        b = _tile([1, 0, 0], 3)
        assert np.allclose(fast_sep_deg(a, b), 0.0)

    def test_ninety_degrees(self):
        a = _tile([1, 0, 0], 2)
        b = _tile([0, 1, 0], 2)
        result = fast_sep_deg(a, b)
        assert np.allclose(result, 90.0)

    def test_opposite(self):
        a = _tile([1, 0, 0], 1)
        b = _tile([-1, 0, 0], 1)
        assert np.allclose(fast_sep_deg(a, b), 180.0)

    def test_arbitrary_angle(self):
        # 60° between two vectors
        a = _tile([1, 0, 0], 1)
        b = _tile([0.5, np.sqrt(3) / 2, 0], 1)
        assert np.allclose(fast_sep_deg(a, b), 60.0, atol=1e-10)


# ============================================================================
# fast_limb_deg
# ============================================================================


class TestFastLimbDeg:
    def test_target_at_zenith(self):
        """Target directly above observer → max limb separation."""
        zenith = _tile([0, 0, 1], 1)
        target = _tile([0, 0, 1], 1)
        limb_rad = np.array([np.deg2rad(60)])  # ~60° Earth half-angle
        result = fast_limb_deg(target, zenith, limb_rad)
        # elev = arcsin(1) = 90°, result = 90 + 60 = 150
        assert np.allclose(result, 150.0)

    def test_target_at_horizon(self):
        """Target perpendicular to zenith → elev = 0."""
        zenith = _tile([0, 0, 1], 1)
        target = _tile([1, 0, 0], 1)
        limb_rad = np.array([np.deg2rad(60)])
        result = fast_limb_deg(target, zenith, limb_rad)
        # elev = arcsin(0) = 0°, result = 0 + 60 = 60
        assert np.allclose(result, 60.0)

    def test_target_below_horizon(self):
        """Target toward nadir → negative elevation, limb angle < limb_rad."""
        zenith = _tile([0, 0, 1], 1)
        target = _tile([0, 0, -1], 1)
        limb_rad = np.array([np.deg2rad(60)])
        result = fast_limb_deg(target, zenith, limb_rad)
        # elev = -90°, result = -90 + 60 = -30
        assert result[0] < 0


# ============================================================================
# earthlimb_is_sunlit
# ============================================================================


class TestEarthlimbIsSunlit:
    """Test day/night limb classification."""

    def test_sunlit_limb(self):
        """Target and Sun on the same side → nearest limb is sunlit."""
        nadir = _tile([0, 0, -1], 1)   # observer above Earth
        target = _tile([1, 0, 0], 1)   # target toward +X
        sun = _tile([1, 0, 0], 1)      # Sun same direction
        assert earthlimb_is_sunlit(target, nadir, sun)[0]

    def test_dark_limb(self):
        """Target and Sun on opposite sides → nearest limb is dark."""
        nadir = _tile([0, 0, -1], 1)
        target = _tile([1, 0, 0], 1)
        sun = _tile([-1, 0, 0], 1)
        assert not earthlimb_is_sunlit(target, nadir, sun)[0]

    def test_array(self):
        """Vectorised over multiple timesteps."""
        nadir = _tile([0, 0, -1], 3)
        target = _tile([1, 0, 0], 3)
        sun = np.array([
            [1, 0, 0],     # sunlit
            [-1, 0, 0],    # dark
            [0, 1, 0],     # perpendicular — limb is at +X, Sun at +Y, dot > 0? No.
        ], dtype=float)
        sun = _normalise(sun)
        result = earthlimb_is_sunlit(target, nadir, sun)
        assert result[0] is np.True_
        assert result[1] is np.False_


# ============================================================================
# effective_earth_threshold
# ============================================================================


class TestEffectiveEarthThreshold:
    def test_uniform_when_none(self):
        """Both day/night None → returns scalar default."""
        target = _tile([1, 0, 0], 2)
        nadir = _tile([0, 0, -1], 2)
        sun = _tile([1, 0, 0], 2)
        result = effective_earth_threshold(target, nadir, sun, None, None, 86.0)
        assert result == 86.0

    def test_day_night_split(self):
        """Explicit day/night values → per-timestep array."""
        nadir = _tile([0, 0, -1], 2)
        target = _tile([1, 0, 0], 2)
        sun = np.array([
            _unit([1, 0, 0]),    # sunlit → day
            _unit([-1, 0, 0]),   # dark   → night
        ])
        result = effective_earth_threshold(target, nadir, sun, 110.0, 80.0, 86.0)
        assert np.isclose(result[0], 110.0)
        assert np.isclose(result[1], 80.0)


# ============================================================================
# Attitude construction
# ============================================================================


class TestSunConstrainedAttitude:
    def test_orthonormal(self):
        """Payload axes must be orthonormal."""
        target = _tile(_unit([1, 1, 0]), 5)
        sun = _tile(_unit([0, 0, 1]), 5)
        x, y, z = sun_constrained_attitude(target, sun)
        for i in range(5):
            frame = np.stack([x[i], y[i], z[i]])
            assert np.allclose(frame @ frame.T, np.eye(3), atol=1e-12)

    def test_boresight_is_target(self):
        target = _tile(_unit([0, 0, 1]), 1)
        sun = _tile(_unit([1, 0, 0]), 1)
        _, _, z = sun_constrained_attitude(target, sun)
        assert np.allclose(z[0], _unit([0, 0, 1]))


class TestFixedRollAttitude:
    def test_orthonormal_at_various_rolls(self):
        target = _tile(_unit([1, 0, 0]), 5)
        for roll_deg in [0, 45, 90, 180, 270]:
            x, y, z = fixed_roll_attitude(target, np.deg2rad(roll_deg))
            for i in range(5):
                frame = np.stack([x[i], y[i], z[i]])
                assert np.allclose(frame @ frame.T, np.eye(3), atol=1e-12)

    def test_zero_roll_aligns_with_north(self):
        """At roll = 0, X_payload should be the projection of +Z onto the
        plane perpendicular to the boresight."""
        target = _tile(_unit([1, 0, 0]), 1)  # boresight along +X
        x, y, z = fixed_roll_attitude(target, 0.0)
        # North = [0,0,1], projection onto plane perp to [1,0,0] = [0,0,1]
        assert np.allclose(x[0], [0, 0, 1], atol=1e-12)


# ============================================================================
# Star tracker ECI transform
# ============================================================================


class TestStarTrackerECI:
    def test_identity_frame(self):
        """When payload frame = ECI axes, ST body vector comes through unchanged."""
        x = _tile([1, 0, 0], 1)
        y = _tile([0, 1, 0], 1)
        z = _tile([0, 0, 1], 1)
        st_body = np.array([1.0, 0.0, 0.0])
        result = star_tracker_eci(x, y, z, st_body)
        assert np.allclose(result[0], [1, 0, 0], atol=1e-12)

    def test_st1_st2_symmetric(self):
        """ST1 and ST2 should be symmetric about the XZ plane."""
        assert np.allclose(ST1_BODY[0], ST2_BODY[0])
        assert np.allclose(ST1_BODY[1], -ST2_BODY[1])
        assert np.allclose(ST1_BODY[2], ST2_BODY[2])


# ============================================================================
# evaluate_star_tracker
# ============================================================================


class TestEvaluateStarTracker:
    def test_all_disabled(self):
        """All thresholds = 0 → always passes."""
        st = _tile(_unit([1, 0, 0]), 3)
        sun = _tile(_unit([0, 1, 0]), 3)
        moon = _tile(_unit([0, 0, 1]), 3)
        zenith = _tile(_unit([0, 0, 1]), 3)
        limb = np.full(3, np.deg2rad(60))
        ok = evaluate_star_tracker(st, sun, moon, zenith, limb, 0.0, 0.0, 0.0)
        assert np.all(ok)

    def test_sun_constraint_fail(self):
        """ST pointing near Sun should fail."""
        st = _tile(_unit([1, 0, 0]), 1)
        sun = _tile(_unit([1, 0.01, 0]), 1)  # ~0.5° from ST
        moon = _tile(_unit([0, 0, 1]), 1)
        zenith = _tile(_unit([0, 0, 1]), 1)
        limb = np.full(1, np.deg2rad(60))
        ok = evaluate_star_tracker(st, sun, moon, zenith, limb, 45.0, 0.0, 0.0)
        assert not ok[0]


# ============================================================================
# solar_power_fraction
# ============================================================================


class TestSolarPowerFraction:
    def test_max_power(self):
        """Y axis perpendicular to Sun → panels face Sun → power = 1."""
        y = _tile([0, 1, 0], 1)
        sun = _tile([1, 0, 0], 1)
        assert np.allclose(solar_power_fraction(y, sun), 1.0)

    def test_zero_power(self):
        """Y axis aligned with Sun → panels edge-on → power = 0."""
        y = _tile([0, 1, 0], 1)
        sun = _tile([0, 1, 0], 1)
        assert np.allclose(solar_power_fraction(y, sun), 0.0, atol=1e-12)


# ============================================================================
# orbit boundaries
# ============================================================================


class TestOrbitBoundaries:
    def test_simple_sinusoid(self):
        """Ascending nodes of a sinusoidal latitude."""
        t = np.linspace(0, 10 * np.pi, 1000)
        lat = np.sin(t) * 50  # 5 full cycles
        boundaries = detect_orbit_boundaries(lat)
        # 5 ascending crossings + index 0
        assert boundaries[0] == 0
        assert len(boundaries) >= 5

    def test_single_orbit(self):
        lat = np.sin(np.linspace(0, 2 * np.pi, 100)) * 50
        boundaries = detect_orbit_boundaries(lat)
        slices = orbit_slices_from_boundaries(boundaries, 100)
        # At least covers all data
        total = sum(s.stop - s.start for s in slices)
        assert total == 100


# ============================================================================
# find_best_roll_per_orbit
# ============================================================================


class TestFindBestRollPerOrbit:
    def test_st_required_zero_skips(self):
        """When st_required=0, no roll sweep should happen."""
        config = _make_config(st_required=0)
        N = 10
        target = _tile(_unit([1, 0, 0]), N)
        zenith = _tile(_unit([0, 0, 1]), N)
        sun = _tile(_unit([0, 1, 0]), N)
        moon = _tile(_unit([0, 0, 1]), N)
        limb = np.full(N, np.deg2rad(60))
        bvis = np.ones(N, dtype=bool)

        # The function should still be callable; results don't matter when
        # the caller gates on _st_thresholds_active.
        roll, st_vis, pwr = find_best_roll_per_orbit(
            target, zenith, sun, moon, limb,
            [slice(0, N)], bvis, config,
        )
        assert roll.shape == (N,)

    def test_finds_a_roll(self):
        """With mild ST constraints, the sweep should find a valid roll."""
        config = _make_config(
            st_sun_min_deg=30.0,
            st_moon_min_deg=0.0,
            st_earthlimb_min_deg=0.0,
            st_required=1,
            roll_step_deg=10.0,
            min_power_frac=0.0,
        )
        N = 20
        # Target along +X, Sun along +Y (90° separation — comfortable)
        target = _tile(_unit([1, 0, 0]), N)
        zenith = _tile(_unit([0, 0, 1]), N)
        sun = _tile(_unit([0, 1, 0]), N)
        moon = _tile(_unit([0, 0, -1]), N)
        limb = np.full(N, np.deg2rad(60))
        bvis = np.ones(N, dtype=bool)

        roll, st_vis, pwr = find_best_roll_per_orbit(
            target, zenith, sun, moon, limb,
            [slice(0, N)], bvis, config,
        )
        # Should find at least some minutes with ST pass
        assert np.any(st_vis)
        assert np.any(~np.isnan(roll))


# ============================================================================
# compute_visibility_with_constraints (integration)
# ============================================================================


class TestComputeVisibilityIntegration:
    def test_uniform_earth_avoidance(self):
        """With day/night=None, should replicate simple threshold."""
        config = _make_config(
            earth_avoidance_deg=90.0,
            earth_avoidance_day_deg=None,
            earth_avoidance_night_deg=None,
            st_required=0,
        )
        N = 5
        target = _tile(_unit([1, 0, 0]), N)
        nadir = _tile(_unit([0, 0, -1]), N)
        sun = _tile(_unit([0, 1, 0]), N)
        moon = _tile(_unit([0, 0, 1]), N)
        zenith = -nadir
        dist_km = np.full(N, 6971.0)
        limb = np.arccos(6371.0 / dist_km)
        # Earth centre separation: angle between nadir and target
        # nadir=[0,0,-1], target=[1,0,0] → 90°. With threshold 90° → NOT visible (must be >)
        earth_sep = fast_sep_deg(nadir, target)
        assert np.allclose(earth_sep, 90.0)

        result = compute_visibility_with_constraints(
            target, nadir, sun, moon, dist_km, zenith, limb,
            [slice(0, N)], earth_sep, config,
        )
        assert not np.any(result["visible"])

    def test_daynight_makes_difference(self):
        """Day = 110, night = 80 should produce different visibility."""
        config = _make_config(
            earth_avoidance_deg=110.0,
            earth_avoidance_day_deg=110.0,
            earth_avoidance_night_deg=80.0,
            st_required=0,
        )
        N = 2
        # Target at 95° from Earth centre — passes night (80), fails day (110)
        # Set up: nadir along -Z, target slightly off +X
        nadir = _tile(_unit([0, 0, -1]), N)
        zenith = -nadir
        # Make target vector such that angle from nadir ≈ 95°
        # nadir = [0,0,-1], we want dot(nadir, target) = cos(95°) ≈ -0.087
        theta = np.deg2rad(95)
        target = _tile([np.sin(theta), 0, -np.cos(theta)], N)
        target = _normalise(target)

        # Sun on same side as target (sunlit) for first timestep,
        # opposite for second (dark)
        sun = np.array([
            _unit([1, 0, 0]),    # sunlit limb → day threshold 110 → FAIL (95 < 110)
            _unit([-1, 0, 0]),   # dark limb   → night threshold 80 → PASS (95 > 80)
        ])
        moon = _tile(_unit([0, 1, 0]), N)
        dist_km = np.full(N, 6971.0)
        limb = np.arccos(6371.0 / dist_km)
        earth_sep = fast_sep_deg(nadir, target)

        result = compute_visibility_with_constraints(
            target, nadir, sun, moon, dist_km, zenith, limb,
            [slice(0, N)], earth_sep, config,
        )
        # Daytime (sunlit limb) should fail, nighttime should pass
        assert not result["visible"][0]
        assert result["visible"][1]


# ============================================================================
# Config validation
# ============================================================================


class TestConfigValidation:
    def test_st_required_invalid(self):
        with pytest.raises(ValueError, match="st_required"):
            _make_config(st_required=3)

    def test_roll_step_zero(self):
        with pytest.raises(ValueError, match="roll_step_deg"):
            _make_config(roll_step_deg=0.0)

    def test_min_power_frac_range(self):
        with pytest.raises(ValueError, match="min_power_frac"):
            _make_config(min_power_frac=1.5)
