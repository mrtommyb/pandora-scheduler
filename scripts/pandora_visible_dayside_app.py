from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from astropy.coordinates import GeocentricMeanEcliptic, get_sun
from astropy.time import Time

R_EARTH_KM = 6378.137
DEFAULT_ORBIT_ELEMENTS = {
    "inclination_deg": 97.8038,
    "raan_deg": 62.0432,
    "eccentricity": 0.0006523,
    "arg_perigee_deg": 113.5826,
    "mean_anomaly_deg": 187.8102,
}
EARTH_OBLIQUITY_DEG = 23.439281
ECLIPTIC_NORTH = np.array([0.0, 0.0, 1.0], dtype=float)
EARTH_SPIN_AXIS = np.array(
    [0.0, -np.sin(np.deg2rad(EARTH_OBLIQUITY_DEG)), np.cos(np.deg2rad(EARTH_OBLIQUITY_DEG))],
    dtype=float,
)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        raise ValueError("zero-length vector")
    return v / n


ECLIPTIC_NORTH = _unit(ECLIPTIC_NORTH)
EARTH_SPIN_AXIS = _unit(EARTH_SPIN_AXIS)


def _sun_hat_now_ecliptic(time_utc: str | Time | None = None) -> tuple[np.ndarray, Time, float, float]:
    if time_utc is None:
        t = Time.now()
    elif isinstance(time_utc, Time):
        t = time_utc
    else:
        t = Time(str(time_utc), scale="utc")
    sun_ecl = get_sun(t).transform_to(GeocentricMeanEcliptic(equinox=t))
    lon = float(sun_ecl.lon.to_value(u.rad))
    lat = float(sun_ecl.lat.to_value(u.rad))
    sun_hat = _unit(
        np.array(
            [
                np.cos(lat) * np.cos(lon),
                np.cos(lat) * np.sin(lon),
                np.sin(lat),
            ],
            dtype=float,
        )
    )
    return sun_hat, t, float(np.rad2deg(lon)) % 360.0, float(np.rad2deg(lat))


def _great_circle_points_from_normal(normal_hat: np.ndarray, npts: int = 721) -> np.ndarray:
    normal_hat = _unit(np.asarray(normal_hat, dtype=float))
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(ref, normal_hat))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    e1 = _unit(ref - normal_hat * float(np.dot(ref, normal_hat)))
    e2 = _unit(np.cross(normal_hat, e1))
    tt = np.linspace(0.0, 2.0 * np.pi, npts)
    return np.cos(tt)[:, None] * e1[None, :] + np.sin(tt)[:, None] * e2[None, :]


def _target_from_radec(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = np.deg2rad(float(ra_deg))
    dec = np.deg2rad(float(dec_deg))
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return _unit(np.array([x, y, z], dtype=float))


def _solve_kepler_eccentric_anomaly(M_rad: float, ecc: float, max_iter: int = 30) -> float:
    E = M_rad if ecc < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - ecc * np.sin(E) - M_rad
        fp = 1.0 - ecc * np.cos(E)
        dE = f / fp
        E -= dE
        if abs(float(dE)) < 1e-12:
            break
    return float(E)


def _sc_state_from_elements(
    alt_km: float,
    mean_anomaly_offset_deg: float = 0.0,
    elements: dict | None = None,
) -> np.ndarray:
    if elements is None:
        elements = DEFAULT_ORBIT_ELEMENTS
    inc = np.deg2rad(float(elements["inclination_deg"]))
    raan = np.deg2rad(float(elements["raan_deg"]))
    ecc = float(elements["eccentricity"])
    argp = np.deg2rad(float(elements["arg_perigee_deg"]))
    M0 = float(elements["mean_anomaly_deg"])
    M = np.deg2rad(M0 + float(mean_anomaly_offset_deg))
    M = float((M + np.pi) % (2.0 * np.pi) - np.pi)

    a = float(R_EARTH_KM + alt_km)
    E = _solve_kepler_eccentric_anomaly(M, ecc)
    r = a * (1.0 - ecc * np.cos(E))
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(E / 2.0),
        np.sqrt(1.0 - ecc) * np.cos(E / 2.0),
    )
    r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0], dtype=float)

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    Q = np.array(
        [
            [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
            [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
            [sw * si, cw * si, ci],
        ],
        dtype=float,
    )
    return Q @ r_pf


def _compute_pr7_n_hat(sc_vec_km: np.ndarray, target_hat: np.ndarray) -> np.ndarray:
    zenith = _unit(sc_vec_km)
    proj = target_hat - zenith * float(np.dot(target_hat, zenith))
    if float(np.linalg.norm(proj)) < 1e-12:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(ref, zenith))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        proj = ref - zenith * float(np.dot(ref, zenith))
    limb_dir = _unit(proj)
    r_sc = float(np.linalg.norm(sc_vec_km))
    limb_angle = float(np.arccos(np.clip(R_EARTH_KM / r_sc, -1.0, 1.0)))
    return _unit(np.cos(limb_angle) * zenith + np.sin(limb_angle) * limb_dir)


def _earthlimb_is_sunlit_pr7(sc_vec_km: np.ndarray, target_hat: np.ndarray, sun_hat: np.ndarray) -> bool:
    """Day/night classification for the nearest Earth limb."""
    n_hat = _compute_pr7_n_hat(sc_vec_km, target_hat)
    return bool(float(np.dot(n_hat, sun_hat)) > 0.0)


def _earth_center_sep_deg(sc_vec_km: np.ndarray, target_hat: np.ndarray) -> float:
    nadir = -_unit(sc_vec_km)
    return float(np.rad2deg(np.arccos(np.clip(np.dot(target_hat, nadir), -1.0, 1.0))))


def _effective_earth_threshold_pr7(
    sc_vec_km: np.ndarray,
    target_hat: np.ndarray,
    sun_hat: np.ndarray,
    default_deg: float,
    day_deg: float | None,
    night_deg: float | None,
) -> tuple[float, bool]:
    sunlit = _earthlimb_is_sunlit_pr7(sc_vec_km, target_hat, sun_hat)
    if day_deg is None and night_deg is None:
        return float(default_deg), sunlit
    day = float(default_deg if day_deg is None else day_deg)
    night = float(default_deg if night_deg is None else night_deg)
    return (day if sunlit else night), sunlit


def _nearest_limb_tangent_direction(sc_vec_km: np.ndarray, target_hat: np.ndarray) -> np.ndarray:
    """Return the spacecraft-to-limb tangent direction nearest the target LOS."""
    zenith = _unit(sc_vec_km)
    nadir = -zenith
    proj = target_hat - zenith * float(np.dot(target_hat, zenith))
    if float(np.linalg.norm(proj)) < 1e-12:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(ref, zenith))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        proj = ref - zenith * float(np.dot(ref, zenith))
    limb_dir = _unit(proj)
    r_sc = float(np.linalg.norm(sc_vec_km))
    earth_angular_radius = float(np.arcsin(np.clip(R_EARTH_KM / r_sc, -1.0, 1.0)))
    return _unit(np.cos(earth_angular_radius) * nadir + np.sin(earth_angular_radius) * limb_dir)


def _panel_mask(
    i: int,
    j: int,
    sc_hat: np.ndarray,
    sc_u: np.ndarray,
    sun_hat: np.ndarray,
    target_hat: np.ndarray,
    thr: float,
    ngrid: int = 260,
):
    k = 3 - i - j
    ugrid = np.linspace(-1.0, 1.0, ngrid)
    vgrid = np.linspace(-1.0, 1.0, ngrid)
    U, V = np.meshgrid(ugrid, vgrid)
    R2 = U * U + V * V
    inside = R2 <= 1.0
    W = np.sqrt(np.clip(1.0 - R2, 0.0, None))

    d_front = np.abs(sc_hat[k] - W)
    d_back = np.abs(sc_hat[k] + W)
    depth = np.where(d_front <= d_back, W, -W)

    n = np.zeros(U.shape + (3,), dtype=float)
    n[..., i] = U
    n[..., j] = V
    n[..., k] = depth

    dot_s = np.einsum("...i,i->...", n, sun_hat)
    dot_sc = np.einsum("...i,i->...", n, sc_hat)

    d = n - sc_u[None, None, :]
    d_norm = np.linalg.norm(d, axis=2)
    d_norm = np.where(d_norm <= 1e-12, 1.0, d_norm)
    d_hat = d / d_norm[..., None]
    dot_t = np.einsum("...i,i->...", d_hat, target_hat)

    mask = inside & (dot_s >= 0.0) & (dot_sc >= thr) & (dot_t >= 0.0)
    return U, V, mask


def _plot_simple_geometry(
    target_ra_deg: float,
    target_dec_deg: float,
    alt_km: float,
    mean_anomaly_offset_deg: float,
    earth_avoidance_default_deg: float,
    earth_avoidance_day_deg: float | None,
    earth_avoidance_night_deg: float | None,
    show_earth_frame: bool = True,
    time_utc: str | Time | None = None,
):
    sun_hat, sun_time, sun_lon_deg, sun_lat_deg = _sun_hat_now_ecliptic(time_utc=time_utc)
    target_hat = _target_from_radec(target_ra_deg, target_dec_deg)

    sc_vec_km = _sc_state_from_elements(alt_km=alt_km, mean_anomaly_offset_deg=mean_anomaly_offset_deg)
    sc_hat = _unit(sc_vec_km)
    sc_u = sc_vec_km / R_EARTH_KM
    thr = float(R_EARTH_KM / np.linalg.norm(sc_vec_km))

    n_hat = _compute_pr7_n_hat(sc_vec_km, target_hat)
    dot_ns = float(np.dot(n_hat, sun_hat))
    earth_center_sep_deg = _earth_center_sep_deg(sc_vec_km, target_hat)
    earth_threshold_deg, earthlimb_sunlit_pr7 = _effective_earth_threshold_pr7(
        sc_vec_km,
        target_hat,
        sun_hat,
        earth_avoidance_default_deg,
        earth_avoidance_day_deg,
        earth_avoidance_night_deg,
    )
    earth_ok = earth_center_sep_deg > earth_threshold_deg

    rng = np.random.default_rng(0)
    n_samp = rng.normal(size=(30000, 3))
    n_samp /= np.linalg.norm(n_samp, axis=1, keepdims=True)
    visible = (n_samp @ sc_hat) >= thr
    dayside = (n_samp @ sun_hat) >= 0.0
    d = n_samp - sc_u[None, :]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    toward_target = (d @ target_hat) >= 0.0
    frac_visible = float(visible.mean())
    frac_dayside_visible = float((visible & dayside).mean())
    frac_dayside_visible_toward = float((visible & dayside & toward_target).mean())

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    planes = [(0, 1, "X", "Y", "XY"), (0, 2, "X", "Z", "XZ"), (1, 2, "Y", "Z", "YZ")]
    viewer_by_title = {
        "XY": np.array([0.0, 0.0, 1.0], dtype=float),
        "XZ": np.array([0.0, 1.0, 0.0], dtype=float),
        "YZ": np.array([1.0, 0.0, 0.0], dtype=float),
    }

    orbit_dm = np.linspace(-180.0, 180.0, 2401)
    orbit_u = np.array([
        _sc_state_from_elements(alt_km=alt_km, mean_anomaly_offset_deg=float(dm)) / R_EARTH_KM
        for dm in orbit_dm
    ])

    def _plot_masked_line(ax, x, y, mask, linestyle):
        xx = np.where(mask, x, np.nan)
        yy = np.where(mask, y, np.nan)
        ax.plot(xx, yy, color="black", linewidth=0.9, linestyle=linestyle, alpha=0.9, zorder=5)

    for idx, (i, j, xlab, ylab, title) in enumerate(planes):
        ax = axes[idx]
        U, V, mask = _panel_mask(i, j, sc_hat, sc_u, sun_hat, target_hat, thr)
        ax.contourf(U, V, np.where(mask, 1.0, np.nan), levels=[0.5, 1.5], colors=["#9fd8ff"], alpha=0.55, zorder=1)
        ax.add_patch(plt.Circle((0.0, 0.0), 1.0, fill=False, color="gray", linewidth=1.5, zorder=3))

        if show_earth_frame:
            def draw_axis_arrow(vec3, color, label):
                d2 = np.array([vec3[i], vec3[j]], dtype=float)
                dn = float(np.linalg.norm(d2))
                if dn < 1e-12:
                    return
                if title == "XY":
                    end = 0.95 * d2
                    text_pos = 1.02 * d2
                else:
                    d2 = d2 / dn
                    end = 0.95 * d2
                    text_pos = 1.02 * d2
                ax.arrow(
                    0.0,
                    0.0,
                    end[0],
                    end[1],
                    color=color,
                    linewidth=1.4,
                    linestyle=":",
                    length_includes_head=True,
                    head_width=0.035,
                    head_length=0.05,
                    zorder=7,
                )
                ax.text(text_pos[0], text_pos[1], label[0], color=color, fontsize=9, zorder=7)

            draw_axis_arrow(EARTH_SPIN_AXIS, "#8a2be2", label="Earth")
            equator_pts = _great_circle_points_from_normal(EARTH_SPIN_AXIS)
            ax.plot(equator_pts[:, i], equator_pts[:, j], color="#8a2be2", linestyle="--", linewidth=0.8, alpha=0.8, zorder=3)

        ox = orbit_u[:, i]
        oy = orbit_u[:, j]
        view_hat = viewer_by_title[title]
        orbit_depth = orbit_u @ view_hat
        in_disk = (ox * ox + oy * oy) <= 1.0
        hidden = (orbit_depth < 0.0) & in_disk
        shown = ~hidden
        _plot_masked_line(ax, ox, oy, shown, "-")
        _plot_masked_line(ax, ox, oy, hidden, "--")

        term_pts = _great_circle_points_from_normal(sun_hat)
        term_depth = term_pts @ view_hat
        term_visible = term_depth >= -1e-9
        ax.plot(np.where(term_visible, term_pts[:, i], np.nan), np.where(term_visible, term_pts[:, j], np.nan), color="#2e5fbf", linewidth=2.0, alpha=0.9, zorder=4)
        ax.plot(np.where(~term_visible, term_pts[:, i], np.nan), np.where(~term_visible, term_pts[:, j], np.nan), color="#2e5fbf", linewidth=1.6, linestyle="--", alpha=0.9, zorder=4)

        S = np.array([sc_u[i], sc_u[j]], dtype=float)
        ax.scatter([S[0]], [S[1]], color="black", s=40, marker="s", zorder=6)
        ax.text(S[0] + 0.03, S[1] + 0.03, "Pandora", color="black", fontsize=14)

        def draw_from_sc(vec3, color):
            d2 = np.array([vec3[i], vec3[j]], dtype=float)
            if float(np.linalg.norm(d2)) < 1e-12:
                return
            end = S + 0.50 * d2
            ax.arrow(S[0], S[1], end[0] - S[0], end[1] - S[1], color=color, linewidth=2.0, length_includes_head=True, head_width=0.03, head_length=0.05, zorder=7)

        draw_from_sc(sun_hat, "gold")
        draw_from_sc(target_hat, "red")



        ax.axhline(0.0, color="lightgray", linewidth=0.8)
        ax.axvline(0.0, color="lightgray", linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")
        if title == "XZ":
            ax.set_xlim(1.4, -1.4)
        else:
            ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    axes[0].plot([], [], color="#9fd8ff", linewidth=8, alpha=0.55, label="dayside toward target")
    axes[0].plot([], [], color="#2e5fbf", linewidth=2.0, label="terminator")
    axes[0].plot([], [], color="#2e5fbf", linewidth=1.6, linestyle="--")
    axes[0].plot([], [], color="black", linewidth=0.9, linestyle="-")#, label="orbit")
    axes[0].plot([], [], color="black", linewidth=0.9, linestyle="--")
    if show_earth_frame:
        axes[0].plot([], [], color="#8a2be2", linewidth=0.8, linestyle="--", label="Earth equator & axis")
    axes[0].legend(loc="lower left", fontsize=14)

    fig.suptitle(
        f"UTC = {sun_time.isot[:19]}\n"
        f"Target(RA, Dec) = ({target_ra_deg:.0f} deg, {target_dec_deg:+.0f} deg)\n"
        f"dayside visible (total) = {100.0 * frac_dayside_visible:.1f}% | dayside visible (toward-target) = {100.0 * frac_dayside_visible_toward:.1f}%",
        fontsize=16,
    )
    plt.tight_layout()
    return fig, {
        "dayside_visible_toward_target": frac_dayside_visible_toward,
        "dot_ns": dot_ns,
        "earthlimb_sunlit_pr7": earthlimb_sunlit_pr7,
        "earth_center_sep_deg": earth_center_sep_deg,
        "earth_threshold_deg": earth_threshold_deg,
        "earth_ok": earth_ok,
    }


st.set_page_config(page_title="Pandora Visible Dayside", layout="wide")
st.title("Pandora Visible Dayside")

if "use_now" not in st.session_state:
    st.session_state.use_now = True
if "manual_utc_time" not in st.session_state:
    st.session_state.manual_utc_time = Time.now().utc.isot[:19]
if "target_ra_deg" not in st.session_state:
    st.session_state.target_ra_deg = 145.0
if "target_dec_deg" not in st.session_state:
    st.session_state.target_dec_deg = 10.0
if "target_ra_slider" not in st.session_state:
    st.session_state.target_ra_slider = float(st.session_state.target_ra_deg)
if "target_ra_input" not in st.session_state:
    st.session_state.target_ra_input = float(st.session_state.target_ra_deg)
if "target_dec_slider" not in st.session_state:
    st.session_state.target_dec_slider = float(st.session_state.target_dec_deg)
if "target_dec_input" not in st.session_state:
    st.session_state.target_dec_input = float(st.session_state.target_dec_deg)

def _sync_target_ra_from_slider() -> None:
    value = float(st.session_state.target_ra_slider)
    st.session_state.target_ra_deg = value
    st.session_state.target_ra_input = value

def _sync_target_ra_from_input() -> None:
    value = float(st.session_state.target_ra_input)
    st.session_state.target_ra_deg = value
    st.session_state.target_ra_slider = value

def _sync_target_dec_from_slider() -> None:
    value = float(st.session_state.target_dec_slider)
    st.session_state.target_dec_deg = value
    st.session_state.target_dec_input = value

def _sync_target_dec_from_input() -> None:
    value = float(st.session_state.target_dec_input)
    st.session_state.target_dec_deg = value
    st.session_state.target_dec_slider = value

with st.sidebar:
    st.slider(
        "Target RA [deg]",
        min_value=0.0,
        max_value=360.0,
        step=1.0,
        key="target_ra_slider",
        on_change=_sync_target_ra_from_slider,
    )
    target_ra_deg = st.number_input(
        "Target RA input [deg]",
        min_value=0.0,
        max_value=360.0,
        step=1.0,
        format="%.3f",
        key="target_ra_input",
        on_change=_sync_target_ra_from_input,
    )
    st.slider(
        "Target Dec [deg]",
        min_value=-90.0,
        max_value=90.0,
        step=1.0,
        key="target_dec_slider",
        on_change=_sync_target_dec_from_slider,
    )
    target_dec_deg = st.number_input(
        "Target Dec input [deg]",
        min_value=-90.0,
        max_value=90.0,
        step=1.0,
        format="%.3f",
        key="target_dec_input",
        on_change=_sync_target_dec_from_input,
    )
    mean_anomaly_offset_deg = st.slider("Mean anomaly Pandora [deg]", min_value=-180.0, max_value=180.0, value=150.0, step=5.0)
    show_earth_frame = st.checkbox("Show Earth equator & axis", value=False)
    use_now = st.checkbox("Use time = now", key="use_now")
    if use_now:
        time_utc = None
    else:
        time_utc = st.text_input("UTC time", value=st.session_state.manual_utc_time, key="manual_utc_time")

    st.markdown("---")
    st.subheader("Earth Keepout")
    earth_avoidance_default_deg = st.number_input("Earth keepout default [deg]", value=40.0, step=1.0, format="%.1f")
    use_day_night_earth = st.checkbox("Use separate day/night Earth keepout", value=False)
    if use_day_night_earth:
        earth_avoidance_day_deg = st.number_input("Earth keepout day [deg]", value=40.0, step=1.0, format="%.1f")
        earth_avoidance_night_deg = st.number_input("Earth keepout night [deg]", value=40.0, step=1.0, format="%.1f")
    else:
        earth_avoidance_day_deg = None
        earth_avoidance_night_deg = None

    st.markdown("---")
    st.subheader("Orbital Elements")
    alt_km = st.number_input("Altitude [km]", min_value=400.0, max_value=2000.0, value=600.0, step=10.0, format="%.1f")
    inclination_deg = st.number_input("Inclination [deg]", value=float(DEFAULT_ORBIT_ELEMENTS['inclination_deg']), format="%.4f")
    raan_deg = st.number_input("RAAN [deg]", value=float(DEFAULT_ORBIT_ELEMENTS['raan_deg']), format="%.4f")
    eccentricity = st.number_input("Eccentricity", min_value=0.0, max_value=0.99, value=float(DEFAULT_ORBIT_ELEMENTS['eccentricity']), format="%.7f")
    arg_perigee_deg = st.number_input("Argument of perigee [deg]", value=float(DEFAULT_ORBIT_ELEMENTS['arg_perigee_deg']), format="%.4f")
    mean_anomaly_deg = st.number_input("Reference mean anomaly [deg]", value=float(DEFAULT_ORBIT_ELEMENTS['mean_anomaly_deg']), format="%.4f")

custom_elements = {
    'inclination_deg': inclination_deg,
    'raan_deg': raan_deg,
    'eccentricity': eccentricity,
    'arg_perigee_deg': arg_perigee_deg,
    'mean_anomaly_deg': mean_anomaly_deg,
}

# Override the module default used by the plotting helpers.
DEFAULT_ORBIT_ELEMENTS.update(custom_elements)

fig, metrics = _plot_simple_geometry(
    target_ra_deg=float(target_ra_deg),
    target_dec_deg=float(target_dec_deg),
    alt_km=alt_km,
    mean_anomaly_offset_deg=mean_anomaly_offset_deg,
    earth_avoidance_default_deg=float(earth_avoidance_default_deg),
    earth_avoidance_day_deg=(None if earth_avoidance_day_deg is None else float(earth_avoidance_day_deg)),
    earth_avoidance_night_deg=(None if earth_avoidance_night_deg is None else float(earth_avoidance_night_deg)),
    show_earth_frame=show_earth_frame,
    time_utc=time_utc,
)

if not metrics["earth_ok"]:
    st.markdown(
        f"""
        <div style="background-color:#ffe5e5;border-left:6px solid #cc0000;padding:0.75rem 1rem;border-radius:0.4rem;color:#660000;font-weight:600;">
            Earth keepout failed: Earth-center separation {metrics['earth_center_sep_deg']:.1f} deg is below the
            active threshold {metrics['earth_threshold_deg']:.1f} deg.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div style="background-color:#e8f6ea;border-left:6px solid #2d7a46;padding:0.75rem 1rem;border-radius:0.4rem;color:#1f4f2d;font-weight:600;">
            Earth keepout passed: Earth-center separation {metrics['earth_center_sep_deg']:.1f} deg exceeds the
            active threshold {metrics['earth_threshold_deg']:.1f} deg.
        </div>
        """,
        unsafe_allow_html=True,
    )

figure_slot = st.empty()
download_slot = st.empty()

figure_slot.pyplot(fig, clear_figure=True)

buf = BytesIO()
fig.savefig(buf, format='png', dpi=180)
buf.seek(0)

download_slot.download_button(
    label="Download figure as PNG",
    data=buf.getvalue(),
    file_name="pandora_visible_dayside.png",
    mime="image/png",
)
