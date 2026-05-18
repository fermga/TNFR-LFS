"""Physics-derived columns on top of a captured telemetry DataFrame.

This module turns the raw schema written by :mod:`lfs_telemetry.telemetry.replay`
into a richer DataFrame with derived signals useful for driver coaching:

* yaw rate vs theoretical (under/oversteer index),
* sideslip angle (β),
* longitudinal & lateral load transfer (real vs theoretical),
* friction-circle utilization per wheel,
* combined tyre work (energy proxy) per wheel,
* steering-feedback (FFB) load %,
* dash-light bitmasks decoded into boolean columns
  (TC active, ABS active, pit limiter, redline shift, oil/battery warn),
* control smoothness metrics (steer reversal rate, throttle/brake rate),
* per-axle real brake bias.

All inputs are read from the canonical CSV schema (see
``lfs_telemetry.telemetry.replay._FIELDS``); columns missing in older captures
are silently skipped. The function always returns a *new* DataFrame.

Usage::

    from lfs_telemetry.telemetry import read_csv_replay, observe_window, car_spec_for
    from lfs_telemetry.telemetry.derived import enrich_dataframe
    import pandas as pd

    df = pd.DataFrame([s.outsim.__dict__ | s.outgauge.__dict__ ...])
    df_rich = enrich_dataframe(df, car_spec_for("FOX"))
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from .constants import GRAVITY as GRAVITY_MS2  # re-export for back-compat
from .observables import CarSpec, car_spec_for
from .protocol.packets import (
    DL_ABS,
    DL_BATTERY,
    DL_FULLBEAM,
    DL_HANDBRAKE,
    DL_OILWARN,
    DL_PITSPEED,
    DL_SHIFT,
    DL_SIGNAL_L,
    DL_SIGNAL_R,
    DL_TC,
    WHEEL_ORDER,
)
from .track.loader import cached_track_geometry

_DASH_LIGHT_COLS: tuple[tuple[int, str], ...] = (
    (DL_SHIFT,     "dl_shift_light"),
    (DL_HANDBRAKE, "dl_handbrake"),
    (DL_PITSPEED,  "dl_pit_limiter"),
    (DL_TC,        "dl_tc_active"),
    (DL_OILWARN,   "dl_oil_warn"),
    (DL_BATTERY,   "dl_battery_warn"),
    (DL_ABS,       "dl_abs_active"),
    (DL_FULLBEAM,  "dl_fullbeam"),
    (DL_SIGNAL_L,  "dl_signal_l"),
    (DL_SIGNAL_R,  "dl_signal_r"),
)


def enrich_dataframe(
    df: pd.DataFrame,
    spec: CarSpec | None = None,
    *,
    smoothness_window_s: float = 1.0,
    ffb_max_torque_nm: float = 25.0,
) -> pd.DataFrame:
    """Return ``df`` with all derived columns appended.

    Parameters
    ----------
    df:
        Telemetry DataFrame in the canonical schema.
    spec:
        Car spec; if ``None`` it is resolved from the ``car`` column.
    smoothness_window_s:
        Rolling window for steering reversal / input rate metrics.
    ffb_max_torque_nm:
        Reference torque for normalizing ``steer_torque_nm`` into a 0..1
        FFB-load metric. 25 Nm is a typical Formula peak.
    """
    # Shallow BlockManager copy: we only ADD columns below, never mutate
    # existing ones, so we don't need a full deep copy of the raw blocks.
    out = df.copy(deep=False)
    if spec is None:
        car = _first_str(df, "car") or ""
        spec = car_spec_for(car)

    # The helpers below append ~80 derived columns one at a time. Pandas
    # emits ``PerformanceWarning: DataFrame is highly fragmented`` for
    # each insertion past its block-manager threshold. We swallow those
    # warnings here and defragment the result with a single ``copy()``
    # at the end, which is the remediation pandas itself recommends.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        _add_chassis_dynamics(out, spec)
        _add_load_transfer(out, spec)
        _add_friction_circle(out, spec)
        _add_tyre_work(out)
        _add_brake_bias(out)
        _add_damper_velocities(out)
        _add_dash_lights(out)
        _add_ffb(out, ffb_max_torque_nm)
        _add_smoothness(out, smoothness_window_s)
        _add_gear_lfs(out)
        _add_combined_channels(out, spec, smoothness_window_s)
        _add_track_geometry(out)
    return out.copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_str(df: pd.DataFrame, col: str) -> str | None:
    if col not in df.columns or df.empty:
        return None
    s = df[col].dropna()
    if s.empty:
        return None
    return str(s.iloc[0])


def _has(df: pd.DataFrame, *cols: str) -> bool:
    return all(c in df.columns for c in cols)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den.where(den.abs() > 1e-9, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _dt_seconds(df: pd.DataFrame) -> pd.Series:
    if "time_ms" not in df.columns:
        return pd.Series(np.full(len(df), 0.01), index=df.index)
    dt = df["time_ms"].diff().fillna(0).clip(lower=0) / 1000.0
    dt = dt.where(dt > 0, 0.01)
    return dt


# ---------------------------------------------------------------------------
# Chassis dynamics: yaw, beta, oversteer/understeer index
# ---------------------------------------------------------------------------


def _add_chassis_dynamics(df: pd.DataFrame, spec: CarSpec) -> None:
    if not _has(df, "ang_vel_z", "speed_ms"):
        return
    yaw = df["ang_vel_z"]
    speed = df["speed_ms"]
    df["yaw_rate_rads"] = yaw

    # Theoretical neutral-steer yaw rate from steered angle and wheelbase.
    if _has(df, "input_steer"):
        # input_steer is (typically) in radians at the wheel for OutSimPack2.
        # For pure-pursuit at low slip: ω_th = v · tan(δ) / L.
        delta = df["input_steer"].astype(float)
        df["yaw_rate_theoretical_rads"] = speed * np.tan(delta) / spec.wheelbase_m
        df["understeer_index"] = (
            df["yaw_rate_theoretical_rads"] - yaw
        )  # >0 → understeer; <0 → oversteer
    if _has(df, "vel_x", "vel_y"):
        # Sideslip angle β = atan2(v_y, v_x) in car frame.
        df["beta_rad"] = np.arctan2(df["vel_y"], df["vel_x"])
        df["beta_deg"] = np.degrees(df["beta_rad"])


# ---------------------------------------------------------------------------
# Load transfer: real vs theoretical
# ---------------------------------------------------------------------------


def _add_load_transfer(df: pd.DataFrame, spec: CarSpec) -> None:
    if not _has(df, "accel_x", "accel_y"):
        return
    g = spec.g
    L = spec.wheelbase_m
    h = spec.cg_height_m
    m = spec.mass_kg
    # Theoretical longitudinal transfer per axle: ΔF = m·a_x·h/L.
    df["transfer_long_n_theoretical"] = m * df["accel_x"] * h / L
    # Theoretical lateral transfer per axle uses average track.
    track_avg = 0.5 * (spec.track_front_m + spec.track_rear_m)
    df["transfer_lat_n_theoretical"] = m * df["accel_y"] * h / track_avg

    cols = [f"wheel_{c}_vertical_load_n" for c in WHEEL_ORDER]
    if not _has(df, *cols):
        return
    fl = df["wheel_FL_vertical_load_n"]
    fr = df["wheel_FR_vertical_load_n"]
    rl = df["wheel_RL_vertical_load_n"]
    rr = df["wheel_RR_vertical_load_n"]
    front = fl + fr
    rear = rl + rr
    total = front + rear
    df["load_total_n"] = total
    df["load_front_frac"] = _safe_div(front, total)
    df["load_left_frac"] = _safe_div(fl + rl, total)
    df["load_diag_fl_rr_frac"] = _safe_div(fl + rr, total)
    static_front_n = m * g * spec.weight_dist_front
    df["transfer_long_n_real"] = front - static_front_n
    df["transfer_lat_n_real"] = (fr + rr) - (fl + rl)


# ---------------------------------------------------------------------------
# Friction circle utilization
# ---------------------------------------------------------------------------


def _add_friction_circle(df: pd.DataFrame, spec: CarSpec) -> None:
    speed = df["speed_ms"] if "speed_ms" in df.columns else None
    for c in WHEEL_ORDER:
        fz_col = f"wheel_{c}_vertical_load_n"
        # Note: LFS stores wheel forces in the wheel's local frame with axes
        # swapped relative to the car frame: y_force == longitudinal,
        # x_force == lateral. We expose them with engineering-friendly names.
        f_long_col = f"wheel_{c}_y_force_n"
        f_lat_col = f"wheel_{c}_x_force_n"
        if not _has(df, fz_col, f_long_col, f_lat_col):
            continue
        fz = df[fz_col].clip(lower=1.0)
        # Treat <50 N (effectively airborne) as missing to avoid spikes.
        airborne = df[fz_col] < 50.0
        # μ at this speed (single value), use scalar fallback.
        if speed is not None and spec.mu_lat_aero_k != 0.0:
            mu_l = pd.Series(
                np.asarray(spec.mu_lat_at(speed.to_numpy())), index=df.index)
        else:
            mu_l = pd.Series(spec.mu_lat, index=df.index)
        mu_x = spec.mu_long
        fx_norm = df[f_long_col] / (mu_x * fz)
        fy_norm = df[f_lat_col] / (mu_l * fz)
        use = np.sqrt(fx_norm**2 + fy_norm**2)
        df[f"friction_use_{c}"] = use.where(~airborne, np.nan)


# ---------------------------------------------------------------------------
# Tyre work / energy proxy per wheel (W → integrated J via dt elsewhere)
# ---------------------------------------------------------------------------


def _add_tyre_work(df: pd.DataFrame) -> None:
    if "speed_ms" not in df.columns:
        return
    v = df["speed_ms"]
    for c in WHEEL_ORDER:
        sa_col = f"wheel_{c}_tan_slip_angle"
        sr_col = f"wheel_{c}_slip_ratio"
        # See _add_friction_circle: LFS swaps long/lat names per wheel.
        f_long_col = f"wheel_{c}_y_force_n"
        f_lat_col = f"wheel_{c}_x_force_n"
        if not _has(df, sa_col, sr_col, f_long_col, f_lat_col):
            continue
        # Slip velocities (approx): v_slip_lat ≈ v · tan(slip_angle);
        # v_slip_long ≈ v · slip_ratio (in driven/braked wheel approximation).
        v_slip_lat = v * df[sa_col]
        v_slip_long = v * df[sr_col]
        # Power dissipated [W]: |F · v_slip|.
        df[f"tyre_work_w_{c}"] = (
            df[f_lat_col].abs() * v_slip_lat.abs()
            + df[f_long_col].abs() * v_slip_long.abs()
        )


# ---------------------------------------------------------------------------
# Real brake bias
# ---------------------------------------------------------------------------


def _add_brake_bias(df: pd.DataFrame) -> None:
    cols = [f"wheel_{c}_y_force_n" for c in WHEEL_ORDER]
    if not _has(df, *cols, "brake"):
        return
    front_brake = -(df["wheel_FL_y_force_n"] + df["wheel_FR_y_force_n"])
    rear_brake = -(df["wheel_RL_y_force_n"] + df["wheel_RR_y_force_n"])
    total = (front_brake + rear_brake).where(df["brake"] > 0.05, np.nan)
    df["brake_bias_front_real"] = _safe_div(front_brake, total)


# ---------------------------------------------------------------------------
# Damper (suspension) velocities
# ---------------------------------------------------------------------------


def _add_damper_velocities(df: pd.DataFrame) -> None:
    """Add ``wheel_<c>_susp_speed_mps`` per wheel.

    Computed as the time derivative of ``susp_deflect_m`` with a sign
    convention matching race-engineer practice:

    * positive  → bump (compression, deflection increasing)
    * negative  → rebound (extension, deflection decreasing)

    The first sample is set to 0 m/s (no previous frame). Sample-to-
    sample noise is left untouched here; downstream histograms can bin
    it as-is, exactly like MoTeC's damper-velocity histogram.
    """
    dt = _dt_seconds(df)
    for c in WHEEL_ORDER:
        col = f"wheel_{c}_susp_deflect_m"
        if col not in df.columns:
            continue
        d = df[col].astype(float).diff().fillna(0.0)
        df[f"wheel_{c}_susp_speed_mps"] = d / dt


# ---------------------------------------------------------------------------
# Dash light decode
# ---------------------------------------------------------------------------


def _add_dash_lights(df: pd.DataFrame) -> None:
    if "dash_lights" not in df.columns:
        return
    bits = df["dash_lights"].fillna(0).astype("int64")
    for mask, name in _DASH_LIGHT_COLS:
        df[name] = (bits & mask).astype(bool)


# ---------------------------------------------------------------------------
# Force-feedback load
# ---------------------------------------------------------------------------


def _add_ffb(df: pd.DataFrame, ffb_max_torque_nm: float) -> None:
    if "steer_torque_nm" not in df.columns:
        return
    df["ffb_load_pct"] = (
        df["steer_torque_nm"].abs() / ffb_max_torque_nm
    ).clip(upper=1.5)


# ---------------------------------------------------------------------------
# Control smoothness
# ---------------------------------------------------------------------------


def _add_smoothness(df: pd.DataFrame, window_s: float) -> None:
    dt = _dt_seconds(df)
    if "input_steer" in df.columns:
        steer = df["input_steer"].astype(float)
        d_steer = steer.diff().fillna(0)
        df["steer_rate_rads"] = d_steer / dt
        sign = np.sign(steer.where(steer.abs() > 0.01, 0))
        reversals = (sign.diff().abs() > 0).astype(int)
        win = max(int(window_s / max(dt.median(), 1e-3)), 1)
        df["steer_reversal_rate_hz"] = reversals.rolling(
            win, min_periods=1).sum() / window_s
    for col in ("throttle", "brake"):
        if col in df.columns:
            df[f"{col}_rate_per_s"] = df[col].diff().fillna(0) / dt
    # trail braking: throttle and brake both > 0.05 at the same time.
    if _has(df, "throttle", "brake"):
        df["overlap_brake_throttle"] = (
            (df["throttle"] > 0.05) & (df["brake"] > 0.05)
        )


# ---------------------------------------------------------------------------
# Canonical LFS gear
# ---------------------------------------------------------------------------


def _add_gear_lfs(df: pd.DataFrame) -> None:
    """Translate raw OutGauge gear byte to canonical LFS numbering.

    OutGauge: 0 = Reverse, 1 = Neutral, 2 = 1st, 3 = 2nd, ...
    Canonical (LFS dashboard): -1 = R, 0 = N, 1 = 1st, 2 = 2nd, ...
    """
    if "gear" not in df.columns:
        return
    raw = pd.to_numeric(df["gear"], errors="coerce")
    df["gear_lfs"] = (raw - 1).astype("Int64")
    # Vectorized label: avoids the per-row Python lambda that was the
    # single hottest line in enrich_dataframe on long stints.
    raw_arr = raw.to_numpy(dtype=float)
    out = np.empty(raw_arr.shape, dtype=object)
    nan_mask = np.isnan(raw_arr)
    out[nan_mask] = ""
    valid = ~nan_mask
    iv = raw_arr[valid].astype(int)
    labels = np.where(
        iv == 0, "R",
        np.where(iv == 1, "N", (iv - 1).astype(str)),
    )
    out[valid] = labels
    df["gear_label"] = pd.array(out, dtype="string")


# ---------------------------------------------------------------------------
# Combined / synergy channels — leverage multiple raw inputs to expose
# information useful for both drivers and engineers that is not directly
# in any single LFS packet.
# ---------------------------------------------------------------------------


def _add_combined_channels(
    df: pd.DataFrame, spec: CarSpec, smoothness_window_s: float,
) -> None:
    """Append cross-channel synergy metrics.

    Channels added (when their inputs are present):

    * ``g_total_g`` — friction-circle magnitude
      ``sqrt(accel_x² + accel_y²) / g``. The classic g–g headline: how
      close the car is to the total grip envelope at any instant.
    * ``susp_compression_front_avg_m`` / ``..._rear_avg_m`` and
      ``rake_compression_m`` (front − rear) — axle-level ride-height
      proxies and dynamic rake. Aero-sensitive cars live or die by these.
    * ``slip_angle_balance_rad`` — ``mean(|α_front|) − mean(|α_rear|)``.
      Direct kinematic balance: positive → front slides (understeer),
      negative → rear slides (oversteer). Complements
      ``understeer_index`` (yaw-rate based).
    * ``wheel_<c>_lockup`` — boolean: braking with deeply negative slip
      ratio at that wheel (``brake > 0.1`` and ``slip_ratio < −0.3``).
      Per-wheel ABS-event detector independent of any tyre radius.
    * ``brake_power_w`` — instantaneous mechanical brake power
      ``|F_long_total| · speed`` while braking. Integrates per lap to a
      reliable brake-heat / brake-pad-wear proxy.
    * ``throttle_reversal_rate_hz`` — mirror of ``steer_reversal_rate``
      for the throttle pedal: smoothness / pedal-tap detector.
    * ``coasting`` — ``throttle < 0.05 ∧ brake < 0.05 ∧ speed > 3 m/s``.
      Identifies mid-corner coasting (lift-off oversteer technique).
    * ``trail_brake_intensity`` — ``brake · |input_steer|``. Non-zero
      only on combined-load corner entry; rises with both pedal and
      hand engagement.
    * ``chassis_roll_per_lat_g_rad_per_g`` — instantaneous roll
      compliance ``roll / (accel_y/g)`` (NaN at low lat. g). Direct
      probe for ARB / spring choice.
    * ``chassis_pitch_per_long_g_rad_per_g`` — analogous
      ``pitch / (accel_x/g)`` (NaN at low long. g): brake-dive /
      power-squat compliance.
    """
    g = spec.g

    # --- g_total: friction-circle magnitude --------------------------
    if _has(df, "accel_x", "accel_y"):
        ax = df["accel_x"].astype(float)
        ay = df["accel_y"].astype(float)
        df["g_total_g"] = np.hypot(ax, ay) / g

    # --- Axle-level susp. compression & rake -------------------------
    susp_cols = [f"wheel_{c}_susp_deflect_m" for c in WHEEL_ORDER]
    if _has(df, *susp_cols):
        fl = df["wheel_FL_susp_deflect_m"].astype(float)
        fr = df["wheel_FR_susp_deflect_m"].astype(float)
        rl = df["wheel_RL_susp_deflect_m"].astype(float)
        rr = df["wheel_RR_susp_deflect_m"].astype(float)
        front_avg = 0.5 * (fl + fr)
        rear_avg = 0.5 * (rl + rr)
        df["susp_compression_front_avg_m"] = front_avg
        df["susp_compression_rear_avg_m"] = rear_avg
        # Positive ``rake_compression_m`` ⇒ front compresses more than
        # rear ⇒ nose-down attitude (typical under braking / aero load).
        df["rake_compression_m"] = front_avg - rear_avg

    # --- Kinematic slip-angle balance --------------------------------
    sa_cols = [f"wheel_{c}_tan_slip_angle" for c in WHEEL_ORDER]
    if _has(df, *sa_cols):
        af = 0.5 * (
            df["wheel_FL_tan_slip_angle"].abs()
            + df["wheel_FR_tan_slip_angle"].abs()
        )
        ar = 0.5 * (
            df["wheel_RL_tan_slip_angle"].abs()
            + df["wheel_RR_tan_slip_angle"].abs()
        )
        # tan(α) ≈ α for the small angles typical here, so reusing the
        # tan column as a near-angle is acceptable for a balance metric.
        df["slip_angle_balance_rad"] = af - ar

    # --- Per-wheel lockup (slip-ratio based, no tyre radius needed) --
    if "brake" in df.columns:
        braking = df["brake"].astype(float) > 0.1
        for c in WHEEL_ORDER:
            sr_col = f"wheel_{c}_slip_ratio"
            if sr_col not in df.columns:
                continue
            sr = df[sr_col].astype(float)
            df[f"wheel_{c}_lockup"] = (braking & (sr < -0.3)).astype(bool)

    # --- Brake power dissipation -------------------------------------
    long_cols = [f"wheel_{c}_y_force_n" for c in WHEEL_ORDER]
    if _has(df, *long_cols, "speed_ms", "brake"):
        # Total longitudinal tyre force; negative under braking. Power
        # bled into brakes ≈ |F_brake_total| · v while pedal is down.
        f_total_long = (
            df["wheel_FL_y_force_n"]
            + df["wheel_FR_y_force_n"]
            + df["wheel_RL_y_force_n"]
            + df["wheel_RR_y_force_n"]
        ).astype(float)
        braking_mask = df["brake"].astype(float) > 0.05
        power = (-f_total_long).clip(lower=0.0) * df["speed_ms"].astype(float)
        df["brake_power_w"] = power.where(braking_mask, 0.0)

    # --- Throttle smoothness (mirror of steer reversal rate) ---------
    if "throttle" in df.columns:
        thr = df["throttle"].astype(float)
        # Direction change in pedal motion (Δ ↔ sign of throttle rate),
        # debounced by a small deadband to ignore quantisation flutter.
        d_thr = thr.diff().fillna(0.0)
        sign = np.sign(d_thr.where(d_thr.abs() > 0.005, 0.0))
        reversals = (sign.diff().abs() > 0).astype(int)
        dt = _dt_seconds(df)
        win = max(int(smoothness_window_s / max(dt.median(), 1e-3)), 1)
        df["throttle_reversal_rate_hz"] = (
            reversals.rolling(win, min_periods=1).sum() / smoothness_window_s
        )

    # --- Coasting flag (mid-corner technique) ------------------------
    if _has(df, "throttle", "brake", "speed_ms"):
        df["coasting"] = (
            (df["throttle"].astype(float) < 0.05)
            & (df["brake"].astype(float) < 0.05)
            & (df["speed_ms"].astype(float) > 3.0)
        ).astype(bool)

    # --- Trail-brake intensity ---------------------------------------
    if _has(df, "brake", "input_steer"):
        df["trail_brake_intensity"] = (
            df["brake"].astype(float) * df["input_steer"].abs().astype(float)
        )

    # --- Chassis compliance ratios -----------------------------------
    # Only meaningful when the axis is actually loaded; below the
    # threshold the ratio is dominated by sensor noise and the divisor
    # crosses zero, so we mask with NaN instead of reporting garbage.
    if _has(df, "roll", "accel_y"):
        ay = df["accel_y"].astype(float)
        roll = df["roll"].astype(float)
        loaded = ay.abs() > 2.0  # ~0.2 g
        df["chassis_roll_per_lat_g_rad_per_g"] = (
            (roll / (ay / g)).where(loaded, np.nan)
        )
    if _has(df, "pitch", "accel_x"):
        ax = df["accel_x"].astype(float)
        pitch = df["pitch"].astype(float)
        loaded = ax.abs() > 2.0
        df["chassis_pitch_per_long_g_rad_per_g"] = (
            (pitch / (ax / g)).where(loaded, np.nan)
        )


# ---------------------------------------------------------------------------
# Track geometry: slope correction, yaw misalignment, segment columns
# ---------------------------------------------------------------------------


def _resolve_track_code(df: pd.DataFrame) -> str | None:
    """Return the most common non-empty value of ``ctx_track``."""
    if "ctx_track" not in df.columns or df.empty:
        return None
    s = df["ctx_track"].dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return None
    return s.mode().iloc[0].upper()


def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angles in radians to (-π, π]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _add_track_geometry(df: pd.DataFrame) -> None:
    """Append track-geometry columns when a racing-line CSV is available.

    For every (``pos_x``, ``pos_y``) sample we project to the nearest
    centerline node of ``racing_lines/<TRACK>_racing.csv`` and add:

    * ``track_node``, ``track_s_m``, ``track_z_m``,
      ``track_heading_rad``, ``track_curvature_1_per_m``,
      ``track_radius_m``, ``track_slope_pct``, ``track_width_m``,
      ``segment_id``, ``segment_kind``, ``track_offset_m``;

    * ``accel_x_road_mps2`` — body-frame longitudinal acceleration with
      the road-grade gravity component removed:
      ``accel_x + g · sin(arctan(slope_pct/100))``. Positive ⇒ engine,
      negative ⇒ braking, regardless of road grade;

    * ``velocity_heading_rad`` and ``yaw_misalign_rad`` — angular delta
      between the car's velocity vector and the centerline tangent; a
      proxy for trajectory slip when no IMU slip-angle channel exists.

    Silently no-ops when ``ctx_track`` is missing/empty, or when no
    racing_lines CSV exists for the detected track.
    """
    if not _has(df, "pos_x", "pos_y"):
        return
    track = _resolve_track_code(df)
    if not track:
        return
    geom = cached_track_geometry(track)
    if geom is None:
        return
    pos = df[["pos_x", "pos_y"]].to_numpy(dtype=float)
    finite = np.isfinite(pos).all(axis=1)
    nonzero = (np.abs(pos).sum(axis=1) > 1e-6)
    if not (finite & nonzero).any():
        return  # no usable positions

    cols = geom.lookup(pos[:, 0], pos[:, 1])
    for name, arr in cols.items():
        # Mark unusable rows (sentinel positions) with NaN / "" / -1 so
        # downstream consumers don't draw bogus values for grid placeholders.
        if name == "segment_id":
            arr = arr.astype(np.int64).copy()
            arr[~(finite & nonzero)] = -1
        elif name == "segment_kind":
            arr = arr.astype(object).copy()
            arr[~(finite & nonzero)] = ""
        else:
            arr = arr.astype(float).copy()
            arr[~(finite & nonzero)] = np.nan
        df[name] = arr

    # Slope-corrected longitudinal acceleration.
    if "accel_x" in df.columns:
        slope_rad = np.arctan(df["track_slope_pct"].astype(float) / 100.0)
        df["accel_x_road_mps2"] = (
            df["accel_x"].astype(float) + GRAVITY_MS2 * np.sin(slope_rad)
        )

    # Trajectory yaw misalignment.
    if _has(df, "vel_x", "vel_y"):
        vx = df["vel_x"].astype(float)
        vy = df["vel_y"].astype(float)
        speed = np.hypot(vx, vy)
        vh = np.arctan2(vy, vx)
        # Define heading only when actually moving (>1 m/s), else NaN.
        vh = np.where(speed > 1.0, vh, np.nan)
        df["velocity_heading_rad"] = vh
        head = df["track_heading_rad"].astype(float).to_numpy()
        delta = _wrap_to_pi(vh - head)
        df["yaw_misalign_rad"] = delta


__all__ = ["enrich_dataframe"]
