"""Synthetic stint generator v2 (physics-lite).

Reads ``racing_lines/<TRACK>_racing.csv`` (s, curvature, v_target, slope, width)
and produces N laps of LFS Schema 1.1 telemetry honoring:

    * the friction circle per wheel (sqrt(Fx² + Fy²) / (μ·Fz) <= ~1.0),
    * real load transfer (longitudinal + lateral),
    * one-DoF suspension dynamics per corner (spring+damper, white-noise road),
    * a coherent steering / yaw / understeer signal,
    * a simple per-corner thermal model.

Usage:

    python tools/synth_stint_v2.py BL1 FBM --laps 5 --out assets/

Output: ``assets/synthetic_<TRACK>_<CAR>_v2_lap0N.csv`` for N=1..laps.

This is a *physics-lite simulator*: the goal is to produce telemetry whose
derived quantities (friction_use_<c>, understeer_index, susp_speed_mps PSD)
behave like real data so downstream tests have a credible fixture. It is
NOT a vehicle-dynamics engine — slip curves are linearised, the engine is
a torque-by-gear table, and the driver is a v-target PID.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lfs_telemetry.telemetry.constants import GRAVITY  # noqa: E402
from lfs_telemetry.telemetry.lap import LapTelemetry  # noqa: E402
from lfs_telemetry.telemetry.observables import CarSpec, car_spec_for  # noqa: E402

WHEEL_ORDER = ("RL", "RR", "FL", "FR")
DT = 0.01  # 100 Hz fixed-step
GRAV = GRAVITY

# Suspension natural frequency / damping per corner (typical race-tuned setup).
SUSP_FN_HZ = 2.5
SUSP_ZETA = 0.35
ROAD_NOISE_STD = 2.5    # vertical input m/s² white-noise RMS (road texture)
KERB_PROB = 0.005       # per-step probability of a curb hit (small spike)
KERB_AMP = 8.0          # curb spike m/s² (occasional, e.g. apex/exit kerb)

# Driver model
V_FRAC_TARGET = 0.96    # fraction of racing-line v_target the driver achieves
A_LONG_FRAC = 0.92      # fraction of available μ·g used longitudinally
BRAKE_BIAS_FRONT = 0.58 # static bias

# Tyre kinematic constants
PEAK_SLIP_RATIO = 0.10
PEAK_SLIP_ANGLE_RAD = 0.12

# Thermal model — carcass + slow core (heat-soak across the stint).
# Maps to LFS setup: tyre compound (R1/R2/R3/R4) sets the operating window,
# pressure shifts the slip-power → ΔT gain. Channel: wheel_<c>_air_temp_c.
T_AMB_C = 22.0
T_CARCASS_INITIAL_C = 50.0   # cold tyres from grid/pit-out
THERMAL_TAU_S = 25.0         # fast carcass time constant (s)
THERMAL_GAIN_C_PER_KW = 12.0 # tuned so T_eq under load reaches ~90–100 °C
T_CORE_TAU_S = 180.0         # slow core (≈3 min) → drives stint heat-soak

# Understeer / oversteer model
K_U_STATIC = 0.08            # base proportional understeer coefficient
K_U_DYNAMIC = 0.0035         # transient: + on entry (dκ/dt rising), − on exit
POWER_ON_REAR_LAT_LOSS = 0.18  # RWD: rear lat grip reduction on full power
POWER_ON_THRESHOLD = 0.4     # throttle above which loss starts


@dataclass
class TrackProfile:
    s: np.ndarray            # arc length (m)
    x: np.ndarray            # racing-line x (m)
    y: np.ndarray            # racing-line y (m)
    heading: np.ndarray      # rad
    curvature: np.ndarray    # 1/m, signed (+ left turn)
    slope_pct: np.ndarray
    width: np.ndarray
    v_target: np.ndarray     # m/s

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def sample(self, s_query: np.ndarray) -> dict:
        # Wrap s_query into [0, length).
        s_mod = np.mod(s_query, self.length)
        out = {}
        for name in ("x", "y", "heading", "curvature", "slope_pct", "width", "v_target"):
            arr = getattr(self, name)
            out[name] = np.interp(s_mod, self.s, arr)
        return out


def load_track(track_code: str) -> TrackProfile:
    path = REPO_ROOT / "racing_lines" / f"{track_code}_racing.csv"
    df = pd.read_csv(path)
    return TrackProfile(
        s=df["s_m"].to_numpy(float),
        x=df["x_line_m"].to_numpy(float),
        y=df["y_line_m"].to_numpy(float),
        heading=df["heading_rad"].to_numpy(float),
        curvature=df["curvature_1_per_m"].to_numpy(float),
        slope_pct=df["slope_pct"].to_numpy(float),
        width=df["width_m"].to_numpy(float),
        v_target=df["v_target_ms"].to_numpy(float),
    )


# ---------------------------------------------------------------------------
# Per-wheel load and force allocation (quasi-static)
# ---------------------------------------------------------------------------

def wheel_vertical_loads(spec: CarSpec, a_x: float, a_y: float, fuel_kg: float) -> dict[str, float]:
    """Return Fz per wheel (N) including longitudinal+lateral transfer."""
    m = spec.mass_kg + fuel_kg
    g = GRAV
    L = spec.wheelbase_m
    h = spec.cg_height_m
    tf = spec.track_front_m
    tr = spec.track_rear_m
    wf = spec.weight_dist_front
    wr = 1.0 - wf
    fz_front_static = m * g * wf
    fz_rear_static = m * g * wr
    # Longitudinal transfer (positive a_x = accel → rear gains)
    dF_long = m * a_x * h / L
    fz_front = fz_front_static - dF_long
    fz_rear = fz_rear_static + dF_long
    # Lateral transfer per axle (positive a_y = left turn → right side gains).
    dF_lat_front = m * a_y * h * wf / tf
    dF_lat_rear = m * a_y * h * wr / tr
    fl = 0.5 * fz_front - dF_lat_front
    fr = 0.5 * fz_front + dF_lat_front
    rl = 0.5 * fz_rear - dF_lat_rear
    rr = 0.5 * fz_rear + dF_lat_rear
    return {"FL": max(fl, 50.0), "FR": max(fr, 50.0),
            "RL": max(rl, 50.0), "RR": max(rr, 50.0)}


def wheel_forces(
    spec: CarSpec,
    fz: dict[str, float],
    a_x: float,
    a_y: float,
    fuel_kg: float,
    speed_ms: float,
    bias_front: float = BRAKE_BIAS_FRONT,
    throttle_frac: float = 0.0,
) -> dict[str, tuple[float, float]]:
    """Return (F_long, F_lat) per wheel in N. Honors friction circle.

    Convention matches ``derived._add_friction_circle``:
      ``y_force_n`` = longitudinal (drive/brake)
      ``x_force_n`` = lateral
    """
    m = spec.mass_kg + fuel_kg
    total_long = m * a_x
    total_lat = m * a_y
    mu_lat = float(spec.mu_lat_at(speed_ms))
    mu_long = spec.mu_long

    # Lateral force distribution: per axle by static weight; per-side weighted
    # by Fz^1.5 (proxy for tyre load sensitivity — outside loaded tyre carries
    # disproportionately MORE lateral than inside, so per-wheel friction_use
    # spreads across the four wheels instead of collapsing to one value).
    f_lat_front_total = total_lat * spec.weight_dist_front
    f_lat_rear_total = total_lat * (1.0 - spec.weight_dist_front)
    w_fl = fz["FL"] ** 1.5
    w_fr = fz["FR"] ** 1.5
    w_rl = fz["RL"] ** 1.5
    w_rr = fz["RR"] ** 1.5
    front_w = w_fl + w_fr
    rear_w = w_rl + w_rr
    f_lat = {
        "FL": f_lat_front_total * w_fl / front_w,
        "FR": f_lat_front_total * w_fr / front_w,
        "RL": f_lat_rear_total * w_rl / rear_w,
        "RR": f_lat_rear_total * w_rr / rear_w,
    }
    # Power-on oversteer signature: on a driven axle under throttle, the
    # tyre's lateral capacity drops (longitudinal slip eats the friction
    # circle). For RWD this transfers some yaw demand to the front.
    if throttle_frac > POWER_ON_THRESHOLD and a_x > 0:
        loss = POWER_ON_REAR_LAT_LOSS * (throttle_frac - POWER_ON_THRESHOLD) / (1.0 - POWER_ON_THRESHOLD)
        if spec.driven == "RWD":
            loss_rear = loss * (f_lat["RL"] + f_lat["RR"])
            f_lat["RL"] *= (1.0 - loss)
            f_lat["RR"] *= (1.0 - loss)
            # Reassign to front (vehicle still needs to corner).
            if abs(f_lat_front_total) > 1e-3:
                scale = 1.0 + loss_rear / f_lat_front_total
                f_lat["FL"] *= scale
                f_lat["FR"] *= scale
        elif spec.driven == "FWD":
            f_lat["FL"] *= (1.0 - loss * 0.5)  # FWD: milder, plus understeer signature
            f_lat["FR"] *= (1.0 - loss * 0.5)
    # Longitudinal: RWD → rear only on accel, brake-bias on braking.
    f_long = dict.fromkeys(WHEEL_ORDER, 0.0)
    if total_long >= 0:
        # Acceleration on driven axle (rear for FBM)
        if spec.driven == "RWD":
            f_long["RL"] = 0.5 * total_long
            f_long["RR"] = 0.5 * total_long
        elif spec.driven == "FWD":
            f_long["FL"] = 0.5 * total_long
            f_long["FR"] = 0.5 * total_long
        else:  # AWD 50/50
            for c in WHEEL_ORDER:
                f_long[c] = 0.25 * total_long
    else:
        bf = bias_front
        f_long["FL"] = 0.5 * total_long * bf
        f_long["FR"] = 0.5 * total_long * bf
        f_long["RL"] = 0.5 * total_long * (1.0 - bf)
        f_long["RR"] = 0.5 * total_long * (1.0 - bf)

    # Friction-circle enforcement per wheel: scale BOTH components by the same
    # factor if combined demand exceeds 0.98·μ·Fz, so the derived friction_use
    # peaks just under 1.0 (not at 1.368).
    out: dict[str, tuple[float, float]] = {}
    for c in WHEEL_ORDER:
        fx = f_long[c]
        fy = f_lat[c]
        denom_x = mu_long * fz[c]
        denom_y = mu_lat * fz[c]
        norm = np.hypot(fx / denom_x, fy / denom_y)
        if norm > 0.98:
            k = 0.98 / norm
            fx *= k
            fy *= k
        out[c] = (fx, fy)
    return out


# ---------------------------------------------------------------------------
# One-DoF suspension dynamics per corner (semi-implicit Euler)
# ---------------------------------------------------------------------------

@dataclass
class SuspensionState:
    z: float = 0.0      # deflection (m), 0 at static
    zdot: float = 0.0   # speed (m/s)

    def step(self, force_input_n: float, dt: float, omega_n: float, zeta: float, mass_unsprung_kg: float) -> tuple[float, float]:
        # m·zdd + 2ζω·zdot + ω²·z = F/m_unsprung (normalized input).
        accel = force_input_n / mass_unsprung_kg - 2 * zeta * omega_n * self.zdot - omega_n ** 2 * self.z
        self.zdot += accel * dt
        self.z += self.zdot * dt
        return self.z, self.zdot


# ---------------------------------------------------------------------------
# Lap generator
# ---------------------------------------------------------------------------

def _gear_for_speed(v: float) -> tuple[int, float]:
    """Return (gear_index, rpm) for the FBM (5-speed)."""
    # Approx gear bands (m/s) and rpm at that band for engine
    bands = [(0, 8, 1), (8, 18, 2), (18, 30, 3), (30, 45, 4), (45, 80, 5)]
    for vmin, vmax, gear in bands:
        if vmin <= v < vmax:
            # rpm = idle + (v - vmin)/(vmax - vmin) * (max_rpm - idle)
            rpm = 1500 + (v - vmin) / max(vmax - vmin, 1e-3) * 6000
            return gear, rpm
    return 5, 7500


def generate_lap(
    track: TrackProfile,
    spec: CarSpec,
    car_name: str,
    track_name: str,
    lap_index: int,
    t0_ms: int,
    fuel_kg_start: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate one lap as a Schema 1.1 DataFrame."""
    # Per-lap variations
    speed_scale = rng.uniform(0.997, 1.003)
    steer_noise_std = rng.uniform(0.002, 0.004)

    # Pre-estimate lap length & duration to pre-allocate.
    L = track.length
    # Time-march with adaptive arc-length progress.
    rows: list[dict] = []
    s = 0.0
    v = float(track.v_target[0] * V_FRAC_TARGET * speed_scale)
    t_ms = int(t0_ms)
    fuel_kg = fuel_kg_start

    # Suspension state per corner.
    sus: dict[str, SuspensionState] = {c: SuspensionState() for c in WHEEL_ORDER}
    M_UNSPRUNG = 25.0
    omega_n = 2 * np.pi * SUSP_FN_HZ

    # Thermal state per corner: fast carcass + slow core (heat-soak across stint).
    T_carcass = {c: T_CARCASS_INITIAL_C + rng.uniform(-2, 2) for c in WHEEL_ORDER}
    T_core = {c: T_carcass[c] for c in WHEEL_ORDER}

    # Previous values for derivatives.
    float(track.heading[0])
    kappa_prev = float(track.curvature[0]) * -1.0

    step = 0
    max_steps = 30000  # safety: 5 min cap at 100 Hz
    while s < L and step < max_steps:
        sample = track.sample(np.array([s]))
        kappa = float(sample["curvature"][0]) * -1.0  # racing-line: positive curvature for right? we negate to align convention
        # NOTE: the sign of curvature does not affect friction_use magnitudes.
        v_tgt = float(sample["v_target"][0]) * V_FRAC_TARGET * speed_scale
        heading = float(sample["heading"][0])

        # Lateral acceleration from current speed and curvature.
        a_y = v * v * kappa

        # Available longitudinal grip given current lateral demand.
        mu_lat = float(spec.mu_lat_at(v))
        mu_long = spec.mu_long
        # Reserved fraction for lateral: a_y/(μ_lat·g) used; remaining for a_x.
        lat_frac = min(abs(a_y) / (mu_lat * GRAV), 1.0)
        long_budget = mu_long * GRAV * np.sqrt(max(0.0, 1.0 - lat_frac ** 2)) * A_LONG_FRAC

        # Driver: PD on v_target, clipped by long_budget.
        err = v_tgt - v
        a_x_demand = 0.8 * err / max(DT, 0.01)  # P controller; converges in ~0.1 s
        a_x = float(np.clip(a_x_demand, -long_budget, long_budget))

        # Throttle / brake input mapping (0..1).
        if a_x >= 0:
            throttle = min(a_x / long_budget if long_budget > 1e-3 else 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(-a_x / long_budget if long_budget > 1e-3 else 0.0, 1.0)

        # Update speed.
        v_new = max(1.0, v + a_x * DT)

        # Update arc length.
        ds = 0.5 * (v + v_new) * DT
        s_new = s + ds

        # Loads and forces.
        fz = wheel_vertical_loads(spec, a_x, a_y, fuel_kg)
        forces = wheel_forces(spec, fz, a_x, a_y, fuel_kg, v, throttle_frac=throttle)

        # Suspension excitation: per-corner Fz change relative to static + road noise.
        statics = spec.static_corner_loads_n()  # FL/FR/RL/RR
        susp_outputs: dict[str, tuple[float, float]] = {}
        for c in WHEEL_ORDER:
            dynamic = fz[c] - statics[c]
            road = rng.normal(0.0, ROAD_NOISE_STD) * M_UNSPRUNG
            if rng.random() < KERB_PROB:
                road += KERB_AMP * M_UNSPRUNG * rng.choice([-1, 1])
            # Treat (dynamic + road) as net force on the unsprung mass.
            # Scale dynamic by 0.02 (small static-bias coupling, not full Fz)
            # and let road noise dominate high-frequency content.
            z, zdot = sus[c].step(0.02 * dynamic + road, DT, omega_n, SUSP_ZETA, M_UNSPRUNG)
            susp_outputs[c] = (z, zdot)

        # Kinematic slip estimates: linear in demand fraction.
        slip: dict[str, tuple[float, float]] = {}
        for c in WHEEL_ORDER:
            fx, fy = forces[c]
            sr = PEAK_SLIP_RATIO * (fx / max(mu_long * fz[c], 1e-3))
            sa = PEAK_SLIP_ANGLE_RAD * (fy / max(mu_lat * fz[c], 1e-3))
            slip[c] = (sr, np.tan(sa))  # store tan(slip_angle) for the schema

        # Thermal update: power dissipated ≈ |F_lat·v_slip_lat + F_long·v_slip_long|
        # Approximate v_slip_lat ≈ v · slip_angle; v_slip_long ≈ v · slip_ratio.
        # Two-time-constant model: fast carcass tracks instantaneous heat,
        # slow core integrates carcass — produces the stint-long heat-soak
        # observable in real LFS `wheel_<c>_air_temp_c` traces (compound +
        # pressure setup choices determine where T_core stabilises).
        for c in WHEEL_ORDER:
            fx, fy = forces[c]
            sr, tan_sa = slip[c]
            p_w = abs(fx * v * sr) + abs(fy * v * np.arctan(tan_sa))
            T_eq = T_AMB_C + THERMAL_GAIN_C_PER_KW * (p_w / 1000.0)
            # Carcass never drops below the slow core (real tyre behaviour).
            target = max(T_eq, T_core[c])
            alpha_fast = 1.0 - np.exp(-DT / THERMAL_TAU_S)
            T_carcass[c] += alpha_fast * (target - T_carcass[c])
            alpha_slow = 1.0 - np.exp(-DT / T_CORE_TAU_S)
            T_core[c] += alpha_slow * (T_carcass[c] - T_core[c])

        # Yaw rate, steering, accelerations.
        yaw_rate = v * kappa
        # Steering angle: Ackermann + steady-state understeer (proportional to
        # a_y/μ·g) + a transient term proportional to dκ/dt that captures the
        # entry-vs-exit asymmetry a real driver exhibits: more steer-in on
        # corner entry (κ rising) and unwind earlier on exit (κ falling).
        # The effect is observable in `input_steer` vs `ang_vel_z`, i.e. in
        # the derived `understeer_index` channel.
        ackermann = np.arctan(spec.wheelbase_m * kappa)
        understeer = K_U_STATIC * a_y / (mu_lat * GRAV)
        dkappa_dt = (kappa - kappa_prev) / DT
        steer_transient = K_U_DYNAMIC * dkappa_dt
        steer = ackermann + understeer + steer_transient + rng.normal(0.0, steer_noise_std)

        # Body-frame velocities (assume small slip → vel_x ≈ v, vel_y small).
        beta = 0.5 * understeer  # small sideslip proxy
        vel_x = v * np.cos(beta)
        vel_y = v * np.sin(beta)
        accel_x = a_x
        accel_y = a_y

        # Engine + gear.
        gear, rpm = _gear_for_speed(v)
        # Fuel burn ~ 0.001 kg/s under part-throttle, more on full throttle.
        fuel_kg = max(0.0, fuel_kg - DT * (0.0008 + 0.0015 * throttle))

        # Position from racing line sample.
        pos_x = float(sample["x"][0])
        pos_y = float(sample["y"][0])

        # Assemble row.
        row = {
            "time_ms": t_ms,
            "ang_vel_x": rng.normal(0.0, 0.05),
            "ang_vel_y": rng.normal(0.0, 0.05),
            "ang_vel_z": yaw_rate,
            "heading": heading,
            "pitch": 0.0,
            "roll": 0.0,
            "accel_x": accel_x,
            "accel_y": accel_y,
            "accel_z": -GRAV,
            "vel_x": vel_x,
            "vel_y": vel_y,
            "vel_z": 0.0,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "pos_z": 0.0,
            "car": car_name,
            "gear": gear,
            "speed_ms": v,
            "rpm": rpm,
            "throttle": throttle,
            "brake": brake,
            "clutch": 0.0,
            "fuel": fuel_kg,
            "eng_temp_c": 92.0,
            "oil_temp_c": 95.0,
            "oil_pressure_bar": 4.0,
            "turbo_bar": 0.6 * throttle,
            "og_flags": 0,
            "dash_lights": 0,
            "show_lights": 0,
            "og_player_id": 0,
            "current_lap_dist_m": s,
            "indexed_distance_m": s,
            "steer_torque_nm": 8.0 * abs(steer),
            "engine_ang_vel_rads": rpm * 2 * np.pi / 60.0,
            "max_torque_at_vel_nm": 180.0,
            "input_throttle": throttle,
            "input_brake": brake,
            "input_steer": steer,
            "input_clutch": 0.0,
            "input_handbrake": 0.0,
        }
        for c in WHEEL_ORDER:
            z, zdot = susp_outputs[c]
            fx, fy = forces[c]
            sr, tan_sa = slip[c]
            # Wheel angular velocity ≈ v_long / r_wheel; r ≈ 0.32 m for FBM
            row[f"wheel_{c}_susp_deflect_m"] = z
            row[f"wheel_{c}_vertical_load_n"] = fz[c]
            row[f"wheel_{c}_slip_ratio"] = sr
            row[f"wheel_{c}_tan_slip_angle"] = tan_sa
            row[f"wheel_{c}_x_force_n"] = fy  # lateral
            row[f"wheel_{c}_y_force_n"] = fx  # longitudinal
            row[f"wheel_{c}_ang_vel_rads"] = v / 0.32 * (1.0 + sr)
            row[f"wheel_{c}_lean_rel_road_rad"] = 0.0
            row[f"wheel_{c}_air_temp_c"] = T_carcass[c]
            row[f"wheel_{c}_slip_fraction"] = min(1.0, np.hypot(sr / PEAK_SLIP_RATIO,
                                                                 tan_sa / PEAK_SLIP_ANGLE_RAD))
            row[f"wheel_{c}_touching"] = 1
            # Per-wheel steer: front wheels follow steer input, rear=0.
            row[f"wheel_{c}_steer_rad"] = steer if c in ("FL", "FR") else 0.0
        # Context fields.
        row.update({
            "ctx_track": track_name,
            "ctx_weather": 1,
            "ctx_wind": 0,
            "ctx_view_plid": 1,
            "ctx_view_car": car_name,
            "ctx_race_in_progress": 1,
            "ctx_race_laps": 0,
            "ctx_qual_minutes": 0,
            "ctx_lfs_version": "0.7E",
            "ctx_view_lap_count": lap_index,
            "ctx_view_last_lap_ms": 0,
            "ctx_view_last_split1_ms": 0,
            "ctx_view_last_split2_ms": 0,
            "ctx_view_last_split3_ms": 0,
            "ctx_view_last_hlv_code": 0,
            "ctx_view_last_hlv_name": "",
            "ctx_view_last_hlv_speed_ms": 0.0,
            "ctx_view_handicap_mass_kg": 0,
            "ctx_view_handicap_t_res": 0,
            "ctx_view_tyre_compounds": "R3R3R3R3",
            "ctx_obh_count": 0,
            "ctx_pit_stop_count": 0,
        })
        rows.append(row)

        # Advance.
        v = v_new
        s = s_new
        kappa_prev = kappa
        t_ms += round(DT * 1000)
        step += 1

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# CSV writing (Schema 1.1 with header comment)
# ---------------------------------------------------------------------------

def write_lap_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("# lfs-telemetry telemetry schema=1.1\n")
        df.to_csv(f, index=False, lineterminator="\n")


# ---------------------------------------------------------------------------
# Self-validation
# ---------------------------------------------------------------------------

def validate_lap(path: Path, car_name: str) -> dict:
    """Run the critical checks that the v1 synthetic failed."""
    lap = LapTelemetry.from_csv(path, car=car_name)
    df = lap.enriched
    spec = lap.car
    report: dict = {"path": str(path)}

    # 1. friction_use per wheel: per-TIMESTEP cross-wheel spread should be
    #    non-trivial in cornering (outside wheel works MORE than inside).
    #    Per-wheel p95 alone is misleading on tracks with both LH and RH
    #    corners (each wheel hits the same ceiling at SOME point in the lap).
    fr_cols = [f"friction_use_{c}" for c in WHEEL_ORDER]
    if all(c in df.columns for c in fr_cols):
        fr = df[fr_cols].to_numpy()
        # Cross-wheel std at each timestep (axis=1), masked to high cornering.
        if "accel_y" in df.columns:
            mask = df["accel_y"].abs() > 0.5 * spec.g
            spreads_per_t = np.nanstd(fr[mask.to_numpy()], axis=1)
        else:
            spreads_per_t = np.nanstd(fr, axis=1)
        peaks = {c: float(df[c].quantile(0.95)) for c in fr_cols}
        report["friction_peaks_p95"] = peaks
        report["friction_cross_wheel_std_p90"] = float(np.nanpercentile(spreads_per_t, 90)) if len(spreads_per_t) else float("nan")
        report["friction_max_peak"] = float(max(peaks.values()))
        ok = (
            report["friction_max_peak"] <= 1.05
            and report["friction_cross_wheel_std_p90"] > 0.08
        )
        report["friction_ok"] = bool(ok)
    else:
        report["friction_ok"] = False

    # 2. understeer_index: notna fraction in cornering > 0.5
    if "understeer_index" in df.columns and "accel_y" in df.columns:
        ay = df["accel_y"]
        mask = ay.abs() > 0.5 * spec.g
        n = int(mask.sum())
        n_finite = int(df.loc[mask, "understeer_index"].notna().sum())
        frac = n_finite / max(n, 1)
        report["understeer_finite_in_corners_frac"] = frac
        report["understeer_ok"] = bool(n > 100 and frac > 0.8)
    else:
        report["understeer_ok"] = False

    # 3. yaw_rate_theoretical_rads: notna fraction > 0.8 overall
    if "yaw_rate_theoretical_rads" in df.columns:
        y = df["yaw_rate_theoretical_rads"]
        report["yaw_theo_finite_frac"] = float(y.notna().mean())
        report["yaw_theo_ok"] = bool(y.notna().mean() > 0.8)
    else:
        report["yaw_theo_ok"] = False

    # 4. Suspension: RMS speed per corner > 0.03 m/s and the signal is non-flat.
    susp_ok = True
    rms_by_corner = {}
    for c in WHEEL_ORDER:
        col = f"wheel_{c}_susp_speed_mps"
        if col not in df.columns:
            susp_ok = False
            continue
        v = df[col].dropna().to_numpy()
        if len(v) < 200:
            susp_ok = False
            continue
        rms = float(np.sqrt(np.mean(v ** 2)))
        rms_by_corner[c] = rms
        if rms < 0.03:
            susp_ok = False
    report["susp_rms_by_corner"] = rms_by_corner
    report["susp_ok"] = bool(susp_ok)

    # 5. Lap time consistency placeholder (just record lap duration).
    if "time_ms" in df.columns and len(df):
        report["lap_time_ms"] = int(df["time_ms"].iloc[-1] - df["time_ms"].iloc[0])

    return report


def print_report(report: dict) -> None:
    print(f"--- VALIDATION: {Path(report['path']).name} ---")
    for key, val in report.items():
        if key == "path":
            continue
        if isinstance(val, dict):
            print(f"  {key}:")
            for k, v in val.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
        elif isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("track", help="Track code (e.g. BL1, FE1, KY3R)")
    parser.add_argument("car", help="Car short name (e.g. FBM, XRR)")
    parser.add_argument("--laps", type=int, default=5, help="Number of laps to generate")
    parser.add_argument("--out", type=str, default="assets",
                        help="Output directory (default: assets/)")
    parser.add_argument("--fuel-kg", type=float, default=20.0,
                        help="Starting fuel (default: 20 kg)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--no-validate", action="store_true", help="Skip self-validation")
    args = parser.parse_args(argv)

    track = load_track(args.track)
    spec = car_spec_for(args.car)
    print(f"Loaded track {args.track}: length={track.length:.1f} m, "
          f"v_target range [{track.v_target.min():.1f}, {track.v_target.max():.1f}] m/s")
    print(f"Car {args.car}: mass={spec.mass_kg} kg, wheelbase={spec.wheelbase_m} m, "
          f"mu_lat={spec.mu_lat}, mu_long={spec.mu_long}, driven={spec.driven}")

    out_dir = Path(args.out)
    rng = np.random.default_rng(args.seed)
    fuel_kg = args.fuel_kg
    t_offset_ms = 0
    written: list[Path] = []
    for lap_i in range(1, args.laps + 1):
        df = generate_lap(
            track, spec, args.car, args.track, lap_i, t_offset_ms, fuel_kg, rng
        )
        path = out_dir / f"synthetic_{args.track}_{args.car}_v2_lap{lap_i:02d}.csv"
        write_lap_csv(df, path)
        lap_ms = int(df["time_ms"].iloc[-1] - df["time_ms"].iloc[0])
        v_avg = float(df["speed_ms"].mean())
        print(f"  lap {lap_i}: {len(df)} samples, "
              f"time={lap_ms/1000:.2f}s, v_avg={v_avg*3.6:.1f} km/h "
              f"-> {path.name}")
        written.append(path)
        t_offset_ms += lap_ms + 100  # small inter-lap gap
        fuel_kg = float(df["fuel"].iloc[-1])

    if not args.no_validate:
        print("\n=== SELF-VALIDATION ===")
        all_ok = True
        for p in written:
            rep = validate_lap(p, args.car)
            print_report(rep)
            for k in ("friction_ok", "understeer_ok", "yaw_theo_ok", "susp_ok"):
                if not rep.get(k, False):
                    all_ok = False
        print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
        return 0 if all_ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
