"""ObservableMapper: LFS enriched telemetry → TNFR :class:`NodeSeed`.

For every subsystem (wheel, axle, brake, engine, chassis, driver, corner)
this module produces a :class:`NodeSeed` with:

* ``epi``   — normalized energy/integrity in [0, 1] (RMS of weighted
  components divided by a per-car reference vector — see
  :meth:`ObservableMapper.seed_wheel`).
* ``vf``    — dominant frequency (Hz) of the subsystem's characteristic
  signal, computed via Welch PSD on a variable-rate-aware grid (1.0 s
  window, 50 % overlap, band [0.5, 25] Hz).
* ``theta`` — phase (rad, in (-π, π]) of the dominant FFT component of
  the characteristic signal. Used downstream by the TNFR engine to
  detect coherent vs. dissonant couplings.
* ``meta``  — dict of raw physical aggregates the rationale renderer uses.

The implementation deliberately does NOT depend on :mod:`tnfr`. The
canonical engine receives ``(epi, vf, theta)`` triples via
:mod:`tnfr_racing.network_*` in Phase 3.

LFS / telemetry mapping (every quantity here exists in schema 1.1):

* ``wheel_<c>_susp_speed_mps``       → suspension damper signature
* ``friction_use_<c>``               → tyre friction-circle usage [0,1]
* ``wheel_<c>_vertical_load_n``      → load
* ``wheel_<c>_air_temp_c``           → thermal proxy (R-spec, ~70–150 °C)
* ``accel_x``, ``accel_y``           → chassis demand
* ``ang_vel_z``                      → yaw rate (chassis)
* ``input_steer``, ``input_throttle``, ``input_brake`` → driver
* ``rpm``                            → engine
* ``brake_bias_front_real``          → brake balance (NaN when brake≤0.05)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from lfs_telemetry.telemetry.observables import CarSpec

WHEEL_ORDER = ("FL", "FR", "RL", "RR")
FRONT_WHEELS = ("FL", "FR")
REAR_WHEELS = ("RL", "RR")

# Welch / PSD constants
_VF_BAND_HZ = (0.5, 25.0)
_WELCH_WINDOW_S = 1.0
_WELCH_OVERLAP = 0.5


@dataclass(frozen=True)
class NodeSeed:
    """Seed for a TNFR node, derived from a slice of enriched telemetry.

    All four scalar fields are finite numbers; ``epi`` is clipped to
    [0, 1], ``vf`` to the band edges, ``theta`` to (-π, π]. Construction
    failures (empty/constant signal, fewer than ``2 × welch_window``
    samples) return ``epi=0``, ``vf=band_lo``, ``theta=0`` with
    ``meta["error"]`` set so downstream consumers can skip the node.
    """

    name: str
    epi: float
    vf: float
    theta: float
    meta: Mapping[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Spectral helpers (variable-rate aware)
# ---------------------------------------------------------------------------

def _sampling_rate_hz(time_ms: pd.Series) -> float:
    """Estimate fs from inter-sample dt; robust to occasional gaps."""
    dt_s = np.diff(time_ms.to_numpy(dtype=float)) / 1000.0
    dt_s = dt_s[(dt_s > 0) & (dt_s < 1.0)]
    if dt_s.size == 0:
        return 100.0
    return float(1.0 / np.median(dt_s))


def _welch_dominant(
    signal: np.ndarray, fs: float, band: tuple[float, float] = _VF_BAND_HZ
) -> tuple[float, float]:
    """Return (peak_freq_hz, peak_phase_rad) of the dominant band component.

    Uses Welch PSD for the magnitude and a windowed FFT on the same data
    for the phase. Falls back to ``(band[0], 0.0)`` on degenerate input.
    """
    from scipy.signal import welch

    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 16:
        return float(band[0]), 0.0
    x = x - x.mean()
    if not np.any(x):
        return float(band[0]), 0.0
    nperseg = max(16, int(round(fs * _WELCH_WINDOW_S)))
    nperseg = min(nperseg, x.size)
    noverlap = int(nperseg * _WELCH_OVERLAP)
    try:
        f, pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    except ValueError:
        return float(band[0]), 0.0
    mask = (f >= band[0]) & (f <= band[1])
    if not mask.any():
        return float(band[0]), 0.0
    idx = int(np.argmax(pxx[mask]))
    f_peak = float(f[mask][idx])
    # Phase via single-segment FFT (cheap, deterministic).
    n_fft = min(x.size, max(nperseg, 64))
    fft = np.fft.rfft(x[:n_fft])
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    k = int(np.argmin(np.abs(fft_freqs - f_peak)))
    phase = float(np.angle(fft[k]))
    return f_peak, phase


def _norm_rms(values: np.ndarray, refs: np.ndarray) -> float:
    """Compute EPI = ||values / refs||_2 / sqrt(len(refs)), clipped to [0,1].

    Matches the §5.2 formula with all weights w_k=1: a perfectly
    saturated subsystem (every component at its reference) yields EPI=1.
    """
    v = np.asarray(values, dtype=float)
    r = np.asarray(refs, dtype=float)
    if r.size == 0 or not np.all(np.isfinite(r)) or np.any(r <= 0):
        return 0.0
    ratios = v / r
    finite = ratios[np.isfinite(ratios)]
    if finite.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(finite))))
    return float(np.clip(rms, 0.0, 1.0))


def _column_or_nan(df: pd.DataFrame, col: str) -> np.ndarray:
    if col in df.columns:
        return df[col].to_numpy(dtype=float)
    return np.full(len(df), np.nan)


# ---------------------------------------------------------------------------
# ObservableMapper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CarRefs:
    """Per-car reference vectors (constants, not lap-dependent)."""

    fz_static: dict[str, float]              # N per wheel
    a_lat_ref_mps2: float                    # μ_lat · g
    a_long_ref_mps2: float                   # μ_long · g
    susp_speed_ref_mps: float = 0.20         # damper RMS reference
    friction_ref: float = 1.0                # full friction circle
    temp_ref_c: float = 110.0                # tyre working-window upper bound
    yaw_rate_ref_rads: float = 2.0           # ~115°/s aggressive cornering
    steer_ref_rad: float = 0.5               # ~28°, full-lock proxy
    rpm_ref: float = 7500.0


class ObservableMapper:
    """Stateless mapper from enriched telemetry → :class:`NodeSeed`.

    The instance only caches per-car reference values; every ``seed_*``
    method is a pure function of the supplied DataFrame slice.
    """

    def __init__(self, car: CarSpec) -> None:
        self.car = car
        statics = car.static_corner_loads_n()
        self._refs = _CarRefs(
            fz_static=statics,
            a_lat_ref_mps2=car.mu_lat * car.g,
            a_long_ref_mps2=car.mu_long * car.g,
        )

    # ------------------------------------------------------------------
    # Per-wheel seed (4 nodes per stint)
    # ------------------------------------------------------------------
    def seed_wheel(self, wheel: str, df: pd.DataFrame) -> NodeSeed:
        assert wheel in WHEEL_ORDER, f"unknown wheel {wheel}"
        susp = _column_or_nan(df, f"wheel_{wheel}_susp_speed_mps")
        fric = _column_or_nan(df, f"friction_use_{wheel}")
        fz = _column_or_nan(df, f"wheel_{wheel}_vertical_load_n")
        temp = _column_or_nan(df, f"wheel_{wheel}_air_temp_c")
        fs = _sampling_rate_hz(df["time_ms"])

        values = np.array([
            float(np.sqrt(np.nanmean(susp ** 2))) if np.isfinite(susp).any() else 0.0,
            float(np.nanmean(np.abs(fric))) if np.isfinite(fric).any() else 0.0,
            float(np.nanmean(fz) / self._refs.fz_static[wheel])
            if np.isfinite(fz).any() else 0.0,
            float(np.nanmean(temp) / self._refs.temp_ref_c)
            if np.isfinite(temp).any() else 0.0,
        ])
        refs = np.array([
            self._refs.susp_speed_ref_mps,
            self._refs.friction_ref,
            1.0,
            1.0,
        ])
        epi = _norm_rms(values, refs)
        vf, theta = _welch_dominant(susp, fs)

        meta = {
            "susp_rms_mps": values[0],
            "friction_mean": values[1],
            "load_mean_n": values[2] * self._refs.fz_static[wheel],
            "temp_mean_c": values[3] * self._refs.temp_ref_c,
            "fs_hz": fs,
        }
        return NodeSeed(f"wheel.{wheel}", epi, vf, theta, meta)

    # ------------------------------------------------------------------
    # Axle seed (front/rear): aggregates the two wheels of the axle
    # ------------------------------------------------------------------
    def seed_axle(self, axle: str, df: pd.DataFrame) -> NodeSeed:
        assert axle in ("front", "rear"), f"unknown axle {axle}"
        wheels = FRONT_WHEELS if axle == "front" else REAR_WHEELS
        seeds = [self.seed_wheel(w, df) for w in wheels]
        epi = float(np.mean([s.epi for s in seeds]))
        # Axle vf: slip-fraction differential left-right (proxy for axle
        # yaw contribution / one-wheel-spin signature).
        s_l = _column_or_nan(df, f"wheel_{wheels[0]}_slip_fraction")
        s_r = _column_or_nan(df, f"wheel_{wheels[1]}_slip_fraction")
        diff = s_l - s_r
        fs = _sampling_rate_hz(df["time_ms"])
        vf, theta = _welch_dominant(diff, fs)
        meta = {
            "epi_left": seeds[0].epi,
            "epi_right": seeds[1].epi,
            "slip_diff_rms": float(np.sqrt(np.nanmean(diff ** 2)))
            if np.isfinite(diff).any() else 0.0,
            "fs_hz": fs,
        }
        return NodeSeed(f"axle.{axle}", epi, vf, theta, meta)

    # ------------------------------------------------------------------
    # Brake seed (front/rear): bias + force concentration
    # ------------------------------------------------------------------
    def seed_brake(self, axle: str, df: pd.DataFrame) -> NodeSeed:
        assert axle in ("front", "rear"), f"unknown axle {axle}"
        wheels = FRONT_WHEELS if axle == "front" else REAR_WHEELS
        # Per derived.py convention: y_force_n is LONGITUDINAL.
        fl = _column_or_nan(df, f"wheel_{wheels[0]}_y_force_n")
        fr = _column_or_nan(df, f"wheel_{wheels[1]}_y_force_n")
        brake = _column_or_nan(df, "input_brake")
        bias = _column_or_nan(df, "brake_bias_front_real")
        fs = _sampling_rate_hz(df["time_ms"])

        mask = brake > 0.05
        if mask.any():
            f_axle_brake = (fl + fr)[mask]
            brake_rms = float(np.sqrt(np.nanmean(f_axle_brake ** 2)))
        else:
            brake_rms = 0.0
        f_axle_max = self.car.mass_kg * self.car.g * self.car.mu_long
        values = np.array([brake_rms / f_axle_max])
        refs = np.array([1.0])
        epi = _norm_rms(values, refs)

        vf, theta = _welch_dominant(brake, fs)
        meta = {
            "brake_force_rms_n": brake_rms,
            "brake_bias_front_real_mean": float(np.nanmean(bias))
            if np.isfinite(bias).any() else float("nan"),
            "fs_hz": fs,
        }
        return NodeSeed(f"brake.{axle}", epi, vf, theta, meta)

    # ------------------------------------------------------------------
    # Engine seed: power / rpm signature
    # ------------------------------------------------------------------
    def seed_engine(self, df: pd.DataFrame) -> NodeSeed:
        rpm = _column_or_nan(df, "rpm")
        thr = _column_or_nan(df, "input_throttle")
        ax = _column_or_nan(df, "accel_x")
        fs = _sampling_rate_hz(df["time_ms"])
        values = np.array([
            float(np.nanmean(rpm) / self._refs.rpm_ref) if np.isfinite(rpm).any() else 0.0,
            float(np.nanmean(thr)) if np.isfinite(thr).any() else 0.0,
            float(np.nanmean(np.maximum(ax, 0.0)) / self._refs.a_long_ref_mps2)
            if np.isfinite(ax).any() else 0.0,
        ])
        refs = np.array([1.0, 1.0, 1.0])
        epi = _norm_rms(values, refs)
        vf, theta = _welch_dominant(rpm, fs)
        meta = {
            "rpm_mean": values[0] * self._refs.rpm_ref,
            "throttle_mean": values[1],
            "accel_x_pos_mean_mps2": values[2] * self._refs.a_long_ref_mps2,
            "fs_hz": fs,
        }
        return NodeSeed("engine", epi, vf, theta, meta)

    # ------------------------------------------------------------------
    # Chassis seed: lateral/longitudinal/yaw demand
    # ------------------------------------------------------------------
    def seed_chassis(self, df: pd.DataFrame) -> NodeSeed:
        ax = _column_or_nan(df, "accel_x")
        ay = _column_or_nan(df, "accel_y")
        yaw = _column_or_nan(df, "ang_vel_z")
        fs = _sampling_rate_hz(df["time_ms"])
        values = np.array([
            float(np.sqrt(np.nanmean(ax ** 2))) / self._refs.a_long_ref_mps2,
            float(np.sqrt(np.nanmean(ay ** 2))) / self._refs.a_lat_ref_mps2,
            float(np.sqrt(np.nanmean(yaw ** 2))) / self._refs.yaw_rate_ref_rads,
        ])
        refs = np.array([1.0, 1.0, 1.0])
        epi = _norm_rms(values, refs)
        vf, theta = _welch_dominant(yaw, fs)
        meta = {
            "a_long_rms_mps2": values[0] * self._refs.a_long_ref_mps2,
            "a_lat_rms_mps2": values[1] * self._refs.a_lat_ref_mps2,
            "yaw_rate_rms_rads": values[2] * self._refs.yaw_rate_ref_rads,
            "fs_hz": fs,
        }
        return NodeSeed("chassis", epi, vf, theta, meta)

    # ------------------------------------------------------------------
    # Driver seed: input activity + steering smoothness
    # ------------------------------------------------------------------
    def seed_driver(self, df: pd.DataFrame) -> NodeSeed:
        steer = _column_or_nan(df, "input_steer")
        thr = _column_or_nan(df, "input_throttle")
        brk = _column_or_nan(df, "input_brake")
        fs = _sampling_rate_hz(df["time_ms"])
        values = np.array([
            float(np.sqrt(np.nanmean(steer ** 2))) / self._refs.steer_ref_rad,
            float(np.nanmean(thr)) if np.isfinite(thr).any() else 0.0,
            float(np.nanmean(brk)) if np.isfinite(brk).any() else 0.0,
        ])
        refs = np.array([1.0, 1.0, 1.0])
        epi = _norm_rms(values, refs)
        vf, theta = _welch_dominant(steer, fs)
        meta = {
            "steer_rms_rad": values[0] * self._refs.steer_ref_rad,
            "throttle_mean": values[1],
            "brake_mean": values[2],
            "fs_hz": fs,
        }
        return NodeSeed("driver", epi, vf, theta, meta)

    # ------------------------------------------------------------------
    # Corner seed: per-sector × phase (entry/apex/exit)
    # ------------------------------------------------------------------
    def seed_corner(
        self,
        sector_id: int,
        phase: str,
        df: pd.DataFrame,
        sector_start_m: float,
        sector_end_m: float,
    ) -> NodeSeed:
        """Slice ``df`` by ``current_lap_dist_m`` ∈ [sector_start, sector_end],
        sub-slice by phase, and seed the node.

        Phase split (by distance fraction within the sector):
        * entry — first 30 %
        * apex  — middle 30 %
        * exit  — last 40 %
        """
        assert phase in ("entry", "apex", "exit"), f"unknown phase {phase}"
        if "current_lap_dist_m" not in df.columns:
            return NodeSeed(
                f"corner.{sector_id}.{phase}", 0.0, _VF_BAND_HZ[0], 0.0,
                {"error": "no_distance"},
            )
        length = sector_end_m - sector_start_m
        if length <= 0:
            return NodeSeed(
                f"corner.{sector_id}.{phase}", 0.0, _VF_BAND_HZ[0], 0.0,
                {"error": "empty_sector"},
            )
        if phase == "entry":
            lo, hi = sector_start_m, sector_start_m + 0.30 * length
        elif phase == "apex":
            lo, hi = sector_start_m + 0.30 * length, sector_start_m + 0.60 * length
        else:
            lo, hi = sector_start_m + 0.60 * length, sector_end_m

        d = df["current_lap_dist_m"]
        sub = df.loc[(d >= lo) & (d <= hi)]
        if len(sub) < 8:
            return NodeSeed(
                f"corner.{sector_id}.{phase}", 0.0, _VF_BAND_HZ[0], 0.0,
                {"error": "too_few_samples", "n": int(len(sub))},
            )
        chassis = self.seed_chassis(sub)
        # EPI inherits chassis but adds phase-specific physical demand:
        # entry = brake/long, apex = lateral, exit = traction.
        ay_rms = chassis.meta["a_lat_rms_mps2"]
        ax_rms = chassis.meta["a_long_rms_mps2"]
        if phase == "entry":
            phys = ax_rms / self._refs.a_long_ref_mps2
        elif phase == "apex":
            phys = ay_rms / self._refs.a_lat_ref_mps2
        else:
            phys = max(ax_rms, ay_rms * 0.5) / self._refs.a_long_ref_mps2
        epi = float(np.clip(0.5 * (chassis.epi + phys), 0.0, 1.0))
        meta = {
            "phase": phase,
            "sector_id": float(sector_id),
            "dist_lo_m": lo,
            "dist_hi_m": hi,
            "n_samples": float(len(sub)),
            "a_lat_rms_mps2": ay_rms,
            "a_long_rms_mps2": ax_rms,
            "yaw_rms_rads": chassis.meta["yaw_rate_rms_rads"],
        }
        return NodeSeed(
            f"corner.{sector_id}.{phase}", epi, chassis.vf, chassis.theta, meta,
        )
