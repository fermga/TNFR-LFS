"""Suspension calculator and frequency-domain analyzer.

Bridges static setup (from ``CAR_info.bin``) with dynamic suspension
behaviour observed in a lap. Outputs are the classic vehicle-dynamics
metrics a race engineer reaches for first:

**Static (per corner / per axle)**

* sprung mass per corner,
* wheel rate, tyre rate, ride rate (spring & tyre in series),
* natural / ride frequency in Hz,
* damping ratio ``ζ`` separately for bump and rebound,
* roll stiffness Nm/rad (springs + ARB, per axle and total),
* front roll-stiffness fraction (a key understeer/oversteer dial),
* "flat ride" ratio ``f_rear / f_front`` (Olley / Maurice Olley's
  rule of thumb: ~1.05–1.20 so the rear settles a beat after the
  front and the chassis doesn't pitch on bumps).

**Dynamic (per corner)**

* dominant frequency of ``wheel_<c>_susp_speed_mps`` via Welch PSD,
* energy distribution across coarse frequency bands (<1, 1–3, 3–8,
  8–20 Hz) — the 3–8 Hz band is where the tyre/unsprung mass lives,
* damper-speed percentiles (P10 rebound side, P50, P90 bump side).

**Cross checks**

* ``compare`` flags when a corner's observed dominant frequency sits
  near its computed natural frequency (resonance risk) or when the
  bump/rebound speed histogram suggests under/over-damping.

LFS conventions (per RAF spec):

* ``CarInfoWheel.spring_const``, ``damping_comp``, ``damping_rebound``
  and ``anti_roll`` are all expressed **at the wheel**, so motion
  ratio is already baked in — no separate conversion needed.
* Wheels are ordered ``(RL, RR, FL, FR)`` in the bin tuple.

This module is pure — no I/O, no UI — so it can be unit-tested with
synthetic ``CarInfoBin`` instances and synthetic DataFrames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import welch

from .car_info_bin import CarInfoBin, CarInfoWheel
from .protocol.packets import WHEEL_ORDER  # ("FL", "FR", "RL", "RR")

# Map LFS bin wheel index → canonical name.
_BIN_INDEX_TO_NAME = {0: "RL", 1: "RR", 2: "FL", 3: "FR"}
_FRONT = ("FL", "FR")
_REAR = ("RL", "RR")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CornerSuspension:
    """Static suspension metrics for a single corner."""

    wheel: str                          # "FL"|"FR"|"RL"|"RR"
    sprung_mass_kg: float
    wheel_rate_n_per_m: float           # spring at the wheel
    tyre_rate_n_per_m: float
    ride_rate_n_per_m: float            # spring & tyre in series
    natural_freq_hz: float              # uses ride rate + sprung mass
    damping_ratio_bump: float           # ζ for compression stroke
    damping_ratio_rebound: float        # ζ for extension stroke
    arb_wheel_rate_n_per_m: float


@dataclass(frozen=True, slots=True)
class AxleSuspension:
    """Per-axle aggregates."""

    axle: str                           # "front"|"rear"
    natural_freq_hz: float              # mean of corner f_n
    roll_stiffness_springs_nm_per_rad: float
    roll_stiffness_arb_nm_per_rad: float
    roll_stiffness_total_nm_per_rad: float
    track_m: float


@dataclass(frozen=True, slots=True)
class SuspensionStatic:
    """Full static report."""

    corners: dict[str, CornerSuspension]
    front: AxleSuspension
    rear: AxleSuspension
    flat_ride_ratio: float              # f_rear / f_front
    front_roll_stiffness_fraction: float
    total_roll_stiffness_nm_per_rad: float


@dataclass(frozen=True, slots=True)
class WheelDynamics:
    """Per-wheel dynamic analysis from one DataFrame."""

    wheel: str
    dominant_freq_hz: float
    psd_peak_power: float
    energy_band_pct: dict[str, float]
    damper_speed_p10_mps: float          # rebound side (negative)
    damper_speed_p50_mps: float
    damper_speed_p90_mps: float          # bump side (positive)


@dataclass(frozen=True, slots=True)
class SuspensionDynamic:
    """Full dynamic report."""

    sample_rate_hz: float
    wheels: dict[str, WheelDynamics]


@dataclass(frozen=True, slots=True)
class AssessmentItem:
    """Single human-readable assessment entry for the UI."""

    name: str
    value: float
    units: str
    target_low: float
    target_high: float
    status: str                          # "green"|"amber"|"red"|"info"
    note: str = ""


# Typical engineering target ranges. ``car_class`` selects which band is
# used for natural-frequency assessment; the rest of the targets are
# fairly universal.
TARGETS_BY_CLASS: dict[str, tuple[float, float]] = {
    "road":     (1.0, 1.6),
    "sport":    (1.5, 2.5),
    "race":     (2.5, 4.0),    # formula / GT without significant aero
    "aero":     (4.0, 7.0),    # high downforce open-wheel / proto
}
_DAMPING_BUMP_TARGET = (0.25, 0.55)
_DAMPING_REBOUND_TARGET = (0.45, 0.75)
_FLAT_RIDE_TARGET = (1.05, 1.20)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _wheels_by_name(car: CarInfoBin) -> dict[str, CarInfoWheel]:
    return {_BIN_INDEX_TO_NAME[i]: w for i, w in enumerate(car.wheels)}


def _ride_rate(k_spring: float, k_tyre: float) -> float:
    """Spring and tyre in series. Returns 0 if either is non-positive."""
    if k_spring <= 0 or k_tyre <= 0:
        return 0.0
    return (k_spring * k_tyre) / (k_spring + k_tyre)


def _natural_freq_hz(k_n_per_m: float, m_kg: float) -> float:
    if k_n_per_m <= 0 or m_kg <= 0:
        return 0.0
    return math.sqrt(k_n_per_m / m_kg) / (2.0 * math.pi)


def _damping_ratio(c_ns_per_m: float, k_n_per_m: float, m_kg: float) -> float:
    """ζ = c / (2 · √(k·m)). Returns 0 if inputs are non-positive."""
    if c_ns_per_m <= 0 or k_n_per_m <= 0 or m_kg <= 0:
        return 0.0
    return c_ns_per_m / (2.0 * math.sqrt(k_n_per_m * m_kg))


def _axle_roll_stiffness_nm_per_rad(
    k_left: float, k_right: float, track_m: float,
) -> float:
    """Wheel-rate-based roll stiffness for one axle.

    Small-angle derivation: for a body roll ``θ`` around the longitudinal
    axis, the outer wheel compresses by ``(t/2)·θ`` and the inner wheel
    extends by the same amount (``t`` = track). The reaction moment is

        M = k_left · (tθ/2) · (t/2) + k_right · (tθ/2) · (t/2)
          = (k_left + k_right) · t² · θ / 4

    so ``K_φ = (k_left + k_right) · t² / 4``. Applies identically to the
    ARB contribution because ``anti_roll`` is published at the wheel by
    LFS (RAF spec).
    """
    if track_m <= 0:
        return 0.0
    return (k_left + k_right) * track_m * track_m / 4.0


def _status_within(value: float, lo: float, hi: float, *, amber: float = 0.15) -> str:
    """Green inside [lo, hi]; amber within +/- ``amber`` of the band; red beyond."""
    if lo <= value <= hi:
        return "green"
    width = hi - lo
    margin = width * amber if width > 0 else amber
    if (lo - margin) <= value <= (hi + margin):
        return "amber"
    return "red"


# ---------------------------------------------------------------------------
# Static calculator
# ---------------------------------------------------------------------------


def compute_static(car: CarInfoBin) -> SuspensionStatic:
    """Build the static suspension report from a parsed ``CAR_info.bin``.

    The math is deliberately conservative: it assumes symmetric left/right
    setups for axle aggregates (true 99% of the time in LFS) and uses the
    *ride rate* (spring & tyre in series) for the natural frequency, which
    is the number that matches what the driver actually feels.
    """
    wheels = _wheels_by_name(car)

    # ---- Sprung mass per corner ----------------------------------------
    # Total static load on each axle from the chassis CG split; then
    # subtract unsprung mass at that corner.
    m_total = car.mass_kg
    wd_f = car.weight_dist_front
    m_front_axle = m_total * wd_f
    m_rear_axle = m_total * (1.0 - wd_f)
    axle_total = {"FL": m_front_axle / 2.0, "FR": m_front_axle / 2.0,
                  "RL": m_rear_axle / 2.0,  "RR": m_rear_axle / 2.0}

    corners: dict[str, CornerSuspension] = {}
    for name in WHEEL_ORDER:
        w = wheels[name]
        sprung = max(axle_total[name] - w.unsprung_mass_kg, 0.0)
        k_spring = w.spring_const
        k_tyre = w.tyre_vert_spring
        k_ride = _ride_rate(k_spring, k_tyre)
        f_n = _natural_freq_hz(k_ride, sprung)
        zeta_b = _damping_ratio(w.damping_comp, k_ride, sprung)
        zeta_r = _damping_ratio(w.damping_rebound, k_ride, sprung)
        corners[name] = CornerSuspension(
            wheel=name,
            sprung_mass_kg=sprung,
            wheel_rate_n_per_m=k_spring,
            tyre_rate_n_per_m=k_tyre,
            ride_rate_n_per_m=k_ride,
            natural_freq_hz=f_n,
            damping_ratio_bump=zeta_b,
            damping_ratio_rebound=zeta_r,
            arb_wheel_rate_n_per_m=w.anti_roll,
        )

    # ---- Per-axle aggregates -------------------------------------------
    def _axle(axle_name: str, corner_names: tuple[str, str],
              track_m: float) -> AxleSuspension:
        c_l, c_r = corners[corner_names[0]], corners[corner_names[1]]
        f_n_avg = 0.5 * (c_l.natural_freq_hz + c_r.natural_freq_hz)
        k_phi_spring = _axle_roll_stiffness_nm_per_rad(
            c_l.wheel_rate_n_per_m, c_r.wheel_rate_n_per_m, track_m,
        )
        k_phi_arb = _axle_roll_stiffness_nm_per_rad(
            c_l.arb_wheel_rate_n_per_m, c_r.arb_wheel_rate_n_per_m, track_m,
        )
        return AxleSuspension(
            axle=axle_name,
            natural_freq_hz=f_n_avg,
            roll_stiffness_springs_nm_per_rad=k_phi_spring,
            roll_stiffness_arb_nm_per_rad=k_phi_arb,
            roll_stiffness_total_nm_per_rad=k_phi_spring + k_phi_arb,
            track_m=track_m,
        )

    front = _axle("front", _FRONT, car.track_front_m)
    rear = _axle("rear", _REAR, car.track_rear_m)

    flat = (rear.natural_freq_hz / front.natural_freq_hz
            if front.natural_freq_hz > 0 else 0.0)
    k_total = front.roll_stiffness_total_nm_per_rad + rear.roll_stiffness_total_nm_per_rad
    front_frac = (front.roll_stiffness_total_nm_per_rad / k_total
                  if k_total > 0 else 0.0)

    return SuspensionStatic(
        corners=corners,
        front=front, rear=rear,
        flat_ride_ratio=flat,
        front_roll_stiffness_fraction=front_frac,
        total_roll_stiffness_nm_per_rad=k_total,
    )


# ---------------------------------------------------------------------------
# Static assessment
# ---------------------------------------------------------------------------


def assess_static(
    static: SuspensionStatic, car_class: str = "race",
) -> list[AssessmentItem]:
    """Compare static metrics against canonical engineering ranges.

    ``car_class`` ∈ {road, sport, race, aero} selects the natural-
    frequency target band. Unknown classes fall back to ``race``.
    """
    fn_lo, fn_hi = TARGETS_BY_CLASS.get(car_class, TARGETS_BY_CLASS["race"])
    items: list[AssessmentItem] = []

    for axle, label in ((static.front, "front"), (static.rear, "rear")):
        items.append(AssessmentItem(
            name=f"natural_freq_{label}",
            value=axle.natural_freq_hz, units="Hz",
            target_low=fn_lo, target_high=fn_hi,
            status=_status_within(axle.natural_freq_hz, fn_lo, fn_hi),
            note=f"Average ride frequency on the {label} axle.",
        ))

    items.append(AssessmentItem(
        name="flat_ride_ratio",
        value=static.flat_ride_ratio, units="",
        target_low=_FLAT_RIDE_TARGET[0], target_high=_FLAT_RIDE_TARGET[1],
        status=_status_within(static.flat_ride_ratio, *_FLAT_RIDE_TARGET),
        note="f_rear / f_front; >1 keeps the chassis flat over bumps.",
    ))

    for name, corner in static.corners.items():
        items.append(AssessmentItem(
            name=f"damping_bump_{name}",
            value=corner.damping_ratio_bump, units="ζ",
            target_low=_DAMPING_BUMP_TARGET[0],
            target_high=_DAMPING_BUMP_TARGET[1],
            status=_status_within(
                corner.damping_ratio_bump, *_DAMPING_BUMP_TARGET),
            note="Compression damping ratio.",
        ))
        items.append(AssessmentItem(
            name=f"damping_rebound_{name}",
            value=corner.damping_ratio_rebound, units="ζ",
            target_low=_DAMPING_REBOUND_TARGET[0],
            target_high=_DAMPING_REBOUND_TARGET[1],
            status=_status_within(
                corner.damping_ratio_rebound, *_DAMPING_REBOUND_TARGET),
            note="Rebound damping ratio.",
        ))

    # Front roll-stiffness fraction is reported but not graded — the
    # right value depends on drivetrain and aero balance, not a single
    # universal range.
    items.append(AssessmentItem(
        name="front_roll_stiffness_fraction",
        value=static.front_roll_stiffness_fraction, units="frac",
        target_low=0.0, target_high=1.0, status="info",
        note=("Share of total roll stiffness on the front axle. "
              "Raise to add understeer, lower to add oversteer."),
    ))

    return items


# ---------------------------------------------------------------------------
# Dynamic (frequency-domain) analyzer
# ---------------------------------------------------------------------------


_BANDS: tuple[tuple[str, float, float], ...] = (
    ("<1Hz",   0.0,  1.0),
    ("1-3Hz",  1.0,  3.0),
    ("3-8Hz",  3.0,  8.0),
    ("8-20Hz", 8.0, 20.0),
)


def _estimate_sample_rate_hz(df: pd.DataFrame) -> float | None:
    if "time_ms" not in df.columns or len(df) < 2:
        return None
    dt_ms = np.diff(df["time_ms"].to_numpy(dtype=float))
    dt_ms = dt_ms[np.isfinite(dt_ms) & (dt_ms > 0)]
    if dt_ms.size == 0:
        return None
    return float(1000.0 / float(np.median(dt_ms)))


def _analyse_wheel(
    name: str, speed: np.ndarray, fs: float,
) -> WheelDynamics | None:
    speed = speed[np.isfinite(speed)]
    if speed.size < 32 or fs <= 0:
        return None
    # Welch PSD with a window short enough that even 1–2 s of data work.
    nper = min(256, max(16, speed.size // 4))
    freqs, psd = welch(speed - speed.mean(), fs=fs, nperseg=nper)
    if psd.size == 0:
        return None
    # Dominant frequency: pick the strongest bin above 0.5 Hz to ignore
    # the DC/quasi-static drift that always dominates a short window.
    mask = freqs >= 0.5
    if not mask.any():
        return None
    idx_local = int(np.argmax(psd[mask]))
    dom_freq = float(freqs[mask][idx_local])
    dom_pow = float(psd[mask][idx_local])

    total_power = float(psd.sum()) or 1.0
    energy = {}
    for label, lo, hi in _BANDS:
        band_mask = (freqs >= lo) & (freqs < hi)
        energy[label] = 100.0 * float(psd[band_mask].sum()) / total_power

    p10, p50, p90 = (float(x) for x in np.quantile(speed, (0.10, 0.50, 0.90)))
    return WheelDynamics(
        wheel=name,
        dominant_freq_hz=dom_freq,
        psd_peak_power=dom_pow,
        energy_band_pct=energy,
        damper_speed_p10_mps=p10,
        damper_speed_p50_mps=p50,
        damper_speed_p90_mps=p90,
    )


def compute_dynamic(df: pd.DataFrame) -> SuspensionDynamic | None:
    """Frequency-domain analysis of the per-wheel damper-speed channels.

    Requires ``enrich_dataframe`` to have already added the
    ``wheel_<c>_susp_speed_mps`` columns. Returns ``None`` if the
    DataFrame is too short or the columns are missing.
    """
    fs = _estimate_sample_rate_hz(df)
    if fs is None:
        return None
    out: dict[str, WheelDynamics] = {}
    for name in WHEEL_ORDER:
        col = f"wheel_{name}_susp_speed_mps"
        if col not in df.columns:
            continue
        wd = _analyse_wheel(name, df[col].to_numpy(dtype=float), fs)
        if wd is not None:
            out[name] = wd
    if not out:
        return None
    return SuspensionDynamic(sample_rate_hz=fs, wheels=out)


# ---------------------------------------------------------------------------
# Cross-check: static vs dynamic
# ---------------------------------------------------------------------------


def compare(
    static: SuspensionStatic, dynamic: SuspensionDynamic,
    *, resonance_tol: float = 0.15,
) -> list[AssessmentItem]:
    """Cross-check the static targets against the observed lap dynamics.

    Two flags per corner:

    * **resonance** — dominant observed frequency within
      ``resonance_tol`` (default 15%) of the corner's natural frequency.
      Means the suspension is being driven near its own resonance peak,
      which costs grip and tyre temperature.
    * **damper bias** — ratio ``|P10| / P90`` of rebound-to-bump damper
      speed percentiles. >1.5 ⇒ rebound is too soft (car ‘falls’ off
      bumps); <0.7 ⇒ rebound too stiff (car ‘packs down’).
    """
    items: list[AssessmentItem] = []
    for name, corner in static.corners.items():
        if name not in dynamic.wheels:
            continue
        wd = dynamic.wheels[name]
        f_n = corner.natural_freq_hz
        if f_n > 0:
            rel = abs(wd.dominant_freq_hz - f_n) / f_n
            status = ("red" if rel <= resonance_tol
                      else "amber" if rel <= 2 * resonance_tol
                      else "green")
            items.append(AssessmentItem(
                name=f"resonance_{name}",
                value=wd.dominant_freq_hz, units="Hz",
                target_low=f_n * (1 - resonance_tol),
                target_high=f_n * (1 + resonance_tol),
                status=status,
                note=(f"Observed dominant frequency vs corner natural "
                      f"frequency ({f_n:.2f} Hz)."),
            ))

        if wd.damper_speed_p90_mps > 1e-4:
            ratio = abs(wd.damper_speed_p10_mps) / wd.damper_speed_p90_mps
            status = ("green" if 0.7 <= ratio <= 1.5
                      else "amber" if 0.5 <= ratio <= 2.0
                      else "red")
            items.append(AssessmentItem(
                name=f"damper_bias_{name}",
                value=ratio, units="|P10|/P90",
                target_low=0.7, target_high=1.5,
                status=status,
                note=("Rebound-vs-bump damper-speed balance from the "
                      "observed histogram."),
            ))
    return items


__all__ = [
    "TARGETS_BY_CLASS",
    "AssessmentItem",
    "AxleSuspension",
    "CornerSuspension",
    "SuspensionDynamic",
    "SuspensionStatic",
    "WheelDynamics",
    "assess_static",
    "compare",
    "compute_dynamic",
    "compute_static",
]


# Keep the import sorter happy without shipping an unused export.
_ = field
