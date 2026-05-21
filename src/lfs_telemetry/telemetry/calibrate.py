"""Telemetry-based calibration of per-car physical parameters.

LFS does not expose tyre coefficients directly, and mod-car ``.vob`` files
are encrypted. The only reliable cross-car estimator we have is the
telemetry itself. This module fits two key parameters from a recorded
session:

* ``mu_lat``  — peak lateral grip coefficient (a_lat,max / g)
* ``mu_long`` — peak combined longitudinal grip (max of |a_long|/g over
  high-throttle and high-brake samples)

Optionally, when extended OutSim wheel data is present (``vert_load_*``
columns), it also estimates ``mass_kg`` from the rest sum of vertical loads.

The estimator uses a high percentile (default 98th) instead of the raw
maximum to suppress single-frame outliers (curb hits, contacts).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .constants import GRAVITY
from .observables import CORNERS, CarSpec, car_spec_for


def _percentile(arr: np.ndarray, q: float) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def estimate_mu_lat(df: pd.DataFrame, *,
                    speed_min_ms: float = 8.0,
                    percentile: float = 98.0) -> float:
    """Estimate the peak lateral grip coefficient from telemetry.

    Uses ``|lat_accel| / g`` at samples where the car is moving and the
    driver is not on the brakes hard (so the lateral budget is full).
    """
    if "lat_accel" not in df:
        return float("nan")
    mask = np.ones(len(df), dtype=bool)
    if "speed_ms" in df:
        mask &= df["speed_ms"].to_numpy() >= speed_min_ms
    if "brake" in df:
        mask &= df["brake"].to_numpy() < 0.5
    if not mask.any():
        return float("nan")
    a = np.abs(df.loc[mask, "lat_accel"].to_numpy()) / GRAVITY
    return _percentile(a, percentile)


def estimate_mu_long(df: pd.DataFrame, *,
                     speed_min_ms: float = 8.0,
                     percentile: float = 98.0) -> float:
    """Estimate the peak longitudinal grip coefficient from telemetry.

    Considers both braking (deceleration) and traction (acceleration) and
    returns the larger of the two percentiles. We require low lateral
    activity so we sample (close to) pure-longitudinal events.
    """
    if "long_accel" not in df:
        return float("nan")
    mask = np.ones(len(df), dtype=bool)
    if "speed_ms" in df:
        mask &= df["speed_ms"].to_numpy() >= speed_min_ms
    if "lat_accel" in df:
        mask &= np.abs(df["lat_accel"].to_numpy()) / GRAVITY < 0.4
    if not mask.any():
        return float("nan")
    a = df.loc[mask, "long_accel"].to_numpy() / GRAVITY
    decel = _percentile(np.abs(a[a < 0]), percentile)
    accel = _percentile(a[a > 0], percentile)
    candidates = [v for v in (decel, accel) if np.isfinite(v)]
    if not candidates:
        return float("nan")
    return max(candidates)


def estimate_mu_lat_curve(df: pd.DataFrame, *,
                          speed_min_ms: float = 8.0,
                          percentile: float = 98.0,
                          n_bins: int = 8,
                          min_per_bin: int = 30) -> tuple[float, float, int]:
    """Fit μ_lat(v) ≈ μ0 + k·v² from telemetry.

    Bins samples by speed, takes the high percentile of |a_y|/g per bin
    to estimate the *peak* μ at that speed, then linear-regresses against
    v². Returns ``(mu0, k, n_used_bins)``. ``k`` is in (m/s)⁻².

    Falls back to ``(estimate_mu_lat(df), 0.0, 0)`` if there are fewer
    than 3 usable bins (so non-aero cars stay flat).
    """
    if "lat_accel" not in df or "speed_ms" not in df:
        return (estimate_mu_lat(df, percentile=percentile), 0.0, 0)
    speed = df["speed_ms"].to_numpy()
    a_lat = np.abs(df["lat_accel"].to_numpy()) / GRAVITY
    brake = (df["brake"].to_numpy() if "brake" in df
             else np.zeros(len(df)))
    mask = (speed >= speed_min_ms) & (brake < 0.5) & np.isfinite(a_lat)
    if mask.sum() < min_per_bin * 3:
        return (estimate_mu_lat(df, percentile=percentile), 0.0, 0)
    speed = speed[mask]
    a_lat = a_lat[mask]
    v_lo, v_hi = float(speed.min()), float(speed.max())
    if v_hi - v_lo < 10.0:
        return (estimate_mu_lat(df, percentile=percentile), 0.0, 0)
    edges = np.linspace(v_lo, v_hi, n_bins + 1)
    v_centres = []
    mu_peaks = []
    for i in range(n_bins):
        m = (speed >= edges[i]) & (speed < edges[i + 1])
        if m.sum() < min_per_bin:
            continue
        v_centres.append(0.5 * (edges[i] + edges[i + 1]))
        mu_peaks.append(_percentile(a_lat[m], percentile))
    if len(v_centres) < 3:
        return (estimate_mu_lat(df, percentile=percentile), 0.0, 0)
    v_arr = np.asarray(v_centres)
    mu_arr = np.asarray(mu_peaks)
    # Linear regression mu = mu0 + k * v²; clamp k to be non-negative
    # (negative k means tyres losing grip at speed → not the aero model
    # we want; in that case fall back to constant μ).
    v2 = v_arr ** 2
    slope, intercept = np.polyfit(v2, mu_arr, 1)
    if slope < 0:
        return (float(np.mean(mu_arr)), 0.0, len(v_centres))
    return (float(intercept), float(slope), len(v_centres))


def estimate_mass_kg(df: pd.DataFrame) -> float:
    """Estimate vehicle mass from extended OutSim wheel loads, if present.

    Looks for either the legacy ``vert_load_{fl,fr,rl,rr}`` columns or the
    canonical ``wheel_{FL,FR,RL,RR}_vertical_load_n`` schema produced by
    :mod:`lfs_telemetry.telemetry.replay`. Returns NaN if neither is available.
    """
    legacy = [f"vert_load_{c.lower()}" for c in CORNERS]
    canonical = [f"wheel_{c}_vertical_load_n" for c in CORNERS]
    if all(c in df.columns for c in canonical):
        cols = canonical
    elif all(c in df.columns for c in legacy):
        cols = legacy
    else:
        return float("nan")
    if "lat_accel" in df and "long_accel" in df:
        # Filter for near-static samples (no transfer).
        mask = ((np.abs(df["lat_accel"]) < 1.0)
                & (np.abs(df["long_accel"]) < 1.0))
        sub = df.loc[mask, cols] if mask.any() else df[cols]
    else:
        sub = df[cols]
    total_n = sub.sum(axis=1).median()
    if not np.isfinite(total_n) or total_n <= 0:
        return float("nan")
    return float(total_n / GRAVITY)


def calibrate_spec(df: pd.DataFrame,
                   car_name: str | None = None,
                   *,
                   percentile: float = 98.0,
                   safety_factor: float = 0.95) -> CarSpec:
    """Return a :class:`CarSpec` whose μ (and mass, if measurable) come from
    telemetry rather than the static table.

    ``safety_factor`` (<=1) trims the percentile to leave a small headroom
    for tyre wear / fuel load drift. Only fields with a finite estimate are
    overridden — everything else stays from the official table.
    """
    base = car_spec_for(car_name)
    overrides: dict[str, float] = {}
    mu0, k_aero, _n_bins = estimate_mu_lat_curve(df, percentile=percentile)
    if np.isfinite(mu0):
        overrides["mu_lat"] = mu0 * safety_factor
        overrides["mu_lat_aero_k"] = k_aero * safety_factor
    else:
        mu_lat = estimate_mu_lat(df, percentile=percentile)
        if np.isfinite(mu_lat):
            overrides["mu_lat"] = mu_lat * safety_factor
            overrides["mu_lat_aero_k"] = 0.0
    mu_long = estimate_mu_long(df, percentile=percentile)
    if np.isfinite(mu_long):
        overrides["mu_long"] = mu_long * safety_factor
    mass = estimate_mass_kg(df)
    if np.isfinite(mass) and mass > 50.0:
        overrides["mass_kg"] = mass
    if not overrides:
        return base
    return replace(base, **overrides)


def calibration_report(df: pd.DataFrame,
                       car_name: str | None = None) -> dict[str, float]:
    """Return a dict of raw estimates (no safety factor) for diagnostics."""
    mu0, k_aero, n_bins = estimate_mu_lat_curve(df, percentile=98.0)
    return {
        "car_name": car_name or "?",
        "n_samples": len(df),
        "mu_lat_p98": estimate_mu_lat(df, percentile=98.0),
        "mu_lat_p99": estimate_mu_lat(df, percentile=99.0),
        "mu_lat_curve_mu0": mu0,
        "mu_lat_curve_k_aero": k_aero,
        "mu_lat_curve_bins": n_bins,
        "mu_long_p98": estimate_mu_long(df, percentile=98.0),
        "mu_long_p99": estimate_mu_long(df, percentile=99.0),
        "mass_kg_estimate": estimate_mass_kg(df),
    }
