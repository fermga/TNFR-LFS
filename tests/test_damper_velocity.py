"""Tests for damper-velocity derived columns + histogram analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lfs_telemetry.telemetry.car_calibration import CarSpec
from lfs_telemetry.telemetry.damper_histogram import (
    DEFAULT_LOW_SPEED_MPS,
    damper_histogram,
)
from lfs_telemetry.telemetry.derived import enrich_dataframe


def _synthetic_with_damper(n: int = 200) -> pd.DataFrame:
    """A short synthetic capture with sinusoidal suspension deflection."""
    t = np.arange(n) * 0.01  # 100 Hz
    df = pd.DataFrame({"time_ms": (t * 1000.0).astype(int), "car": "FOX"})
    # Each corner gets its own ~5 Hz sine of different amplitudes so the
    # derivative magnitude differs per wheel and we can verify the order
    # is preserved.
    amps = {"FL": 0.03, "FR": 0.03, "RL": 0.05, "RR": 0.05}
    for c, a in amps.items():
        df[f"wheel_{c}_susp_deflect_m"] = a * np.sin(2 * np.pi * 5.0 * t)
    return df


def test_enrich_adds_damper_velocity_columns():
    df = _synthetic_with_damper()
    out = enrich_dataframe(df, CarSpec())
    for c in ("FL", "FR", "RL", "RR"):
        col = f"wheel_{c}_susp_speed_mps"
        assert col in out.columns
        # First sample is the diff fill; remaining values must be finite.
        assert np.isfinite(out[col].iloc[1:]).all()
    # Rear corners have larger amplitude → larger peak velocity.
    peak_fl = out["wheel_FL_susp_speed_mps"].abs().max()
    peak_rl = out["wheel_RL_susp_speed_mps"].abs().max()
    assert peak_rl > peak_fl


def test_damper_velocity_sign_convention_bump_positive():
    """Compression (deflection growing) → positive velocity."""
    n = 50
    df = pd.DataFrame({
        "time_ms": np.arange(n) * 10,
        "car": ["FOX"] * n,
        # Monotonically increasing deflection = pure compression.
        "wheel_FL_susp_deflect_m": np.linspace(0.0, 0.05, n),
        "wheel_FR_susp_deflect_m": np.linspace(0.0, 0.05, n),
        "wheel_RL_susp_deflect_m": np.linspace(0.0, 0.05, n),
        "wheel_RR_susp_deflect_m": np.linspace(0.0, 0.05, n),
    })
    out = enrich_dataframe(df, CarSpec())
    speeds = out["wheel_FL_susp_speed_mps"].iloc[1:]
    assert (speeds > 0).all(), "monotone compression must yield positive speeds"


def test_damper_histogram_basic_shape_and_metrics():
    rng = np.random.default_rng(42)
    # Mostly low-speed bump-biased distribution.
    speeds = np.concatenate([
        rng.normal(0.005, 0.005, 800),    # low-speed bump
        rng.normal(-0.004, 0.005, 600),   # low-speed rebound
        rng.normal(0.060, 0.010, 80),     # high-speed bump (kerbs)
        rng.normal(-0.050, 0.008, 60),    # high-speed rebound
    ])
    hist = damper_histogram(speeds, low_speed_mps=DEFAULT_LOW_SPEED_MPS)
    # Counts sum to total (some clamped to extreme bins, none lost).
    assert int(hist.counts.sum()) == speeds.size
    # Symmetric bins around 0.
    assert hist.bins.size > 0
    assert abs(hist.bins[0] + hist.bins[-1]) < 1e-9
    # Bump average is positive, rebound average is positive (it's |·|).
    assert hist.bump_avg_mps > 0
    assert hist.rebound_avg_mps > 0
    # Quadrant percentages sum to ~100 (samples exactly at zero are rare
    # given the random draw but allow tiny slack).
    total = (hist.bump_low_pct + hist.bump_high_pct
             + hist.rebound_low_pct + hist.rebound_high_pct)
    assert 99.0 < total <= 100.5
    # Most samples are low-speed in this setup.
    assert (hist.bump_low_pct + hist.rebound_low_pct) > (
        hist.bump_high_pct + hist.rebound_high_pct
    )


def test_damper_histogram_handles_empty_input():
    hist = damper_histogram(np.zeros(0))
    assert hist.bins.size == 0
    assert hist.bump_avg_mps == 0.0
    assert hist.rebound_avg_mps == 0.0
