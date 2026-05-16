"""Tests for the indexed_distance_m fallback in _unwrapped_lap_arrays."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lfs_telemetry.telemetry.car_calibration import CarSpec
from lfs_telemetry.telemetry.comparison import _unwrapped_lap_arrays
from lfs_telemetry.telemetry.lap import LapTelemetry


def _make_lap(df: pd.DataFrame) -> LapTelemetry:
    return LapTelemetry(raw=df, car=CarSpec())


def test_unwrap_uses_current_lap_dist_when_present():
    n = 50
    df = pd.DataFrame({
        "time_ms": np.arange(n) * 10,
        "current_lap_dist_m": np.linspace(0, 500, n),
        "indexed_distance_m": np.linspace(1000, 1500, n),  # decoy
    })
    _, d, _ = _unwrapped_lap_arrays(_make_lap(df))
    # Anchored at 0 (first sample) — must follow current_lap_dist_m, not
    # the decoy column.
    assert d[0] == 0.0
    assert d[-1] == pytest_approx_500()


def test_unwrap_falls_back_to_indexed_distance_when_current_missing():
    n = 50
    df = pd.DataFrame({
        "time_ms": np.arange(n) * 10,
        "indexed_distance_m": np.linspace(2000, 2500, n),
    })
    _, d, _ = _unwrapped_lap_arrays(_make_lap(df))
    assert d.size == n
    assert d[0] == 0.0
    assert d[-1] == pytest_approx_500()


def test_unwrap_falls_back_when_current_is_all_nan():
    n = 50
    df = pd.DataFrame({
        "time_ms": np.arange(n) * 10,
        "current_lap_dist_m": np.full(n, np.nan),
        "indexed_distance_m": np.linspace(0, 500, n),
    })
    _, d, _ = _unwrapped_lap_arrays(_make_lap(df))
    assert d.size == n
    assert d[-1] == pytest_approx_500()


def pytest_approx_500() -> float:
    # Tiny helper to keep the assertion expressive without needing a
    # pytest.approx import everywhere.
    return 500.0
