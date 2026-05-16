"""Tests for StintTelemetry multi-lap aggregator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lfs_telemetry.telemetry import LapTelemetry, StintTelemetry
from lfs_telemetry.telemetry.protocol.packets import WHEEL_ORDER


def _synthetic_lap_df(
    *,
    n: int = 600,
    fuel_start: float = 0.50,
    fuel_end: float = 0.45,
    speed_ms: float = 30.0,
) -> pd.DataFrame:
    """Build a minimal but enrich-compatible lap DataFrame at 100 Hz."""
    t_ms = np.arange(n, dtype=np.int64) * 10
    fuel = np.linspace(fuel_start, fuel_end, n)
    cols: dict[str, np.ndarray] = {
        "time_ms": t_ms,
        "ang_vel_x": np.zeros(n), "ang_vel_y": np.zeros(n), "ang_vel_z": np.full(n, 0.05),
        "heading": np.zeros(n), "pitch": np.zeros(n), "roll": np.zeros(n),
        "accel_x": np.full(n, 0.5), "accel_y": np.full(n, 0.3), "accel_z": np.full(n, 9.81),
        "vel_x": np.full(n, speed_ms), "vel_y": np.zeros(n), "vel_z": np.zeros(n),
        "pos_x": np.arange(n, dtype=float), "pos_y": np.zeros(n), "pos_z": np.zeros(n),
        "car": np.array(["FBM"] * n, dtype=object),
        "gear": np.full(n, 4), "speed_ms": np.full(n, speed_ms), "rpm": np.full(n, 8000.0),
        "throttle": np.full(n, 0.7), "brake": np.zeros(n), "clutch": np.zeros(n),
        "fuel": fuel, "eng_temp_c": np.full(n, 85.0), "oil_temp_c": np.full(n, 92.0),
        "oil_pressure_bar": np.full(n, 4.5), "turbo_bar": np.zeros(n),
        "og_flags": np.zeros(n, dtype=int), "dash_lights": np.zeros(n, dtype=int),
        "show_lights": np.zeros(n, dtype=int), "og_player_id": np.ones(n, dtype=int),
        "current_lap_dist_m": np.arange(n, dtype=float),
        "indexed_distance_m": np.arange(n, dtype=float),
        "steer_torque_nm": np.full(n, 5.0),
        "engine_ang_vel_rads": np.full(n, 800.0),
        "max_torque_at_vel_nm": np.full(n, 200.0),
        "input_throttle": np.full(n, 0.7), "input_brake": np.zeros(n),
        "input_steer": np.full(n, 0.05), "input_clutch": np.zeros(n),
        "input_handbrake": np.zeros(n),
    }
    for c in WHEEL_ORDER:
        cols[f"wheel_{c}_susp_deflect_m"] = np.full(n, 0.02)
        cols[f"wheel_{c}_vertical_load_n"] = np.full(n, 1500.0)
        cols[f"wheel_{c}_slip_ratio"] = np.zeros(n)
        cols[f"wheel_{c}_tan_slip_angle"] = np.zeros(n)
        cols[f"wheel_{c}_x_force_n"] = np.zeros(n)
        cols[f"wheel_{c}_y_force_n"] = np.zeros(n)
        cols[f"wheel_{c}_ang_vel_rads"] = np.full(n, 20.0)
        cols[f"wheel_{c}_lean_rel_road_rad"] = np.zeros(n)
        cols[f"wheel_{c}_air_temp_c"] = np.full(n, 25)
        cols[f"wheel_{c}_slip_fraction"] = np.zeros(n)
        cols[f"wheel_{c}_touching"] = np.ones(n, dtype=int)
        cols[f"wheel_{c}_steer_rad"] = np.zeros(n)
    return pd.DataFrame(cols)


def _make_lap(**kwargs) -> LapTelemetry:
    return LapTelemetry.from_dataframe(_synthetic_lap_df(**kwargs), car="FBM")


def test_stint_from_laps_aggregates_and_trends() -> None:
    laps = [
        _make_lap(fuel_start=0.50, fuel_end=0.45),
        _make_lap(fuel_start=0.45, fuel_end=0.40),
        _make_lap(fuel_start=0.40, fuel_end=0.35),
    ]
    stint = StintTelemetry.from_laps(laps)
    assert len(stint) == 3

    df = stint.per_lap
    assert list(df["lap_index"]) == [1, 2, 3]
    assert (df["car"] == "FBM").all()
    assert df["fuel_pct_used"].between(4.9, 5.1).all()

    trends = stint.trends
    assert trends["num_laps"] == 3
    assert trends["car"] == "FBM"
    assert trends["fuel_pct_per_lap_mean"] == pytest.approx(5.0, abs=0.1)
    assert trends["fuel_laps_remaining"] == pytest.approx(7.0, abs=0.1)
    assert abs(trends["pace_dropoff_s_per_lap"]) < 0.05


def test_stint_to_csv_round_trip(tmp_path: Path) -> None:
    laps = [_make_lap(), _make_lap(fuel_start=0.45, fuel_end=0.40)]
    stint = StintTelemetry.from_laps(laps)
    out = stint.to_csv(tmp_path / "stint_summary.csv")
    df = pd.read_csv(out)
    assert len(df) == 2
    assert "fuel_pct_used" in df.columns


def test_stint_empty_returns_empty_structures() -> None:
    stint = StintTelemetry(laps=[])
    assert len(stint) == 0
    assert stint.per_lap.empty
    assert stint.trends == {}
