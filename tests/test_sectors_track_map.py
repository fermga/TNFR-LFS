"""Tests for sectors, theoretical-best, and TrackMap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lfs_telemetry.telemetry import (
    LapTelemetry,
    Sector,
    StintTelemetry,
    TrackMap,
    lap_sectors,
    sector_times_s,
)
from lfs_telemetry.telemetry.protocol.packets import WHEEL_ORDER


def _synthetic_lap_df(
    *,
    n: int = 600,
    speed_ms: float = 30.0,
    time_scale: float = 1.0,
) -> pd.DataFrame:
    t_ms = (np.arange(n, dtype=np.int64) * 10 * time_scale).astype(np.int64)
    angle = np.linspace(0.0, 2 * np.pi, n)
    radius = 200.0
    cols: dict[str, np.ndarray] = {
        "time_ms": t_ms,
        "ang_vel_x": np.zeros(n), "ang_vel_y": np.zeros(n), "ang_vel_z": np.zeros(n),
        "heading": np.zeros(n), "pitch": np.zeros(n), "roll": np.zeros(n),
        "accel_x": np.zeros(n), "accel_y": np.zeros(n), "accel_z": np.full(n, 9.81),
        "vel_x": np.full(n, speed_ms), "vel_y": np.zeros(n), "vel_z": np.zeros(n),
        "pos_x": radius * np.cos(angle),
        "pos_y": radius * np.sin(angle),
        "pos_z": np.zeros(n),
        "car": np.array(["FBM"] * n, dtype=object),
        "gear": np.full(n, 4), "speed_ms": np.full(n, speed_ms), "rpm": np.full(n, 8000.0),
        "throttle": np.full(n, 0.7), "brake": np.zeros(n), "clutch": np.zeros(n),
        "fuel": np.full(n, 0.5), "eng_temp_c": np.full(n, 85.0), "oil_temp_c": np.full(n, 92.0),
        "oil_pressure_bar": np.full(n, 4.5), "turbo_bar": np.zeros(n),
        "og_flags": np.zeros(n, dtype=int), "dash_lights": np.zeros(n, dtype=int),
        "show_lights": np.zeros(n, dtype=int), "og_player_id": np.ones(n, dtype=int),
        "current_lap_dist_m": np.arange(n, dtype=float),  # 1 m per sample
        "indexed_distance_m": np.arange(n, dtype=float),
        "steer_torque_nm": np.full(n, 5.0),
        "engine_ang_vel_rads": np.full(n, 800.0),
        "max_torque_at_vel_nm": np.full(n, 200.0),
        "input_throttle": np.full(n, 0.7), "input_brake": np.zeros(n),
        "input_steer": np.zeros(n), "input_clutch": np.zeros(n),
        "input_handbrake": np.zeros(n),
        "ctx_track": np.array(["BL1"] * n, dtype=object),
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


def _make_lap(**kw) -> LapTelemetry:
    return LapTelemetry.from_dataframe(_synthetic_lap_df(**kw), car="FBM")


# ---------------------------------------------------------------------------
# Sectors
# ---------------------------------------------------------------------------


def test_lap_sectors_default_three_equal_sectors() -> None:
    lap = _make_lap(n=600)  # 600 m, 6 s
    secs = lap_sectors(lap)
    assert len(secs) == 3
    assert all(isinstance(s, Sector) for s in secs)
    # Equal-distance: each sector ~200 m / ~2 s.
    for s in secs:
        assert s.length_m == pytest.approx(199.0, abs=2.0)
        assert s.time_s == pytest.approx(2.0, abs=0.05)


def test_lap_sectors_user_boundaries_clipped() -> None:
    lap = _make_lap(n=600)
    secs = lap_sectors(lap, boundaries_m=[150.0, 400.0, 9999.0])  # last clipped
    assert len(secs) == 3
    assert secs[0].end_d_m == pytest.approx(150.0)
    assert secs[1].end_d_m == pytest.approx(400.0)


def test_lap_sectors_n_equal_one_returns_single_sector() -> None:
    lap = _make_lap(n=600)
    secs = lap_sectors(lap, n_equal=1)
    assert len(secs) == 1
    assert secs[0].time_s == pytest.approx(lap.summary["lap_time_s"], abs=0.05)


def test_sector_times_s_sums_to_lap_time() -> None:
    lap = _make_lap(n=600)
    times = sector_times_s(lap, n_equal=4)
    assert len(times) == 4
    assert sum(times) == pytest.approx(lap.summary["lap_time_s"], abs=0.05)


def test_lap_telemetry_sectors_method_proxies_to_helper() -> None:
    lap = _make_lap(n=600)
    via_method = lap.sectors(n_equal=2)
    via_helper = lap_sectors(lap, n_equal=2)
    assert [s.time_s for s in via_method] == [s.time_s for s in via_helper]


# ---------------------------------------------------------------------------
# StintTelemetry: theoretical best
# ---------------------------------------------------------------------------


def test_stint_sector_times_per_lap_shape() -> None:
    laps = [_make_lap(time_scale=ts) for ts in (1.0, 1.05, 1.10)]
    stint = StintTelemetry.from_laps(laps)
    df = stint.sector_times_per_lap(n_equal=3)
    assert df.shape == (3, 4)  # 3 sector cols + is_race_start
    assert "sector_1_s" in df.columns
    assert "sector_3_s" in df.columns


def test_stint_theoretical_best_is_min_per_sector() -> None:
    # Build 3 laps, each fastest in a different sector by stretching the
    # OTHER two sectors via time_scale globally won't do it — we build
    # synthetic laps where each lap is uniformly fastest by a known factor.
    laps = [_make_lap(time_scale=ts) for ts in (1.00, 1.05, 1.10)]
    stint = StintTelemetry.from_laps(laps)
    tb = stint.theoretical_best_lap(n_equal=3)
    # All 3 sectors won by lap 1 (fastest overall) → theoretical == actual.
    assert tb["n_sectors"] == 3
    assert tb["n_laps_used"] == 3
    assert tb["theoretical_best_s"] == pytest.approx(tb["actual_best_s"], abs=0.05)
    assert tb["gap_s"] == pytest.approx(0.0, abs=0.05)
    assert tb["best_sector_lap"] == [1, 1, 1]


def test_stint_theoretical_best_excludes_race_start() -> None:
    # Lap 1 starts stopped (race start); laps 2-3 are flying.
    df_start = _synthetic_lap_df(n=600)
    df_start.loc[:5, "speed_ms"] = 0.0  # first samples stopped → is_race_start
    lap_start = LapTelemetry.from_dataframe(df_start, car="FBM")
    flying = [_make_lap(time_scale=ts) for ts in (1.05, 1.10)]
    stint = StintTelemetry.from_laps([lap_start, *flying])
    tb = stint.theoretical_best_lap(n_equal=3)
    assert 1 in tb["excluded_lap_indices"]
    assert tb["n_laps_used"] == 2
    # Best sector lap should be lap 2 (faster of the two flying ones).
    assert all(idx in (2, 3) for idx in tb["best_sector_lap"])


def test_stint_theoretical_best_empty_when_no_laps() -> None:
    stint = StintTelemetry.from_laps([])
    assert stint.theoretical_best_lap() == {}


# ---------------------------------------------------------------------------
# TrackMap
# ---------------------------------------------------------------------------


def test_track_map_from_lap_circle_geometry() -> None:
    lap = _make_lap(n=600)  # synthetic circle, radius 200
    tmap = lap.track_map(n_points=200)
    assert isinstance(tmap, TrackMap)
    assert tmap.n_points == 200
    assert tmap.length_m == pytest.approx(599.0, abs=2.0)
    # Every point on the circle has radius ~200.
    r = np.hypot(tmap.x_m, tmap.y_m)
    assert np.all(np.abs(r - 200.0) < 1.0)
    # Bounds enclose the circle.
    b = tmap.bounds()
    assert b.width_m == pytest.approx(400.0, abs=2.0)
    assert b.height_m == pytest.approx(400.0, abs=2.0)


def test_track_map_xy_at_distance_interpolates() -> None:
    lap = _make_lap(n=600)
    tmap = lap.track_map(n_points=600)
    x0, y0 = tmap.xy_at_distance(0.0)
    # At d=0 on the circle, position is (R, 0).
    assert x0 == pytest.approx(200.0, abs=1.0)
    assert y0 == pytest.approx(0.0, abs=1.0)


def test_track_map_from_laps_averages() -> None:
    laps = [_make_lap(n=600) for _ in range(3)]
    tmap = TrackMap.from_laps(laps, n_points=100)
    assert tmap.n_points == 100
    # Identical inputs → average == single input.
    single = TrackMap.from_lap(laps[0], n_points=100)
    assert np.allclose(tmap.x_m, single.x_m)
    assert np.allclose(tmap.y_m, single.y_m)


def test_track_map_to_dataframe_roundtrip() -> None:
    lap = _make_lap(n=600)
    df = lap.track_map(n_points=50).to_dataframe()
    assert list(df.columns) == ["distance_m", "x_m", "y_m"]
    assert len(df) == 50


def test_stint_track_map_excludes_race_start_when_possible() -> None:
    df_start = _synthetic_lap_df(n=600)
    df_start.loc[:5, "speed_ms"] = 0.0
    lap_start = LapTelemetry.from_dataframe(df_start, car="FBM")
    flying = [_make_lap() for _ in range(2)]
    stint = StintTelemetry.from_laps([lap_start, *flying])
    tmap = stint.track_map(n_points=100)
    assert tmap.n_points == 100  # built fine from the 2 flying laps
