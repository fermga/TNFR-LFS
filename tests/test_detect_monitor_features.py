"""Tests for the Detect&Monitor-style additions: fuel_usage,
clean/stint/total/rolling averages, and traffic time gaps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lfs_telemetry.telemetry import LapTelemetry, StintTelemetry
from lfs_telemetry.telemetry.protocol.packets import WHEEL_ORDER, CompCar
from lfs_telemetry.telemetry.traffic import _build_snapshot


def _synthetic_lap_df(
    *, n: int = 600, fuel_start: float = 0.50, fuel_end: float = 0.45,
    speed_ms: float = 30.0, race_start: bool = False,
) -> pd.DataFrame:
    t_ms = np.arange(n, dtype=np.int64) * 10
    fuel = np.linspace(fuel_start, fuel_end, n)
    speed = np.full(n, speed_ms)
    if race_start:
        # First few seconds at standstill → triggers is_race_start
        n_zero = min(200, n // 2)
        speed[:n_zero] = 0.0
    cols: dict[str, np.ndarray] = {
        "time_ms": t_ms,
        "ang_vel_x": np.zeros(n), "ang_vel_y": np.zeros(n), "ang_vel_z": np.full(n, 0.05),
        "heading": np.zeros(n), "pitch": np.zeros(n), "roll": np.zeros(n),
        "accel_x": np.full(n, 0.5), "accel_y": np.full(n, 0.3), "accel_z": np.full(n, 9.81),
        "vel_x": speed, "vel_y": np.zeros(n), "vel_z": np.zeros(n),
        "pos_x": np.arange(n, dtype=float), "pos_y": np.zeros(n), "pos_z": np.zeros(n),
        "car": np.array(["FBM"] * n, dtype=object),
        "gear": np.full(n, 4), "speed_ms": speed, "rpm": np.full(n, 8000.0),
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


# ---------------------------------------------------------------------------
# fuel_usage
# ---------------------------------------------------------------------------


def test_fuel_usage_basic():
    laps = [
        _make_lap(fuel_start=0.50, fuel_end=0.45),
        _make_lap(fuel_start=0.45, fuel_end=0.40),
        _make_lap(fuel_start=0.40, fuel_end=0.35),
    ]
    stint = StintTelemetry.from_laps(laps)
    fu = stint.fuel_usage
    assert fu["mean_pct"] == pytest.approx(5.0, abs=0.1)
    assert fu["last_fuel_pct"] == pytest.approx(35.0, abs=0.1)
    assert fu["laps_remaining"] == pytest.approx(7.0, abs=0.1)
    assert len(fu["per_lap_pct"]) == 3


def test_fuel_usage_excludes_race_start_from_mean():
    # Race-start lap with weird consumption would pollute the mean.
    rs = _make_lap(fuel_start=0.50, fuel_end=0.30, race_start=True)
    flying = [
        _make_lap(fuel_start=0.30, fuel_end=0.25),
        _make_lap(fuel_start=0.25, fuel_end=0.20),
    ]
    stint = StintTelemetry.from_laps([rs, *flying])
    assert 1 in stint.race_start_lap_indices
    fu = stint.fuel_usage
    # mean only across the two flying laps (5%) — not across all three (10%).
    assert fu["mean_pct"] == pytest.approx(5.0, abs=0.1)


# ---------------------------------------------------------------------------
# average_lap_time modes
# ---------------------------------------------------------------------------


def test_average_lap_time_modes():
    rs = _make_lap(race_start=True)                          # lap 1 (slow)
    flying = [_make_lap() for _ in range(4)]                 # laps 2..5
    stint = StintTelemetry.from_laps([rs, *flying])
    stint.mark_lap_invalid(3)                                # OBH on lap 3

    total = stint.average_lap_time("total")
    stint_avg = stint.average_lap_time("stint")
    clean = stint.average_lap_time("clean")
    rolling = stint.average_lap_time("rolling", rolling=2)

    # All three identical lap times for synthetic flying laps → stint == clean
    # numerically; total should differ if race-start lap differs.
    assert total is not None and stint_avg is not None
    assert clean is not None and rolling is not None
    # clean drops lap 3 → indices [2,4,5]; rolling tail(2) → [4,5]
    assert stint.clean_lap_indices == [2, 4, 5]


def test_average_lap_time_unknown_mode_raises():
    stint = StintTelemetry.from_laps([_make_lap()])
    with pytest.raises(ValueError):
        stint.average_lap_time("nonsense")
    with pytest.raises(ValueError):
        stint.average_lap_time("rolling")  # missing rolling arg


def test_mark_invalid_from_records():
    laps = [_make_lap() for _ in range(3)]
    stint = StintTelemetry.from_laps(laps)

    class _Rec:
        def __init__(self, valid=True, obh=0):
            self.valid = valid
            self.obh_count = obh

    stint.mark_invalid_from_records([
        _Rec(valid=True),                # lap 1 ok
        _Rec(valid=False),               # lap 2 invalid (HLV)
        _Rec(valid=True, obh=2),         # lap 3 invalid (object hits)
    ])
    assert stint.invalid_lap_indices == {2, 3}
    assert stint.clean_lap_indices == [1]


# ---------------------------------------------------------------------------
# Traffic time gaps
# ---------------------------------------------------------------------------


def _cc(plid: int, pos: int, x: float, y: float, speed: float) -> CompCar:
    return CompCar(node=0, lap=1, player_id=plid, position=pos,
                   info=0, x_m=x, y_m=y, z_m=0.0, speed_ms=speed,
                   direction_rad=0.0, heading_rad=0.0, ang_vel_rads=0.0)


def test_traffic_time_gap_ahead_and_behind():
    view = _cc(plid=1, pos=2, x=0.0, y=0.0, speed=50.0)
    ahead = _cc(plid=2, pos=1, x=100.0, y=0.0, speed=48.0)
    behind = _cc(plid=3, pos=3, x=-200.0, y=0.0, speed=52.0)
    snap = _build_snapshot(view, [view, ahead, behind])
    assert snap.gap_to_ahead_m == pytest.approx(100.0)
    # gap_s for ahead uses view speed (50 m/s) → 2.0 s
    assert snap.gap_to_ahead_s == pytest.approx(2.0)
    # gap_s for behind also uses view speed (50 m/s, timing-tower
    # convention: both gaps are computed at the reference car's
    # rate so they tick down at a comparable cadence) → 200/50
    assert snap.gap_to_behind_s == pytest.approx(200.0 / 50.0)


def test_traffic_time_gap_none_when_stationary():
    view = _cc(plid=1, pos=2, x=0.0, y=0.0, speed=0.0)
    ahead = _cc(plid=2, pos=1, x=100.0, y=0.0, speed=0.0)
    snap = _build_snapshot(view, [view, ahead])
    assert snap.gap_to_ahead_m == pytest.approx(100.0)
    # speed below 0.5 m/s threshold → no time gap
    assert snap.gap_to_ahead_s is None


def test_traffic_skips_disconnect_gap_in_positions():
    # Positions 1, 2, _, 4 — driver at position 3 disconnected. View
    # at pos 4 should find pos 2 as ahead (not silently fall back to
    # spatial because pos 3 is missing).
    view = _cc(plid=1, pos=4, x=0.0, y=0.0, speed=50.0)
    ahead = _cc(plid=2, pos=2, x=300.0, y=0.0, speed=50.0)
    leader = _cc(plid=3, pos=1, x=900.0, y=0.0, speed=50.0)
    snap = _build_snapshot(view, [view, ahead, leader])
    assert snap.car_ahead_plid == 2  # immediate-above, not skipped
    assert snap.gap_to_ahead_m == pytest.approx(300.0)


def test_traffic_spatial_skips_stationary_pit_car():
    # View moving on track; one opponent racing behind, another
    # stationary (pit/spectator). Spatial fallback must pick the
    # racing one, not the pit car which is geometrically closer.
    view = _cc(plid=1, pos=0, x=0.0, y=0.0, speed=50.0)
    racing_behind = _cc(plid=2, pos=0, x=0.0, y=-80.0, speed=50.0)
    pit_car = _cc(plid=3, pos=0, x=0.0, y=-20.0, speed=0.0)
    snap = _build_snapshot(view, [view, racing_behind, pit_car])
    assert snap.car_behind_plid == 2  # racing one, not the pit car


def test_traffic_gap_arclength_falls_back_to_euclidean_on_lap_wrap():
    # Two cars physically adjacent (eu ≈ 5 m) but at opposite ends of
    # the node table → node arclength would wrap to ~track_length.
    # Sanity check must return the euclidean value instead.
    from lfs_telemetry.telemetry.traffic import _gap_on_track_m
    track_length = 3000.0
    node_to_s = [float(i) for i in range(0, int(track_length))]
    view = _cc(plid=1, pos=0, x=0.0, y=0.0, speed=50.0)
    view.node = 5  # s=5.0
    other = _cc(plid=2, pos=0, x=5.0, y=0.0, speed=50.0)
    other.node = 2995  # s=2995.0 → forward gap by node ≈ 2990 m
    gap = _gap_on_track_m(
        view, other, forward=True,
        node_to_s_m=node_to_s, track_length_m=track_length,
    )
    assert gap == pytest.approx(5.0)  # euclidean, not 2990


def test_resilient_text_stream_swallows_oserror():
    """Frozen Windows builds can have stdout/stderr raise OSError 22;
    the resilient wrapper must absorb those without re-raising."""
    import io as _io

    from lfs_telemetry.cli import _ResilientTextStream

    class _Brokenstream(_io.StringIO):
        def write(self, s):
            raise OSError(22, "Invalid argument")
        def flush(self):
            raise OSError(22, "Invalid argument")

    rs = _ResilientTextStream(_Brokenstream())
    # Should not raise
    assert rs.write("hello") == len("hello")
    rs.flush()
    # None inner is also tolerated
    rs2 = _ResilientTextStream(None)
    assert rs2.write("x") == 1
    rs2.flush()
