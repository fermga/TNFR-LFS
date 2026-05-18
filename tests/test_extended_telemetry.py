"""Tests for the new InSim packets, derived physics and lap summary."""

from __future__ import annotations

import struct

import numpy as np
import pandas as pd
import pytest

from lfs_telemetry.telemetry import (
    build_lap_records,
    enrich_dataframe,
    traffic_snapshot,
)
from lfs_telemetry.telemetry.observables import CarSpec
from lfs_telemetry.telemetry.protocol.insim import RaceContext
from lfs_telemetry.telemetry.protocol.packets import (
    CCI_BLUE,
    DL_ABS,
    DL_PITSPEED,
    DL_TC,
    HLVC_WALL,
    CompCar,
    InSimHotLapValid,
    InSimLap,
    InSimMCI,
    InSimNewPlayer,
    InSimNodeLap,
    InSimObjectHit,
    InSimSplit,
    NodeLap,
    decode_dash_lights,
    hlvc_name,
)

# ---------------------------------------------------------------------------
# Bit decoders
# ---------------------------------------------------------------------------


def test_decode_dash_lights():
    bits = DL_TC | DL_ABS | DL_PITSPEED
    names = decode_dash_lights(bits)
    assert "tc" in names
    assert "abs" in names
    assert "pit_limiter" in names
    assert "shift" not in names


def test_hlvc_name():
    assert hlvc_name(0) == "ground"
    assert hlvc_name(1) == "wall"
    assert hlvc_name(99).startswith("hlvc_")


# ---------------------------------------------------------------------------
# IS_OBH / IS_NLP wire parse
# ---------------------------------------------------------------------------


def test_object_hit_parse():
    payload = struct.pack(
        "<BBBB HH BBBB hh HH BBBB",
        6, 51, 0, 7,                # Size/4=6, Type=ISP_OBH, ReqI=0, PLID=7
        # SpClose is 12 bits; 10 units == 1 m/s per InSim.txt v10.
        # 150 -> 15.0 m/s. Top nibble is OBH flag bits, leave zero here.
        150, 1234,
        128, 64, 25, 0,             # CarContOBJ Direction, Heading, Speed, Zb
        100, -50,                   # contact X/Y * 16
        300 * 16, 200 * 16,         # map X/Y in 1/16 m
        4, 0, 17, 0x05,             # Zbyte, Sp1, Index, OBHFlags
    )
    pkt = InSimObjectHit.parse(payload)
    assert pkt.player_id == 7
    assert pkt.closing_speed_ms == pytest.approx(15.0, rel=1e-3)
    assert pkt.time_ms == 12340
    assert pkt.contact_speed_ms == 25
    assert pkt.contact_x_m == pytest.approx(100 / 16.0)
    assert pkt.map_x_m == pytest.approx(300.0)
    assert pkt.object_index == 17
    assert pkt.flags == 0x05


def test_node_lap_parse():
    # Header: Size/4=4, Type=ISP_NLP=37, ReqI=0, NumP=2, then 2*NodeLap(6).
    payload = struct.pack(
        "<BBBB HHBB HHBB",
        4, 37, 0, 2,
        100, 5, 1, 1,
        110, 5, 2, 2,
    )
    pkt = InSimNodeLap.parse(payload)
    assert len(pkt.entries) == 2
    assert pkt.entries[0] == NodeLap(node=100, lap=5, player_id=1, position=1)
    assert pkt.entries[1].player_id == 2


def test_mci_parse_binary_header_offsets():
    # Regression: parser used to read NumC at offset 4 (off by one) and
    # CompCar array at offset 8 (off by 4), producing
    # "unpack_from requires a buffer of at least N+4 bytes" on every
    # IS_MCI packet. Real LFS layout: Size, Type, ReqI, NumC then the
    # CompCar array at offset 4. Total size = 4 + NumC*28.
    numc = 2
    car0 = struct.pack("<HHBBBB iii HHHh",
                       100, 3, 1, 1, 0, 0,
                       1 << 16, 2 << 16, 3 << 16,
                       3276, 0, 0, 0)  # ~10 m/s
    car1 = struct.pack("<HHBBBB iii HHHh",
                       110, 3, 2, 2, 0, 0,
                       4 << 16, 5 << 16, 6 << 16,
                       6553, 0, 0, 0)  # ~20 m/s
    size = 4 + numc * 28
    payload = struct.pack("<BBBB", size // 4, 38, 0, numc) + car0 + car1
    assert len(payload) == size
    pkt = InSimMCI.parse(payload)
    assert len(pkt.cars) == 2
    assert pkt.cars[0].player_id == 1
    assert pkt.cars[0].x_m == 1.0
    assert pkt.cars[1].player_id == 2
    assert pkt.cars[1].position == 2


# ---------------------------------------------------------------------------
# RaceContext history (laps, splits, fuel, HLV, OBH, MCI)
# ---------------------------------------------------------------------------


def _make_lap(plid: int, lap_ms: int, etime: int, fuel200: int) -> InSimLap:
    return InSimLap(player_id=plid, laps_done=1, lap_time_ms=lap_ms,
                    elapsed_time_ms=etime, flags=0, penalty=0,
                    num_stops=0, fuel200=fuel200)


def _make_split(plid: int, split: int, ms: int) -> InSimSplit:
    return InSimSplit(player_id=plid, split_time_ms=ms, elapsed_time_ms=ms,
                      split=split, penalty=0, num_stops=0, fuel200=0)


def test_race_context_lap_history():
    ctx = RaceContext()
    ctx.view_player_id = 1
    ctx.players[1] = InSimNewPlayer(
        player_id=1, connection_id=0, player_type=0, flags=0,
        player_name="x", plate="", car_name="FOX", skin_name="",
        tyres=(2, 2, 2, 2),
        handicap_mass_kg=0, handicap_t_res=0, model=0, passengers=0,
        rear_wheel_adjust=0, front_wheel_adjust=0,
        set_flags=0, num_in_race=1, config=0, fuel_pct=80,
    )
    # First lap: splits then lap.
    ctx.update(_make_split(1, 1, 30000))
    ctx.update(_make_split(1, 2, 60000))
    ctx.update(_make_lap(1, lap_ms=90000, etime=90000, fuel200=160))
    # Second lap.
    ctx.update(_make_split(1, 1, 29500))
    ctx.update(_make_lap(1, lap_ms=88000, etime=178000, fuel200=140))

    assert ctx.lap_count[1] == 2
    assert ctx.lap_times_ms[1] == [90000, 88000]
    assert ctx.split_times_ms[1][0] == {1: 30000, 2: 60000}
    assert ctx.split_times_ms[1][1] == {1: 29500}
    assert ctx.lap_fuel_pct[1] == [80.0, 70.0]


def test_race_context_obh_and_hlv_history():
    ctx = RaceContext()
    ctx.view_player_id = 3
    obh = InSimObjectHit(
        player_id=3, closing_speed_ms=10.0, time_ms=1000,
        contact_direction_rad=0.0, contact_heading_rad=0.0,
        contact_speed_ms=20, contact_x_m=0.0, contact_y_m=0.0,
        map_x_m=10.0, map_y_m=20.0, map_z_m=0.5,
        object_index=1, flags=1,
    )
    hlv = InSimHotLapValid(player_id=3, hlvc=HLVC_WALL,
                           time_ms=2000,
                           car_speed_ms=42.0, car_direction_rad=0.0,
                           car_heading_rad=0.0, car_x_m=0.0, car_y_m=0.0)
    ctx.update(obh)
    ctx.update(hlv)
    assert len(ctx.obh_events) == 1
    assert len(ctx.hlv_events) == 1
    assert ctx.last_hlv[3].hlvc == HLVC_WALL


def test_race_context_mci_traffic():
    ctx = RaceContext()
    ctx.view_player_id = 1
    cars = [
        CompCar(node=10, lap=1, player_id=2, position=1, info=0,
                x_m=100.0, y_m=0.0, z_m=0.0,
                speed_ms=55.0, direction_rad=0.0,
                heading_rad=0.0, ang_vel_rads=0.0),
        CompCar(node=10, lap=1, player_id=1, position=2, info=CCI_BLUE,
                x_m=70.0, y_m=0.0, z_m=0.0,
                speed_ms=50.0, direction_rad=0.0,
                heading_rad=0.0, ang_vel_rads=0.0),
        CompCar(node=10, lap=1, player_id=3, position=3, info=0,
                x_m=40.0, y_m=0.0, z_m=0.0,
                speed_ms=48.0, direction_rad=0.0,
                heading_rad=0.0, ang_vel_rads=0.0),
    ]
    ctx.update(InSimMCI(cars=cars))
    snap = traffic_snapshot(ctx)
    assert snap is not None
    assert snap.view_position == 2
    assert snap.car_ahead_plid == 2
    assert snap.gap_to_ahead_m == pytest.approx(30.0)
    assert snap.closing_speed_to_ahead_ms == pytest.approx(-5.0)
    assert snap.car_behind_plid == 3
    assert snap.gap_to_behind_m == pytest.approx(30.0)
    assert snap.blue_flag_for_view is True


# ---------------------------------------------------------------------------
# Lap summary
# ---------------------------------------------------------------------------


def test_build_lap_records_basic():
    ctx = RaceContext()
    ctx.view_player_id = 1
    ctx.players[1] = InSimNewPlayer(
        player_id=1, connection_id=0, player_type=0, flags=0,
        player_name="x", plate="", car_name="FOX", skin_name="",
        tyres=(2, 2, 3, 3),
        handicap_mass_kg=10, handicap_t_res=5, model=0, passengers=0,
        rear_wheel_adjust=0, front_wheel_adjust=0,
        set_flags=0, num_in_race=1, config=0, fuel_pct=80,
    )
    ctx.update(_make_split(1, 1, 30000))
    ctx.update(_make_split(1, 2, 60000))
    ctx.update(_make_lap(1, 90000, 90000, 160))   # 80%
    ctx.update(_make_split(1, 1, 29500))
    ctx.update(_make_lap(1, 88000, 178000, 140))  # 70%
    # OBH and HLV in lap 2.
    ctx.update(InSimObjectHit(
        player_id=1, closing_speed_ms=5.0, time_ms=180000,
        contact_direction_rad=0, contact_heading_rad=0,
        contact_speed_ms=30, contact_x_m=0, contact_y_m=0,
        map_x_m=0, map_y_m=0, map_z_m=0,
        object_index=1, flags=1))
    ctx.update(InSimHotLapValid(player_id=1, hlvc=HLVC_WALL,
                                time_ms=180100,
                                car_speed_ms=20.0, car_direction_rad=0,
                                car_heading_rad=0.0,
                                car_x_m=0.0, car_y_m=0.0))

    laps = build_lap_records(ctx)
    assert len(laps) == 2
    assert laps[0].lap_time_ms == 90000
    assert laps[0].split1_ms == 30000
    assert laps[0].split2_ms == 60000
    assert laps[0].fuel_pct_end == 80.0
    assert laps[0].fuel_pct_used is None       # first lap
    assert laps[0].tyre_compounds == (2, 2, 3, 3)
    assert laps[0].handicap_mass_kg == 10
    assert laps[0].valid is True
    assert laps[1].fuel_pct_used == pytest.approx(10.0)
    assert laps[1].valid is False
    assert laps[1].invalid_reason == "wall"
    assert laps[1].obh_count == 1


# ---------------------------------------------------------------------------
# Derived physics
# ---------------------------------------------------------------------------


def _synthetic_df() -> pd.DataFrame:
    n = 50
    t = np.arange(n) * 20  # 50 Hz
    speed = np.linspace(20, 30, n)
    df = pd.DataFrame({
        "time_ms": t,
        "car": "FOX",
        "speed_ms": speed,
        "ang_vel_z": np.full(n, 0.5),
        "input_steer": np.full(n, 0.05),
        "vel_x": speed,
        "vel_y": np.full(n, 0.5),
        "accel_x": np.full(n, -2.0),
        "accel_y": np.full(n, 5.0),
        "throttle": np.linspace(0.0, 1.0, n),
        "brake": np.linspace(1.0, 0.0, n),
        "steer_torque_nm": np.full(n, 8.0),
        "dash_lights": np.full(n, DL_TC | DL_ABS, dtype=int),
    })
    # 4 wheels with simple equal vertical loads.
    for c in ("FL", "FR", "RL", "RR"):
        df[f"wheel_{c}_vertical_load_n"] = 1500.0
        df[f"wheel_{c}_y_force_n"] = 400.0
        df[f"wheel_{c}_x_force_n"] = 600.0
        df[f"wheel_{c}_tan_slip_angle"] = 0.05
        df[f"wheel_{c}_slip_ratio"] = 0.02
    return df


def test_enrich_dataframe_adds_columns():
    df = _synthetic_df()
    spec = CarSpec()
    out = enrich_dataframe(df, spec)
    # Chassis dynamics
    for col in ("yaw_rate_theoretical_rads", "understeer_index",
                "beta_deg", "transfer_long_n_theoretical",
                "load_total_n", "load_front_frac",
                "transfer_long_n_real",
                "friction_use_FL", "friction_use_RR",
                "tyre_work_w_FL", "brake_bias_front_real",
                "ffb_load_pct", "steer_rate_rads",
                "steer_reversal_rate_hz", "throttle_rate_per_s",
                "overlap_brake_throttle",
                "dl_tc_active", "dl_abs_active"):
        assert col in out.columns, f"missing derived column: {col}"
    # TC and ABS lights decoded as True.
    assert bool(out["dl_tc_active"].iloc[0]) is True
    assert bool(out["dl_abs_active"].iloc[0]) is True
    assert bool(out["dl_pit_limiter"].iloc[0]) is False
    # Load total = 6000 N (4 × 1500) → front_frac = 0.5.
    assert out["load_total_n"].iloc[0] == pytest.approx(6000.0)
    assert out["load_front_frac"].iloc[0] == pytest.approx(0.5)
    # Beta in deg = atan2(0.5, ~25) > 0.
    assert out["beta_deg"].iloc[0] > 0
    # FFB = 8 / 25 = 0.32.
    assert out["ffb_load_pct"].iloc[0] == pytest.approx(0.32, rel=1e-3)
    # Friction use uses default mu_lat=1.40, mu_long=1.20:
    # fy_norm = 600 / (1.40·1500) ≈ 0.286; fx_norm = 400 / (1.20·1500) ≈ 0.222
    # → sqrt = 0.362.
    assert out["friction_use_FL"].iloc[0] == pytest.approx(0.362, rel=1e-2)


def test_enrich_dataframe_combined_channels():
    """Combined / synergy channels expose the right semantics."""
    df = _synthetic_df()
    # Add the raw inputs the combined block needs but the base fixture
    # doesn't have: suspension deflection, roll/pitch, and a lockup
    # scenario on the FL wheel.
    for c in ("FL", "FR", "RL", "RR"):
        df[f"wheel_{c}_susp_deflect_m"] = 0.030 if c.startswith("F") else 0.020
    df["roll"] = 0.05    # rad
    df["pitch"] = -0.02  # rad
    df.loc[0, "wheel_FL_slip_ratio"] = -0.5   # locked under braking
    df.loc[0, "brake"] = 0.8

    out = enrich_dataframe(df, CarSpec())

    # g_total = sqrt(2² + 5²) / 9.80665 ≈ 0.549
    assert out["g_total_g"].iloc[0] == pytest.approx(0.549, rel=1e-2)
    # Front compression = 0.030, rear = 0.020 → rake = +0.010 (nose down).
    assert out["susp_compression_front_avg_m"].iloc[0] == pytest.approx(0.030)
    assert out["susp_compression_rear_avg_m"].iloc[0] == pytest.approx(0.020)
    assert out["rake_compression_m"].iloc[0] == pytest.approx(0.010)
    # All wheels share |tan α|=0.05 ⇒ balance = 0.
    assert out["slip_angle_balance_rad"].iloc[0] == pytest.approx(0.0)
    # Lockup detected on FL only at row 0.
    assert bool(out["wheel_FL_lockup"].iloc[0]) is True
    assert bool(out["wheel_FR_lockup"].iloc[0]) is False
    # Brake power > 0 at row 0 (long. force is +400 N per wheel in the
    # fixture, which is *positive*; the brake-power channel clips
    # negative-net-force only, so net positive long. force yields 0 W).
    # Validate the channel exists and is non-negative everywhere.
    assert (out["brake_power_w"] >= 0).all()
    # Coasting flag: row with throttle=0 & brake>0 should be False.
    assert bool(out["coasting"].iloc[0]) is False
    # Trail-brake intensity at row 0: brake=0.8, |steer|=0.05 → 0.04.
    assert out["trail_brake_intensity"].iloc[0] == pytest.approx(0.04)
    # Throttle reversal rate is present.
    assert "throttle_reversal_rate_hz" in out.columns
    # Compliance ratios: ay=5 m/s²>2 ⇒ defined; ax=-2 m/s² not >2 ⇒ NaN.
    assert np.isfinite(out["chassis_roll_per_lat_g_rad_per_g"].iloc[0])
    assert np.isnan(out["chassis_pitch_per_long_g_rad_per_g"].iloc[0])


def test_enrich_dataframe_handles_missing_columns():
    df = pd.DataFrame({"time_ms": [0, 20, 40], "car": ["FOX"] * 3,
                       "speed_ms": [10, 11, 12]})
    out = enrich_dataframe(df)
    # Should not crash; some derived columns simply absent.
    assert "yaw_rate_theoretical_rads" not in out.columns  # missing input_steer
    assert "load_total_n" not in out.columns
