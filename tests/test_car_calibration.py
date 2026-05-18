"""Tests for the per-car auto-calibration store (no LFS required)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from lfs_telemetry.telemetry.car_calibration import (
    CarCalibration,
    CarSpecStore,
    RestCalibrator,
)
from lfs_telemetry.telemetry.constants import GRAVITY
from lfs_telemetry.telemetry.live import TelemetrySample, _outsim2_to_basic
from lfs_telemetry.telemetry.protocol.packets import OSO_ALL, OutGaugePacket, OutSimPack2


def _build_outsim2(
    time_ms: int,
    *,
    loads_n: tuple[float, float, float, float],  # native LFS order: RL,RR,FL,FR
    ax: float = 0.0,
    ay: float = 0.0,
    touching: int = 1,
) -> bytes:
    parts: list[bytes] = []
    parts.append(b"LFST")
    parts.append(struct.pack("<i", 1))
    parts.append(struct.pack("<I", time_ms))
    # MAIN: 12 floats + 3 ints. Layout: ang_vel(3), heading,pitch,roll,
    # accel(3), vel(3), pos_xyz_int(3). Index of accel: floats 6,7,8.
    parts.append(struct.pack(
        "<12f3i",
        0.0, 0.0, 0.0,        # ang_vel
        0.0, 0.0, 0.0,        # heading,pitch,roll
        ax, ay, GRAVITY,      # accel
        0.0, 0.0, 0.0,        # velocity
        0, 0, 0,
    ))
    parts.append(struct.pack("<5f", 0.0, 0.0, 0.0, 0.0, 0.0))   # INPUTS
    parts.append(struct.pack("<4B2f", 0, 0, 0, 0, 800.0, 250.0))  # DRIVE
    parts.append(struct.pack("<2f", 0.0, 0.0))                  # DISTANCE
    for load in loads_n:                                         # WHEELS x4
        parts.append(struct.pack(
            "<7f4B2f",
            0.0, 0.0, 0.0, 0.0,
            float(load),
            0.0, 0.0,
            20, 0, touching, 0,
            0.0, 0.0,
        ))
    parts.append(struct.pack("<2f", 0.0, 0.0))                  # EXTRA_1
    return b"".join(parts)


def _make_rest_sample(
    time_ms: int,
    *,
    car: str = "FBM",
    loads_n: tuple[float, float, float, float] = (1500.0, 1500.0, 1200.0, 1200.0),
    speed_ms: float = 0.0,
    throttle: float = 0.0,
    brake: float = 0.0,
) -> TelemetrySample:
    raw = _build_outsim2(time_ms, loads_n=loads_n)
    pkt2 = OutSimPack2.parse(raw, OSO_ALL)
    basic = _outsim2_to_basic(pkt2)
    assert basic is not None
    og = OutGaugePacket(
        time_ms=time_ms, car=car, flags=0, gear=1, player_id=0,
        speed_ms=speed_ms, rpm=1500.0, turbo_bar=0.0,
        eng_temp_c=80.0, fuel=0.5, oil_pressure_bar=3.0,
        oil_temp_c=80.0, dash_lights=0, show_lights=0,
        throttle=throttle, brake=brake, clutch=1.0,
        display1="", display2="", packet_id=0,
    )
    return TelemetrySample(time_ms=time_ms, outsim=basic, outgauge=og,
                           outsim2=pkt2, race_context=None)


def test_calibrator_emits_after_window_at_rest() -> None:
    cal = RestCalibrator(window=100)
    # Native LFS order: RL,RR,FL,FR  →  rear=1500 each, front=1200 each.
    loads = (1500.0, 1500.0, 1200.0, 1200.0)
    result = None
    for i in range(120):
        s = _make_rest_sample(i * 10, loads_n=loads)
        result = cal.feed(s)
        if result is not None:
            break
    assert result is not None
    assert result.car_id == "FBM"
    total = sum(loads)                                      # 5400 N
    assert result.sum_load_n == pytest.approx(total, rel=1e-6)
    assert result.mass_kg == pytest.approx(total / GRAVITY, rel=1e-6)
    # front fraction = (FL+FR)/total = 2400/5400
    assert result.weight_dist_front == pytest.approx(2400.0 / 5400.0, rel=1e-6)


def test_calibrator_does_not_emit_when_moving() -> None:
    cal = RestCalibrator(window=50)
    for i in range(80):
        s = _make_rest_sample(i * 10, speed_ms=5.0)
        assert cal.feed(s) is None


def test_calibrator_does_not_emit_when_throttled() -> None:
    cal = RestCalibrator(window=50)
    for i in range(80):
        s = _make_rest_sample(i * 10, throttle=0.5)
        assert cal.feed(s) is None


def test_store_roundtrip_persists_calibration(tmp_path: Path) -> None:
    p = tmp_path / "cars.json"
    store = CarSpecStore(p)
    cal = CarCalibration(
        car_id="MOD1", mass_kg=720.5, weight_dist_front=0.48,
        sample_count=100, sum_load_n=720.5 * GRAVITY,
        front_fraction=0.48, left_fraction=0.501,
    )
    store.put(cal)
    store.save()
    assert p.exists()

    store2 = CarSpecStore(p)
    got = store2.get("MOD1")
    assert got is not None
    assert got.mass_kg == pytest.approx(720.5)
    assert got.weight_dist_front == pytest.approx(0.48)


def test_spec_for_uses_calibration_over_defaults(tmp_path: Path) -> None:
    p = tmp_path / "cars.json"
    store = CarSpecStore(p)
    cal = CarCalibration(
        car_id="FBM", mass_kg=999.0, weight_dist_front=0.5,
        sample_count=100, sum_load_n=999.0 * GRAVITY,
        front_fraction=0.5, left_fraction=0.5,
    )
    store.put(cal)
    spec = store.spec_for("FBM")
    assert spec.mass_kg == pytest.approx(999.0)
    assert spec.weight_dist_front == pytest.approx(0.5)
    # Geometry comes from bundled FBM defaults
    assert spec.wheelbase_m == pytest.approx(2.59)


def test_spec_for_unknown_mod_uses_generic_geometry(tmp_path: Path) -> None:
    p = tmp_path / "cars.json"
    store = CarSpecStore(p)
    cal = CarCalibration(
        car_id="X1ABCD", mass_kg=820.0, weight_dist_front=0.55,
        sample_count=100, sum_load_n=820.0 * GRAVITY,
        front_fraction=0.55, left_fraction=0.5,
    )
    store.put(cal)
    spec = store.spec_for("X1ABCD")
    assert spec.mass_kg == pytest.approx(820.0)
    assert spec.weight_dist_front == pytest.approx(0.55)
    # Generic fallback geometry (Formula-class)
    assert spec.driven == "RWD"
    assert spec.wheelbase_m > 0
