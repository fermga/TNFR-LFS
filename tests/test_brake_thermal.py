from __future__ import annotations

import math

import pytest

from tnfr_lfs.analysis.brake_thermal import BrakeThermalConfig, BrakeThermalEstimator
from tnfr_lfs.acquisition.fusion import TelemetryFusion
from tnfr_lfs.acquisition.outgauge_udp import OutGaugePacket
from tnfr_lfs.acquisition.outsim_udp import (
    OutSimDriverInputs,
    OutSimPacket,
    OutSimWheelState,
)


def test_brake_thermal_estimator_heats_and_cools() -> None:
    config = BrakeThermalConfig(
        ambient=30.0,
        heat_capacity=500.0,
        heating_efficiency=1.0,
        convective_coefficient=0.1,
        still_air_coefficient=0.02,
        minimum_brake_input=0.0,
    )
    estimator = BrakeThermalEstimator(config)

    initial = estimator.temperatures
    heated = estimator.step(
        dt=1.0,
        speed=40.0,
        deceleration=5.0,
        brake_input=1.0,
        wheel_loads=(4000.0, 4000.0, 4000.0, 4000.0),
    )
    assert all(temp > base for temp, base in zip(heated, initial))

    cooled = estimator.step(
        dt=2.0,
        speed=25.0,
        deceleration=0.0,
        brake_input=0.0,
        wheel_loads=(4000.0, 4000.0, 4000.0, 4000.0),
    )
    assert all(temp <= prev for temp, prev in zip(cooled, heated))
    assert all(temp >= config.ambient for temp in cooled)


def _make_outsim_packet(
    timestamp_ms: int,
    speed: float,
    accel_x: float,
    brake: float,
    load: float,
) -> OutSimPacket:
    wheels = tuple(
        OutSimWheelState(
            slip_ratio=0.0,
            slip_angle=0.0,
            longitudinal_force=0.0,
            lateral_force=0.0,
            load=load,
            suspension_deflection=0.0,
            decoded=True,
        )
        for _ in range(4)
    )
    inputs = OutSimDriverInputs(throttle=0.0, brake=brake, clutch=0.0, handbrake=0.0, steer=0.0)
    return OutSimPacket(
        time=timestamp_ms,
        ang_vel_x=0.0,
        ang_vel_y=0.0,
        ang_vel_z=0.0,
        heading=0.0,
        pitch=0.0,
        roll=0.0,
        accel_x=accel_x,
        accel_y=0.0,
        accel_z=9.81,
        vel_x=speed,
        vel_y=0.0,
        vel_z=0.0,
        pos_x=0.0,
        pos_y=0.0,
        pos_z=0.0,
        inputs=inputs,
        wheels=wheels,
    )


def _make_outgauge_packet(timestamp_ms: int, speed: float) -> OutGaugePacket:
    return OutGaugePacket(
        time=timestamp_ms,
        car="FZR",
        player_name="",
        plate="",
        track="AS5",
        layout="",
        flags=0,
        gear=3,
        plid=0,
        speed=speed,
        rpm=4500.0,
        turbo=0.0,
        eng_temp=90.0,
        fuel=0.5,
        oil_pressure=0.0,
        oil_temp=80.0,
        dash_lights=0,
        show_lights=0,
        throttle=0.0,
        brake=1.0,
        clutch=0.0,
        display1="",
        display2="",
        packet_id=0,
        brake_temps=(0.0, 0.0, 0.0, 0.0),
    )


def test_fusion_brake_proxy_fills_missing_outgauge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TNFR_LFS_BRAKE_THERMAL", raising=False)

    fusion = TelemetryFusion()

    first_outsim = _make_outsim_packet(0, speed=25.0, accel_x=-1.0, brake=1.0, load=3800.0)
    first_outgauge = _make_outgauge_packet(0, speed=25.0)
    fusion.fuse(first_outsim, first_outgauge)

    second_outsim = _make_outsim_packet(100, speed=35.0, accel_x=-5.0, brake=1.0, load=4200.0)
    second_outgauge = _make_outgauge_packet(100, speed=35.0)
    record = fusion.fuse(second_outsim, second_outgauge)

    assert math.isfinite(record.brake_temp_fl)
    assert math.isfinite(record.brake_temp_fr)
    assert math.isfinite(record.brake_temp_rl)
    assert math.isfinite(record.brake_temp_rr)
    assert record.brake_temp_fl >= fusion._brake_thermal_defaults.ambient
