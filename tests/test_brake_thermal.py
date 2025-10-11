from __future__ import annotations

import importlib
import math
from pathlib import Path

import pytest

from tnfr_lfs.analysis.brake_thermal import BrakeThermalConfig, BrakeThermalEstimator
from tnfr_lfs.ingestion.live import (
    OutSimDriverInputs,
    OutSimWheelState,
    TelemetryFusion,
)

from tests.helpers import build_outgauge_packet, build_outsim_packet, create_brake_thermal_pack


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


def test_telemetry_fusion_uses_packaged_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tnfr_lfs import _pack_resources

    pack_root = create_brake_thermal_pack(tmp_path / "pack")

    _pack_resources.set_pack_root_override(pack_root)
    ingestion_mod = importlib.import_module("tnfr_lfs.ingestion.live")
    fusion_mod = importlib.import_module(ingestion_mod.TelemetryFusion.__module__)
    importlib.reload(fusion_mod)
    ingestion_module = importlib.reload(ingestion_mod)

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    monkeypatch.chdir(workspace)

    try:
        ingestion_module.TelemetryFusion()  # should locate packaged resources without cwd data
    finally:
        _pack_resources.set_pack_root_override(None)
        importlib.reload(fusion_mod)
def test_fusion_brake_proxy_fills_missing_outgauge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TNFR_LFS_BRAKE_THERMAL", raising=False)

    fusion = TelemetryFusion()

    first_inputs = OutSimDriverInputs(throttle=0.0, brake=1.0, clutch=0.0, handbrake=0.0, steer=0.0)
    first_wheels = tuple(
        OutSimWheelState(
            slip_ratio=0.0,
            slip_angle=0.0,
            longitudinal_force=0.0,
            lateral_force=0.0,
            load=3800.0,
            suspension_deflection=0.0,
            decoded=True,
        )
        for _ in range(4)
    )
    first_outsim = build_outsim_packet(
        time=0,
        accel_x=-1.0,
        accel_z=9.81,
        vel_x=25.0,
        inputs=first_inputs,
        wheels=first_wheels,
    )
    first_outgauge = build_outgauge_packet(
        time=0,
        car="FZR",
        player_name="",
        plate="",
        track="AS5",
        layout="",
        gear=3,
        speed=25.0,
        rpm=4500.0,
        eng_temp=90.0,
        fuel=0.5,
        oil_temp=80.0,
        throttle=0.0,
        brake=1.0,
        clutch=0.0,
        brake_temps=(0.0, 0.0, 0.0, 0.0),
    )
    fusion.fuse(first_outsim, first_outgauge)

    second_inputs = OutSimDriverInputs(throttle=0.0, brake=1.0, clutch=0.0, handbrake=0.0, steer=0.0)
    second_wheels = tuple(
        OutSimWheelState(
            slip_ratio=0.0,
            slip_angle=0.0,
            longitudinal_force=0.0,
            lateral_force=0.0,
            load=4200.0,
            suspension_deflection=0.0,
            decoded=True,
        )
        for _ in range(4)
    )
    second_outsim = build_outsim_packet(
        time=100,
        accel_x=-5.0,
        accel_z=9.81,
        vel_x=35.0,
        inputs=second_inputs,
        wheels=second_wheels,
    )
    second_outgauge = build_outgauge_packet(
        time=100,
        car="FZR",
        player_name="",
        plate="",
        track="AS5",
        layout="",
        gear=3,
        speed=35.0,
        rpm=4500.0,
        eng_temp=90.0,
        fuel=0.5,
        oil_temp=80.0,
        throttle=0.0,
        brake=1.0,
        clutch=0.0,
        brake_temps=(0.0, 0.0, 0.0, 0.0),
    )
    record = fusion.fuse(second_outsim, second_outgauge)

    assert math.isfinite(record.brake_temp_fl)
    assert math.isfinite(record.brake_temp_fr)
    assert math.isfinite(record.brake_temp_rl)
    assert math.isfinite(record.brake_temp_rr)
    assert record.brake_temp_fl >= fusion._brake_thermal_defaults.ambient
