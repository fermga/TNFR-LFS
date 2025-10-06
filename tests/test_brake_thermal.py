import math
from dataclasses import replace

from tnfr_lfs.analysis.brake_thermal import BrakeThermalConfig, BrakeThermalEstimator
from tnfr_lfs.acquisition.fusion import TelemetryFusion
from tnfr_lfs.acquisition.outgauge_udp import OutGaugePacket
from tnfr_lfs.acquisition.outsim_udp import OutSimPacket, OutSimWheelState
from tnfr_lfs.core.metrics import compute_window_metrics


def test_brake_proxy_heats_and_cools() -> None:
    estimator = BrakeThermalEstimator(BrakeThermalConfig(ambient_C=20.0))

    temps = None
    for _ in range(50):
        temps = estimator.update(0.1, 50.0, 7.0, 0.9, (2500.0, 2500.0, 1500.0, 1500.0))
    assert temps is not None
    assert all(temp > 20.0 for temp in temps)
    peak = max(temps)

    for _ in range(100):
        temps = estimator.update(0.1, 80.0, 0.0, 0.0, (2000.0, 2000.0, 2000.0, 2000.0))
    assert temps is not None
    assert max(temps) < peak
    assert all(temp >= 20.0 for temp in temps)


def test_brake_proxy_respects_dt_zero() -> None:
    estimator = BrakeThermalEstimator()
    initial = estimator.update(0.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0, 0.0))
    assert all(math.isclose(value, estimator.cfg.ambient_C) for value in initial)


def _make_outgauge(brake: float) -> OutGaugePacket:
    base = OutGaugePacket(
        time=0,
        car="FZR",
        player_name="",
        plate="",
        track="AS5",
        layout="",
        flags=0,
        gear=3,
        plid=0,
        speed=0.0,
        rpm=6000.0,
        turbo=0.0,
        eng_temp=0.0,
        fuel=0.0,
        oil_pressure=0.0,
        oil_temp=0.0,
        dash_lights=0,
        show_lights=0,
        throttle=0.0,
        brake=brake,
        clutch=0.0,
        display1="",
        display2="",
        packet_id=0,
    )
    return base


def _make_outsim(time_ms: int, accel_x: float) -> OutSimPacket:
    wheel_front = OutSimWheelState(load=2500.0, decoded=True)
    wheel_rear = OutSimWheelState(load=1500.0, decoded=True)
    wheels = (wheel_front, wheel_front, wheel_rear, wheel_rear)
    inputs = None
    return OutSimPacket(
        time=time_ms,
        ang_vel_x=0.0,
        ang_vel_y=0.0,
        ang_vel_z=0.0,
        heading=0.0,
        pitch=0.0,
        roll=0.0,
        accel_x=accel_x,
        accel_y=0.0,
        accel_z=-9.81,
        vel_x=50.0,
        vel_y=0.0,
        vel_z=0.0,
        pos_x=float(time_ms) * 0.01,
        pos_y=0.0,
        pos_z=0.0,
        inputs=inputs,
        wheels=wheels,
    )


def test_fusion_brake_proxy_produces_metrics() -> None:
    fusion = TelemetryFusion()
    outgauge = _make_outgauge(0.0)

    # Prime fusion with an initial record.
    fusion.fuse(_make_outsim(0, 0.0), outgauge)

    # Apply sustained braking samples without OutGauge brake temperatures.
    for index in range(1, 40):
        timestamp_ms = index * 100
        braking_outsim = _make_outsim(timestamp_ms, -7.0)
        braking_outgauge = replace(outgauge, brake=0.9)
        record = fusion.fuse(braking_outsim, braking_outgauge)

    assert math.isfinite(record.brake_temp_fl)
    assert fusion._brake_thermal is not None
    assert record.brake_temp_fl > fusion._brake_thermal.cfg.ambient_C

    window_metrics = compute_window_metrics(fusion._records)
    assert window_metrics.brake_headroom.temperature_available
    assert math.isfinite(window_metrics.brake_headroom.temperature_mean)
