"""Unit tests for the helper methods introduced in :mod:`fusion`."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from tnfr_lfs.ingestion.live import (
    FusionCalibration,
    OutSimWheelState,
    TelemetryFusion,
    _WheelTelemetry,
)


def _simple_outsim(*wheels: OutSimWheelState) -> SimpleNamespace:
    return SimpleNamespace(wheels=wheels)


def test_preprocess_wheels_handles_partial_payload() -> None:
    fusion = TelemetryFusion()
    wheels = (
        OutSimWheelState(
            slip_ratio=2.0,
            slip_angle=0.1,
            lateral_force=5.0,
            longitudinal_force=10.0,
            load=250.0,
            suspension_deflection=0.02,
            decoded=True,
        ),
    )
    data = fusion._preprocess_wheels(_simple_outsim(*wheels))

    assert len(data.wheels) == 4
    assert math.isclose(data.slip_ratios[0], 1.0)
    assert math.isnan(data.slip_ratios[1])
    assert data.data_present is True


def test_aggregate_wheel_slip_weights_by_load() -> None:
    fusion = TelemetryFusion()
    wheels = (
        OutSimWheelState(slip_ratio=0.1, slip_angle=0.1, load=200.0, decoded=True),
        OutSimWheelState(slip_ratio=0.2, slip_angle=0.4, load=100.0, decoded=True),
        OutSimWheelState(slip_ratio=-0.1, slip_angle=-0.2, load=100.0, decoded=True),
        OutSimWheelState(slip_ratio=0.0, slip_angle=0.0, load=100.0, decoded=True),
    )
    data = fusion._preprocess_wheels(_simple_outsim(*wheels))

    ratio, angle = fusion._aggregate_wheel_slip(data)

    expected_ratio = sum(w.slip_ratio for w in wheels) / len(wheels)
    expected_angle = (
        (0.1 * 200.0) + (0.4 * 100.0) + (-0.2 * 100.0) + (0.0 * 100.0)
    ) / (200.0 + 100.0 + 100.0 + 100.0)

    assert math.isclose(ratio, expected_ratio)
    assert math.isclose(angle, expected_angle)


def test_aggregate_wheel_slip_handles_missing_data() -> None:
    fusion = TelemetryFusion()
    data = fusion._preprocess_wheels(
        _simple_outsim(*(OutSimWheelState() for _ in range(4)))
    )

    ratio, angle = fusion._aggregate_wheel_slip(data)

    assert math.isnan(ratio)
    assert math.isnan(angle)


def test_calculate_suspension_with_wheel_data() -> None:
    fusion = TelemetryFusion()
    wheels = (
        OutSimWheelState(suspension_deflection=0.05, decoded=True),
        OutSimWheelState(suspension_deflection=0.07, decoded=True),
        OutSimWheelState(suspension_deflection=0.08, decoded=True),
        OutSimWheelState(suspension_deflection=0.10, decoded=True),
    )
    data = fusion._preprocess_wheels(_simple_outsim(*wheels))
    previous = SimpleNamespace(
        suspension_travel_front=0.04, suspension_travel_rear=0.09
    )

    travel_front, travel_rear, vel_front, vel_rear = fusion._calculate_suspension(
        data,
        0.5,
        0.5,
        previous,
        0.1,
        FusionCalibration(),
    )

    expected_front = (0.05 + 0.07) / 2.0
    expected_rear = (0.08 + 0.10) / 2.0

    assert math.isclose(travel_front, expected_front)
    assert math.isclose(travel_rear, expected_rear)
    assert math.isclose(vel_front, (expected_front - 0.04) / 0.1)
    assert math.isclose(vel_rear, (expected_rear - 0.09) / 0.1)


def test_calculate_suspension_without_wheel_data() -> None:
    fusion = TelemetryFusion()
    data = _WheelTelemetry(
        wheels=tuple(OutSimWheelState() for _ in range(4)),
        slip_ratios=tuple(math.nan for _ in range(4)),
        slip_angles=tuple(math.nan for _ in range(4)),
        lateral_forces=tuple(math.nan for _ in range(4)),
        longitudinal_forces=tuple(math.nan for _ in range(4)),
        loads=tuple(math.nan for _ in range(4)),
        deflections=tuple(math.nan for _ in range(4)),
        data_present=False,
    )
    previous = SimpleNamespace(
        suspension_travel_front=0.1, suspension_travel_rear=0.1
    )

    travel_front, travel_rear, vel_front, vel_rear = fusion._calculate_suspension(
        data,
        0.6,
        0.4,
        previous,
        0.1,
        FusionCalibration(),
    )

    assert math.isnan(travel_front)
    assert math.isnan(travel_rear)
    assert math.isnan(vel_front)
    assert math.isnan(vel_rear)


def test_estimate_thermal_delegates_to_resolvers(monkeypatch: pytest.MonkeyPatch) -> None:
    fusion = TelemetryFusion()
    sentinel_layers = ((1.0, 2.0, 3.0, 4.0),) * 3
    sentinel_temps = (90.0, 91.0, 92.0, 93.0)
    sentinel_pressures = (20.0, 21.0, 22.0, 23.0)
    sentinel_brakes = (300.0, 301.0, 302.0, 303.0)

    monkeypatch.setattr(
        fusion,
        "_resolve_wheel_temperature_layers",
        lambda outgauge, previous: sentinel_layers,
    )
    monkeypatch.setattr(
        fusion,
        "_resolve_wheel_temperatures",
        lambda outgauge, previous, layers=None: sentinel_temps,
    )
    monkeypatch.setattr(
        fusion,
        "_resolve_wheel_pressures",
        lambda outgauge, previous: sentinel_pressures,
    )

    recorded_args: dict[str, tuple[object, ...]] = {}

    def fake_brakes(*args: object) -> tuple[float, float, float, float]:
        recorded_args["args"] = args
        return sentinel_brakes

    monkeypatch.setattr(fusion, "_resolve_brake_temperatures", fake_brakes)

    gauge_stub = SimpleNamespace()

    result = fusion._estimate_thermal(
        gauge_stub,
        None,
        0.1,
        30.0,
        2.0,
        0.5,
        (1.0, 2.0, 3.0, 4.0),
    )

    assert result == (
        sentinel_layers,
        sentinel_temps,
        sentinel_pressures,
        sentinel_brakes,
    )
    assert recorded_args["args"] == (
        0.1,
        30.0,
        2.0,
        0.5,
        (1.0, 2.0, 3.0, 4.0),
        gauge_stub,
        None,
    )


def test_resolve_wheel_temperatures_prefers_layers_when_missing_primary() -> None:
    fusion = TelemetryFusion()
    layers = (
        (80.0, 0.0, 82.0, float("nan")),
        (82.0, 84.0, 86.0, 88.0),
        (84.0, 86.0, float("nan"), 90.0),
    )
    previous = SimpleNamespace(
        tyre_temp_fl=70.0,
        tyre_temp_fr=72.0,
        tyre_temp_rl=74.0,
        tyre_temp_rr=76.0,
    )

    result = fusion._resolve_wheel_temperatures(
        SimpleNamespace(tyre_temps=(0.0, None, -5.0, float("nan"))),
        previous,
        layers=layers,
    )

    assert result == pytest.approx((82.0, 85.0, 84.0, 89.0))


def test_resolve_wheel_temperatures_uses_previous_when_layers_invalid() -> None:
    fusion = TelemetryFusion()
    layers = (
        (0.0, 0.0, 0.0, 0.0),
        (float("nan"),) * 4,
        (0.0, 0.0, 0.0, 0.0),
    )
    previous = SimpleNamespace(
        tyre_temp_fl=70.0,
        tyre_temp_fr=72.0,
        tyre_temp_rl=74.0,
        tyre_temp_rr=76.0,
    )

    result = fusion._resolve_wheel_temperatures(
        SimpleNamespace(tyre_temps=(0.0, 0.0, 0.0, 0.0)),
        previous,
        layers=layers,
    )

    assert result == pytest.approx((70.0, 72.0, 74.0, 76.0))


def test_construct_record_populates_fields() -> None:
    fusion = TelemetryFusion()
    wheels = (
        OutSimWheelState(
            slip_ratio=0.1,
            slip_angle=0.2,
            load=150.0,
            lateral_force=5.0,
            longitudinal_force=6.0,
            suspension_deflection=0.03,
            decoded=True,
        ),
    ) * 4
    wheel_data = fusion._preprocess_wheels(_simple_outsim(*wheels))

    record = fusion._construct_record(
        timestamp=1.23,
        outsim=SimpleNamespace(
            accel_y=1.0,
            accel_x=-0.5,
            pitch=0.1,
            roll=0.2,
        ),
        outgauge=SimpleNamespace(
            brake=0.3,
            throttle=0.4,
            gear=3,
            rpm=4500.0,
            dash_lights=0,
        ),
        wheel_data=wheel_data,
        slip_ratio=0.12,
        slip_angle=0.34,
        throttle=0.56,
        brake_input=0.78,
        clutch_input=0.11,
        handbrake_input=0.22,
        steer_input=0.33,
        speed=40.0,
        yaw=0.5,
        yaw_rate=0.6,
        steer=0.7,
        vertical_load=600.0,
        front_share=0.5,
        rear_share=0.5,
        front_load=300.0,
        rear_load=300.0,
        mu_front=1.1,
        mu_front_lat=1.2,
        mu_front_long=1.3,
        mu_rear=1.4,
        mu_rear_lat=1.5,
        mu_rear_long=1.6,
        suspension_travel_front=0.04,
        suspension_travel_rear=0.05,
        suspension_velocity_front=0.06,
        suspension_velocity_rear=0.07,
        tyre_temp_layers=((80.0, 81.0, 82.0, 83.0),) * 3,
        tyre_temps=(90.0, 91.0, 92.0, 93.0),
        tyre_pressures=(20.0, 21.0, 22.0, 23.0),
        brake_temps=(300.0, 301.0, 302.0, 303.0),
        line_deviation=1.5,
    )

    assert math.isclose(record.timestamp, 1.23)
    assert math.isclose(record.vertical_load, 600.0)
    assert math.isclose(record.brake_input, 0.78)
    assert math.isclose(record.tyre_temp_fl_inner, 80.0)
    assert math.isclose(record.tyre_pressure_rr, 23.0)
    assert math.isclose(record.wheel_load_fl, wheel_data.loads[0])
    assert math.isclose(record.slip_ratio_fr, wheel_data.slip_ratios[1])
    assert math.isclose(record.yaw, 0.5)

