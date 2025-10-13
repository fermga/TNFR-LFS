"""Synthetic tests for the contraction (NUL) detector."""

from __future__ import annotations

import pytest

from tnfr_core.operators.operator_detection import detect_nul

from tests.helpers import build_telemetry_record


_BASE_PAYLOAD = dict(
    vertical_load=5200.0,
    slip_ratio=0.0,
    lateral_accel=0.6,
    longitudinal_accel=-0.2,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    brake_pressure=0.2,
    locking=0.0,
    nfr=150.0,
    si=0.35,
    speed=45.0,
    yaw_rate=0.02,
    slip_angle=0.01,
    steer=0.08,
    throttle=0.2,
    gear=4,
    vertical_load_front=2700.0,
    vertical_load_rear=2500.0,
    mu_eff_front=1.0,
    mu_eff_rear=1.0,
    suspension_velocity_front=0.01,
    suspension_velocity_rear=0.01,
    rpm=5800.0,
)


def _build_series(samples: list[dict]) -> list:
    series = []
    for index, payload in enumerate(samples):
        data = dict(_BASE_PAYLOAD)
        data.update(payload)
        data.setdefault("timestamp", index * 0.1)
        series.append(build_telemetry_record(**data))
    return series


def test_detect_nul_detects_braking_contraction() -> None:
    samples = [
        {"longitudinal_accel": -0.5, "speed": 48.0, "brake_pressure": 0.3},
        {"longitudinal_accel": -1.2, "speed": 42.0, "brake_pressure": 0.55},
        {"longitudinal_accel": -2.0, "speed": 36.0, "brake_pressure": 0.6},
        {"longitudinal_accel": -2.4, "speed": 31.0, "brake_pressure": 0.62},
        {"longitudinal_accel": -1.8, "speed": 29.0, "brake_pressure": 0.58},
        {"longitudinal_accel": -0.4, "speed": 35.0, "brake_pressure": 0.4},
    ]
    series = _build_series(samples)

    events = detect_nul(
        series,
        window=4,
        decel_threshold=1.5,
        speed_drop_threshold=10.0,
        brake_pressure_threshold=0.5,
    )

    assert events
    event = events[0]
    assert event["name"] == "Contraction"
    assert event["decel_peak"] >= 1.5
    assert event["speed_drop"] >= 10.0
    assert event["brake_pressure_mean"] >= 0.5


@pytest.mark.parametrize(
    "samples",
    [
        [{"longitudinal_accel": -0.6, "speed": 40.0, "brake_pressure": 0.3} for _ in range(5)],
        [
            {"longitudinal_accel": -2.2, "speed": 45.0, "brake_pressure": 0.4},
            {"longitudinal_accel": -2.4, "speed": 43.0, "brake_pressure": 0.42},
            {"longitudinal_accel": -2.3, "speed": 42.0, "brake_pressure": 0.43},
            {"longitudinal_accel": -2.1, "speed": 41.0, "brake_pressure": 0.41},
        ],
    ],
)
def test_detect_nul_requires_speed_drop_and_pressure(samples: list[dict]) -> None:
    series = _build_series(samples)
    events = detect_nul(series, window=3, decel_threshold=1.5, speed_drop_threshold=10.0, brake_pressure_threshold=0.5)
    assert events == []
