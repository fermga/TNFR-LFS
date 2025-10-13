"""Synthetic tests for the amplification (VAL) detector."""

from __future__ import annotations

import pytest

from tnfr_core.operators.operator_detection import detect_val

from tests.helpers import build_telemetry_record


_BASE_PAYLOAD = dict(
    vertical_load=5000.0,
    slip_ratio=0.0,
    lateral_accel=0.0,
    longitudinal_accel=0.0,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    brake_pressure=0.1,
    locking=0.0,
    nfr=120.0,
    si=0.4,
    speed=40.0,
    yaw_rate=0.05,
    slip_angle=0.02,
    steer=0.12,
    throttle=0.3,
    gear=4,
    vertical_load_front=2600.0,
    vertical_load_rear=2400.0,
    mu_eff_front=1.05,
    mu_eff_rear=1.0,
    suspension_velocity_front=0.015,
    suspension_velocity_rear=0.01,
    rpm=6200.0,
)


def _build_series(samples: list[dict]) -> list:
    series = []
    for index, payload in enumerate(samples):
        data = dict(_BASE_PAYLOAD)
        data.update(payload)
        data.setdefault("timestamp", index * 0.1)
        series.append(build_telemetry_record(**data))
    return series


def test_detect_val_emits_event_on_high_lateral_and_throttle() -> None:
    samples = [
        {"lateral_accel": 0.8, "throttle": 0.45, "vertical_load": 5200.0},
        {"lateral_accel": 1.2, "throttle": 0.55, "vertical_load": 5400.0},
        {"lateral_accel": 2.1, "throttle": 0.62, "vertical_load": 5900.0},
        {"lateral_accel": 2.4, "throttle": 0.68, "vertical_load": 6100.0},
        {"lateral_accel": 2.5, "throttle": 0.7, "vertical_load": 6350.0},
        {"lateral_accel": 1.8, "throttle": 0.6, "vertical_load": 6000.0},
        {"lateral_accel": 1.0, "throttle": 0.35, "vertical_load": 5400.0},
    ]
    series = _build_series(samples)

    events = detect_val(
        series,
        window=4,
        lateral_threshold=1.8,
        throttle_threshold=0.55,
        load_span_threshold=500.0,
    )

    assert events
    event = events[0]
    assert event["name"] == "Amplification"
    assert event["lateral_peak"] >= 1.8
    assert event["throttle_peak"] >= 0.55
    assert event["load_span"] >= 500.0


@pytest.mark.parametrize(
    "samples",
    [
        [{"lateral_accel": 0.5, "throttle": 0.3, "vertical_load": 5200.0} for _ in range(5)],
        [
            {"lateral_accel": 2.0, "throttle": 0.4, "vertical_load": 5400.0},
            {"lateral_accel": 2.1, "throttle": 0.45, "vertical_load": 5450.0},
            {"lateral_accel": 2.2, "throttle": 0.47, "vertical_load": 5500.0},
            {"lateral_accel": 2.3, "throttle": 0.48, "vertical_load": 5550.0},
        ],
    ],
)
def test_detect_val_requires_all_thresholds(samples: list[dict]) -> None:
    series = _build_series(samples)
    events = detect_val(series, window=3, lateral_threshold=1.8, throttle_threshold=0.55, load_span_threshold=500.0)
    assert events == []
