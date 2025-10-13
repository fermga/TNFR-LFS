"""Synthetic tests for the auto-organisation (THOL) detector."""

from __future__ import annotations

import pytest

from tnfr_core.operators.operator_detection import detect_thol

from tests.helpers import build_telemetry_record


_BASE_PAYLOAD = dict(
    vertical_load=5200.0,
    slip_ratio=0.0,
    lateral_accel=0.6,
    longitudinal_accel=0.2,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    brake_pressure=0.1,
    locking=0.0,
    nfr=140.0,
    si=0.4,
    speed=50.0,
    yaw_rate=0.05,
    slip_angle=0.01,
    steer=0.15,
    throttle=0.35,
    gear=5,
    vertical_load_front=2600.0,
    vertical_load_rear=2600.0,
    mu_eff_front=1.0,
    mu_eff_rear=1.0,
    suspension_velocity_front=0.0,
    suspension_velocity_rear=0.0,
    rpm=6400.0,
)


def _build_series(samples: list[dict]) -> list:
    series = []
    for index, payload in enumerate(samples):
        data = dict(_BASE_PAYLOAD)
        data.update(payload)
        data.setdefault("timestamp", index * 0.1)
        series.append(build_telemetry_record(**data))
    return series


def test_detect_thol_reports_suspension_activity_with_stable_heading() -> None:
    samples = [
        {
            "suspension_velocity_front": -0.02,
            "suspension_velocity_rear": 0.03,
            "steer": 0.14,
            "yaw_rate": 0.05,
        },
        {
            "suspension_velocity_front": -0.03,
            "suspension_velocity_rear": 0.04,
            "steer": 0.145,
            "yaw_rate": 0.052,
        },
        {
            "suspension_velocity_front": -0.01,
            "suspension_velocity_rear": 0.05,
            "steer": 0.147,
            "yaw_rate": 0.051,
        },
        {
            "suspension_velocity_front": -0.04,
            "suspension_velocity_rear": 0.06,
            "steer": 0.148,
            "yaw_rate": 0.049,
        },
        {
            "suspension_velocity_front": -0.03,
            "suspension_velocity_rear": 0.07,
            "steer": 0.146,
            "yaw_rate": 0.05,
        },
        {"steer": 0.2, "yaw_rate": 0.15},
    ]
    series = _build_series(samples)

    events = detect_thol(
        series,
        window=5,
        suspension_span_threshold=0.05,
        steer_span_threshold=0.02,
        yaw_rate_span_threshold=0.04,
    )

    assert events
    event = events[0]
    assert event["name"] == "Auto-organisation"
    assert event["suspension_activity"] >= 0.05
    assert event["steer_span"] <= 0.02
    assert event["yaw_rate_span"] <= 0.04


@pytest.mark.parametrize(
    "samples",
    [
        [
            {
                "suspension_velocity_front": -0.01,
                "suspension_velocity_rear": 0.02,
                "steer": 0.16,
                "yaw_rate": 0.08,
            }
            for _ in range(5)
        ],
        [
            {
                "suspension_velocity_front": -0.03,
                "suspension_velocity_rear": 0.05,
                "steer": 0.16 + 0.03 * index,
                "yaw_rate": 0.05 + 0.05 * index,
            }
            for index in range(5)
        ],
    ],
)
def test_detect_thol_rejects_large_heading_variations(samples: list[dict]) -> None:
    series = _build_series(samples)
    events = detect_thol(series, window=4, suspension_span_threshold=0.05, steer_span_threshold=0.02, yaw_rate_span_threshold=0.04)
    assert events == []
