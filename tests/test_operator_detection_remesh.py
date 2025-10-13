"""Synthetic tests for the remeshing (REMESH) detector."""

from __future__ import annotations

import math

import pytest

from tnfr_core.operators.operator_detection import detect_remesh

from tests.helpers import build_telemetry_record


_BASE_PAYLOAD = dict(
    vertical_load=5050.0,
    slip_ratio=0.0,
    lateral_accel=0.4,
    longitudinal_accel=0.2,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    brake_pressure=0.12,
    locking=0.0,
    nfr=90.0,
    si=0.3,
    speed=38.0,
    yaw_rate=0.04,
    slip_angle=0.02,
    steer=0.1,
    throttle=0.4,
    gear=3,
    vertical_load_front=2500.0,
    vertical_load_rear=2550.0,
    mu_eff_front=1.0,
    mu_eff_rear=1.0,
    suspension_velocity_front=0.01,
    suspension_velocity_rear=0.02,
    rpm=5600.0,
    line_deviation=0.1,
    structural_timestamp=0.0,
)


def _build_series(samples: list[dict]) -> list:
    series = []
    structural = 0.0
    for index, payload in enumerate(samples):
        data = dict(_BASE_PAYLOAD)
        data.update(payload)
        data.setdefault("timestamp", index * 0.1)
        structural = data.get("structural_timestamp", structural + 0.1)
        data["structural_timestamp"] = structural
        series.append(build_telemetry_record(**data))
    return series


def test_detect_remesh_marks_structural_discontinuities() -> None:
    samples = [
        {"line_deviation": 0.0, "yaw_rate": 0.01, "structural_timestamp": 0.0},
        {"line_deviation": 0.3, "yaw_rate": 0.12, "structural_timestamp": 0.5},
        {"line_deviation": 0.8, "yaw_rate": 0.4, "structural_timestamp": 1.3},
        {"line_deviation": 1.4, "yaw_rate": 0.75, "structural_timestamp": 2.5},
        {"line_deviation": 1.7, "yaw_rate": 0.9, "structural_timestamp": 3.3},
        {"line_deviation": 1.8, "yaw_rate": 0.95, "structural_timestamp": 3.9},
        {"line_deviation": 1.6, "yaw_rate": 0.6, "structural_timestamp": 4.2},
    ]
    series = _build_series(samples)

    events = detect_remesh(
        series,
        window=5,
        line_gradient_threshold=0.25,
        yaw_rate_gradient_threshold=0.22,
        structural_gap_threshold=0.5,
    )

    assert events
    event = events[0]
    assert event["name"] == "Remeshing"
    assert event["line_gradient_mean"] >= 0.25
    assert event["yaw_rate_gradient_mean"] >= 0.22
    assert event["structural_gap_mean"] >= 0.5


@pytest.mark.parametrize(
    "samples",
    [
        [
            {"line_deviation": 0.1 + index * 0.02, "yaw_rate": 0.05 + index * 0.01, "structural_timestamp": index * 0.1}
            for index in range(6)
        ],
        [
            {"line_deviation": 0.0, "yaw_rate": 0.02, "structural_timestamp": 0.0},
            {"line_deviation": math.nan, "yaw_rate": 0.02, "structural_timestamp": 0.2},
            {"line_deviation": 0.02, "yaw_rate": 0.02, "structural_timestamp": 0.4},
            {"line_deviation": 0.04, "yaw_rate": 0.03, "structural_timestamp": 0.6},
            {"line_deviation": 0.05, "yaw_rate": 0.03, "structural_timestamp": 0.8},
        ],
    ],
)
def test_detect_remesh_requires_sharp_gradients(samples: list[dict]) -> None:
    series = _build_series(samples)
    events = detect_remesh(
        series,
        window=4,
        line_gradient_threshold=0.25,
        yaw_rate_gradient_threshold=0.22,
        structural_gap_threshold=0.5,
    )
    assert events == []
