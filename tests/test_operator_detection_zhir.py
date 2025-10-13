"""Synthetic tests for the transformation (ZHIR) detector."""

from __future__ import annotations

import pytest

from tnfr_core.operators.operator_detection import detect_zhir

from tests.helpers import build_telemetry_record


_BASE_PAYLOAD = dict(
    vertical_load=5150.0,
    slip_ratio=0.0,
    lateral_accel=0.3,
    longitudinal_accel=0.1,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    brake_pressure=0.15,
    locking=0.0,
    nfr=50.0,
    si=0.2,
    speed=42.0,
    yaw_rate=0.03,
    slip_angle=0.01,
    steer=0.12,
    throttle=0.4,
    gear=4,
    vertical_load_front=2550.0,
    vertical_load_rear=2600.0,
    mu_eff_front=1.0,
    mu_eff_rear=1.0,
    suspension_velocity_front=0.012,
    suspension_velocity_rear=0.015,
    rpm=6000.0,
    line_deviation=0.1,
)


def _build_series(samples: list[dict]) -> list:
    series = []
    for index, payload in enumerate(samples):
        data = dict(_BASE_PAYLOAD)
        data.update(payload)
        data.setdefault("timestamp", index * 0.1)
        series.append(build_telemetry_record(**data))
    return series


def test_detect_zhir_marks_large_si_and_nfr_shift() -> None:
    samples = [
        {"si": 0.18, "nfr": 40.0, "line_deviation": 0.05},
        {"si": 0.22, "nfr": 60.0, "line_deviation": 0.15},
        {"si": 0.28, "nfr": 95.0, "line_deviation": 0.35},
        {"si": 0.34, "nfr": 120.0, "line_deviation": 0.55},
        {"si": 0.46, "nfr": 165.0, "line_deviation": 0.75},
        {"si": 0.5, "nfr": 180.0, "line_deviation": 0.85},
        {"si": 0.48, "nfr": 150.0, "line_deviation": 0.9},
    ]
    series = _build_series(samples)

    events = detect_zhir(
        series,
        window=5,
        si_delta_threshold=0.25,
        nfr_delta_threshold=60.0,
        line_deviation_threshold=0.5,
    )

    assert events
    event = events[0]
    assert event["name"] == "Transformation"
    assert event["si_delta"] >= 0.25
    assert event["nfr_delta"] >= 60.0
    assert event["line_deviation_span"] >= 0.5


@pytest.mark.parametrize(
    "samples",
    [
        [{"si": 0.2 + index * 0.01, "nfr": 50.0 + index * 3.0, "line_deviation": 0.1} for index in range(6)],
        [
            {"si": 0.2, "nfr": 50.0, "line_deviation": 0.1},
            {"si": 0.22, "nfr": 65.0, "line_deviation": 0.15},
            {"si": 0.24, "nfr": 80.0, "line_deviation": 0.18},
            {"si": 0.26, "nfr": 95.0, "line_deviation": 0.2},
        ],
    ],
)
def test_detect_zhir_requires_all_signals(samples: list[dict]) -> None:
    series = _build_series(samples)
    events = detect_zhir(series, window=4, si_delta_threshold=0.25, nfr_delta_threshold=60.0, line_deviation_threshold=0.5)
    assert events == []
