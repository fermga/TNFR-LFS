"""Synthetic tests for the reception (EN) detector."""

from __future__ import annotations

import math

import pytest

from tnfr_core.operators.operator_detection import detect_en

from tests.helpers import build_telemetry_record


_BASE_PAYLOAD = dict(
    vertical_load=5200.0,
    slip_ratio=0.0,
    lateral_accel=0.0,
    longitudinal_accel=0.0,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    brake_pressure=0.2,
    locking=0.0,
    nfr=90.0,
    si=0.45,
    speed=38.0,
    yaw_rate=0.02,
    slip_angle=0.0,
    steer=0.1,
    throttle=0.25,
    gear=3,
    vertical_load_front=2600.0,
    vertical_load_rear=2600.0,
    mu_eff_front=1.05,
    mu_eff_rear=1.01,
    suspension_travel_front=0.03,
    suspension_travel_rear=0.03,
    suspension_velocity_front=0.01,
    suspension_velocity_rear=0.01,
    tyre_temp_fl=80.0,
    tyre_temp_fr=79.0,
    tyre_temp_rl=76.0,
    tyre_temp_rr=77.0,
    tyre_pressure_fl=1.5,
    tyre_pressure_fr=1.5,
    tyre_pressure_rl=1.6,
    tyre_pressure_rr=1.6,
    rpm=5200.0,
    line_deviation=0.05,
)


def _build_series(samples: list[dict]) -> list:
    series = []
    for index, payload in enumerate(samples):
        data = dict(_BASE_PAYLOAD)
        data.update(payload)
        data.setdefault("timestamp", index * 0.1)
        series.append(build_telemetry_record(**data))
    return series


def test_detect_en_emits_event_with_high_si_and_stable_nfr() -> None:
    samples = [
        {"si": 0.58, "nfr": 100.0, "throttle": 0.4},
        {"si": 0.64, "nfr": 101.5, "throttle": 0.52},
        {"si": 0.69, "nfr": 102.0, "throttle": 0.55},
        {"si": 0.73, "nfr": 100.8, "throttle": 0.58},
        {"si": 0.72, "nfr": 102.2, "throttle": 0.6, "yaw_rate": float("nan")},
        {"si": 0.45, "nfr": 140.0, "throttle": 0.15},
    ]
    series = _build_series(samples)

    events = detect_en(series, window=4, si_threshold=0.6, nfr_span_threshold=5.0, throttle_threshold=0.4)

    assert events
    event = events[0]
    assert event["name"] == "Reception"
    assert event["severity"] >= 1.0
    assert event["si_mean"] >= 0.6
    assert event["nfr_span"] <= 5.0
    assert event["throttle_mean"] >= 0.4


@pytest.mark.parametrize(
    "samples",
    [
        [{"si": 0.4, "nfr": 100.0, "throttle": 0.5} for _ in range(5)],
        [
            {"si": 0.65, "nfr": 100.0, "throttle": 0.5},
            {"si": 0.66, "nfr": 100.0, "throttle": 0.2},
            {"si": 0.67, "nfr": 112.0, "throttle": 0.55},
            {"si": 0.68, "nfr": 120.0, "throttle": 0.6},
        ],
        [
            {"si": 0.62, "nfr": 100.0, "throttle": 0.38},
            {"si": 0.63, "nfr": 101.0, "throttle": 0.4},
            {"si": math.nan, "nfr": 103.0, "throttle": 0.41},
            {"si": 0.61, "nfr": 109.0, "throttle": 0.39},
        ],
    ],
)
def test_detect_en_ignores_series_below_threshold(samples: list[dict]) -> None:
    series = _build_series(samples)
    events = detect_en(series, window=3, si_threshold=0.6, nfr_span_threshold=4.0, throttle_threshold=0.45)
    assert events == []
