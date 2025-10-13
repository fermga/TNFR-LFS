"""Synthetic tests for the propagation (RA) detector."""

from __future__ import annotations

import pytest

from tnfr_core.operators.operator_detection import detect_ra

from tests.helpers import build_telemetry_record


_BASE_PAYLOAD = dict(
    vertical_load=5100.0,
    slip_ratio=0.0,
    lateral_accel=0.4,
    longitudinal_accel=0.2,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    brake_pressure=0.1,
    locking=0.0,
    nfr=80.0,
    si=0.2,
    speed=25.0,
    yaw_rate=0.04,
    slip_angle=0.0,
    steer=0.05,
    throttle=0.5,
    gear=3,
    vertical_load_front=2550.0,
    vertical_load_rear=2550.0,
    mu_eff_front=1.0,
    mu_eff_rear=1.0,
    suspension_velocity_front=0.01,
    suspension_velocity_rear=0.01,
    rpm=5400.0,
)


def _build_series(samples: list[dict]) -> list:
    series = []
    for index, payload in enumerate(samples):
        data = dict(_BASE_PAYLOAD)
        data.update(payload)
        data.setdefault("timestamp", index * 0.1)
        series.append(build_telemetry_record(**data))
    return series


def test_detect_ra_identifies_high_delta_diffusion() -> None:
    samples = [
        {"nfr": 80.0, "si": 0.2, "speed": 30.0},
        {"nfr": 115.0, "si": 0.36, "speed": 36.0},
        {"nfr": 150.0, "si": 0.44, "speed": 39.0},
        {"nfr": 205.0, "si": 0.5, "speed": 42.0},
        {"nfr": 250.0, "si": 0.55, "speed": 45.0},
        {"nfr": 252.0, "si": 0.56, "speed": 43.0},
        {"nfr": 210.0, "si": 0.48, "speed": 40.0},
    ]
    series = _build_series(samples)

    events = detect_ra(
        series,
        window=4,
        nfr_rate_threshold=30.0,
        si_span_threshold=0.2,
        speed_threshold=30.0,
    )

    assert events
    event = events[0]
    assert event["name"] == "Propagation"
    assert event["nfr_rate_mean"] >= 30.0
    assert event["si_span"] >= 0.2
    assert event["speed_mean"] >= 30.0


@pytest.mark.parametrize(
    "samples",
    [
        [{"nfr": 100.0 + index, "si": 0.25, "speed": 22.0} for index in range(6)],
        [
            {"nfr": 100.0, "si": 0.2, "speed": 45.0},
            {"nfr": 102.0, "si": 0.22, "speed": 44.0},
            {"nfr": 104.0, "si": 0.23, "speed": 43.0},
            {"nfr": 106.0, "si": 0.24, "speed": 42.0},
        ],
    ],
)
def test_detect_ra_skips_low_diffusion(samples: list[dict]) -> None:
    series = _build_series(samples)
    events = detect_ra(series, window=3, nfr_rate_threshold=30.0, si_span_threshold=0.2, speed_threshold=28.0)
    assert events == []
