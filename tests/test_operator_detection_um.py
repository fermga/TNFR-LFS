"""Synthetic tests for the coupling (UM) detector."""

from __future__ import annotations

import pytest

from tnfr_core.operators.operator_detection import detect_um

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
    nfr=110.0,
    si=0.5,
    speed=35.0,
    yaw_rate=0.03,
    slip_angle=0.0,
    steer=0.1,
    throttle=0.32,
    gear=3,
    vertical_load_front=2500.0,
    vertical_load_rear=2700.0,
    mu_eff_front=1.0,
    mu_eff_rear=1.0,
    suspension_travel_front=0.03,
    suspension_travel_rear=0.03,
    suspension_velocity_front=0.005,
    suspension_velocity_rear=0.005,
    rpm=5000.0,
)


def _build_series(samples: list[dict]) -> list:
    series = []
    for index, payload in enumerate(samples):
        data = dict(_BASE_PAYLOAD)
        data.update(payload)
        data.setdefault("timestamp", index * 0.1)
        series.append(build_telemetry_record(**data))
    return series


def test_detect_um_flags_large_mu_delta_and_load_shift() -> None:
    samples = [
        {
            "mu_eff_front": 1.2,
            "mu_eff_rear": 0.85,
            "vertical_load_front": 3000.0,
            "vertical_load_rear": 2400.0,
            "suspension_velocity_front": 0.04,
            "suspension_velocity_rear": -0.02,
        },
        {
            "mu_eff_front": 1.18,
            "mu_eff_rear": 0.82,
            "vertical_load_front": 3050.0,
            "vertical_load_rear": 2350.0,
            "suspension_velocity_front": 0.05,
            "suspension_velocity_rear": -0.03,
        },
        {
            "mu_eff_front": 1.17,
            "mu_eff_rear": 0.81,
            "vertical_load_front": 3100.0,
            "vertical_load_rear": 2300.0,
            "suspension_velocity_front": 0.06,
            "suspension_velocity_rear": -0.04,
        },
        {
            "mu_eff_front": 1.16,
            "mu_eff_rear": 0.8,
            "vertical_load_front": 3120.0,
            "vertical_load_rear": 2280.0,
            "suspension_velocity_front": 0.05,
            "suspension_velocity_rear": -0.05,
        },
        {
            "mu_eff_front": 1.12,
            "mu_eff_rear": 0.79,
            "vertical_load_front": 2800.0,
            "vertical_load_rear": 2400.0,
            "suspension_velocity_front": 0.02,
            "suspension_velocity_rear": -0.01,
        },
        {
            "mu_eff_front": 1.0,
            "mu_eff_rear": 0.99,
            "vertical_load_front": 2550.0,
            "vertical_load_rear": 2650.0,
            "suspension_velocity_front": 0.01,
            "suspension_velocity_rear": 0.0,
        },
    ]
    series = _build_series(samples)

    events = detect_um(
        series,
        window=4,
        mu_delta_threshold=0.25,
        load_ratio_threshold=0.05,
        suspension_delta_threshold=0.02,
    )

    assert events
    event = events[0]
    assert event["name"] == "Coupling"
    assert event["mu_delta"] >= 0.25
    assert event["load_ratio_delta"] >= 0.05
    assert event["suspension_velocity_delta"] >= 0.02


def test_detect_um_rejects_balanced_series() -> None:
    samples = [
        {
            "mu_eff_front": 1.05,
            "mu_eff_rear": 1.02,
            "vertical_load_front": 2700.0,
            "vertical_load_rear": 2500.0,
            "suspension_velocity_front": 0.01,
            "suspension_velocity_rear": 0.015,
        }
        for _ in range(6)
    ]
    series = _build_series(samples)
    events = detect_um(series, window=4, mu_delta_threshold=0.25, load_ratio_threshold=0.05)
    assert events == []
