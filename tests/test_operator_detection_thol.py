from __future__ import annotations

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
    nfr=0.0,
    si=0.0,
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


def _series(payloads: list[dict]) -> list:
    samples = []
    for index, overrides in enumerate(payloads):
        payload = dict(_BASE_PAYLOAD)
        payload.update(overrides)
        payload.setdefault("timestamp", index * 0.1)
        samples.append(build_telemetry_record(**payload))
    return samples


def test_detect_thol_reports_epi_acceleration_followed_by_stability() -> None:
    samples = _series(
        [
            {"nfr": 0.5, "si": 0.02},
            {"nfr": 3.0, "si": 0.05},
            {"nfr": 10.0, "si": 0.1},
            {"nfr": 22.0, "si": 0.18},
            {"nfr": 35.0, "si": 0.25},
            {"nfr": 35.1, "si": 0.25},
            {"nfr": 35.1, "si": 0.25},
            {"nfr": 35.1, "si": 0.25},
        ]
    )

    events = detect_thol(
        samples,
        epi_accel_min=2.0,
        stability_window=0.3,
        stability_tolerance=0.05,
    )

    assert events
    event = events[0]
    assert event["name"] == "Auto-organisation"
    assert event["epi_second_derivative"] >= 5.0
    assert event["stability_duration"] >= 0.2


def test_detect_thol_requires_post_acceleration_stability() -> None:
    samples = _series(
        [
            {"nfr": 0.5, "si": 0.02},
            {"nfr": 1.5, "si": 0.05},
            {"nfr": 3.8, "si": 0.09},
            {"nfr": 7.2, "si": 0.14},
            {"nfr": 11.0, "si": 0.2},
            {"nfr": 13.5, "si": 0.23},
            {"nfr": 16.0, "si": 0.26},
        ]
    )

    events = detect_thol(
        samples,
        epi_accel_min=2.0,
        stability_window=0.3,
        stability_tolerance=0.05,
    )

    assert events == []


def test_detect_thol_ignores_negative_epi_acceleration() -> None:
    samples = _series(
        [
            {"nfr": 40.0, "si": 0.0},
            {"nfr": 30.0, "si": 0.0},
            {"nfr": 15.0, "si": 0.0},
            {"nfr": 15.0, "si": 0.0},
            {"nfr": 15.0, "si": 0.0},
            {"nfr": 15.0, "si": 0.0},
        ]
    )

    events = detect_thol(
        samples,
        epi_accel_min=2.0,
        stability_window=0.2,
        stability_tolerance=0.05,
    )

    assert events == []
