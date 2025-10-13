from __future__ import annotations

from tnfr_core.operators.operator_detection import detect_zhir

from tests.helpers import build_telemetry_record


_BASE_PAYLOAD = dict(
    vertical_load=5150.0,
    slip_ratio=0.05,
    lateral_accel=0.3,
    longitudinal_accel=0.1,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    brake_pressure=0.15,
    locking=0.0,
    nfr=20.0,
    si=0.1,
    speed=42.0,
    yaw_rate=0.03,
    slip_angle=0.01,
    steer=0.12,
    throttle=0.4,
    gear=4,
    vertical_load_front=2550.0,
    vertical_load_rear=2600.0,
    mu_eff_front=0.95,
    mu_eff_rear=0.96,
    suspension_velocity_front=0.012,
    suspension_velocity_rear=0.015,
    rpm=6000.0,
    line_deviation=0.1,
)


def _series(payloads: list[dict]) -> list:
    samples = []
    for index, overrides in enumerate(payloads):
        payload = dict(_BASE_PAYLOAD)
        payload.update(overrides)
        payload.setdefault("timestamp", index * 0.1)
        samples.append(build_telemetry_record(**payload))
    return samples


def test_detect_zhir_marks_phase_change_with_epi_derivative() -> None:
    samples = _series(
        [
            {
                "mu_eff_front": 0.8,
                "mu_eff_rear": 0.82,
                "slip_ratio": 0.02,
                "slip_angle": 0.01,
                "si": 0.1,
                "nfr": 16.0,
            },
            {
                "mu_eff_front": 0.85,
                "mu_eff_rear": 0.86,
                "slip_ratio": 0.03,
                "slip_angle": 0.015,
                "si": 0.11,
                "nfr": 18.0,
            },
            {
                "mu_eff_front": 1.05,
                "mu_eff_rear": 1.02,
                "slip_ratio": 0.1,
                "slip_angle": 0.06,
                "si": 0.24,
                "nfr": 35.0,
            },
            {
                "mu_eff_front": 1.12,
                "mu_eff_rear": 1.08,
                "slip_ratio": 0.14,
                "slip_angle": 0.08,
                "si": 0.32,
                "nfr": 52.0,
            },
            {
                "mu_eff_front": 1.15,
                "mu_eff_rear": 1.1,
                "slip_ratio": 0.16,
                "slip_angle": 0.09,
                "si": 0.34,
                "nfr": 58.0,
            },
            {
                "mu_eff_front": 1.17,
                "mu_eff_rear": 1.11,
                "slip_ratio": 0.17,
                "slip_angle": 0.09,
                "si": 0.35,
                "nfr": 60.0,
            },
            {
                "mu_eff_front": 1.18,
                "mu_eff_rear": 1.12,
                "slip_ratio": 0.17,
                "slip_angle": 0.09,
                "si": 0.35,
                "nfr": 61.0,
            },
        ]
    )

    events = detect_zhir(
        samples,
        window=6,
        xi_min=0.15,
        min_persistence=0.3,
        phase_jump_min=0.15,
    )

    assert events
    event = events[0]
    assert event["name"] == "Transformation"
    assert event["phase_jump"] >= 0.2
    assert event["epi_derivative"] >= 0.3
    assert event["persistence"] >= 0.3


def test_detect_zhir_requires_phase_jump() -> None:
    samples = _series(
        [
            {
                "mu_eff_front": 0.95 + 0.01 * index,
                "mu_eff_rear": 0.96 + 0.01 * index,
                "slip_ratio": 0.05 + 0.005 * index,
                "slip_angle": 0.02 + 0.004 * index,
                "si": 0.14 + 0.01 * index,
                "nfr": 22.0 + 1.5 * index,
            }
            for index in range(10)
        ]
    )

    events = detect_zhir(
        samples,
        window=6,
        xi_min=0.15,
        min_persistence=0.3,
        phase_jump_min=0.15,
    )

    assert events == []
