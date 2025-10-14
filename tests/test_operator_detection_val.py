from __future__ import annotations

from math import nan

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
    nfr=20.0,
    si=0.2,
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
    wheel_load_fl=400.0,
    wheel_load_fr=400.0,
    wheel_load_rl=150.0,
    wheel_load_rr=150.0,
    rpm=6200.0,
)


def _series(payloads: list[dict]) -> list:
    samples = []
    for index, overrides in enumerate(payloads):
        payload = dict(_BASE_PAYLOAD)
        payload.update(overrides)
        payload.setdefault("timestamp", index * 0.1)
        samples.append(build_telemetry_record(**payload))
    return samples


def test_detect_val_emits_event_on_epi_growth_and_nodes() -> None:
    samples = _series(
        [
            {"nfr": 18.0, "si": 0.18, "wheel_load_rl": 120.0, "wheel_load_rr": 120.0},
            {"nfr": 20.0, "si": 0.22, "wheel_load_rl": 180.0, "wheel_load_rr": 180.0},
            {"nfr": 23.5, "si": 0.28, "wheel_load_rl": 260.0, "wheel_load_rr": 260.0},
            {"nfr": 27.0, "si": 0.35, "wheel_load_rl": 320.0, "wheel_load_rr": 320.0},
            {"nfr": 30.0, "si": 0.4, "wheel_load_rl": 330.0, "wheel_load_rr": 330.0},
            {"nfr": 28.0, "si": 0.36, "wheel_load_rl": 280.0, "wheel_load_rr": 280.0},
        ]
    )

    events = detect_val(
        samples,
        window=4,
        epi_growth_min=0.3,
        active_nodes_delta_min=2,
        active_node_load_min=200.0,
    )

    assert events
    event = events[0]
    assert event["name"] == "Amplification"
    assert event["epi_growth_rate"] >= 0.3
    assert event["active_nodes_delta"] >= 2


def test_detect_val_requires_node_growth() -> None:
    samples = _series(
        [
            {"nfr": 18.0, "si": 0.2, "wheel_load_rl": 260.0, "wheel_load_rr": 260.0},
            {"nfr": 21.0, "si": 0.23, "wheel_load_rl": 255.0, "wheel_load_rr": 255.0},
            {"nfr": 24.0, "si": 0.26, "wheel_load_rl": 250.0, "wheel_load_rr": 250.0},
            {"nfr": 27.0, "si": 0.29, "wheel_load_rl": 245.0, "wheel_load_rr": 245.0},
            {"nfr": 29.0, "si": 0.32, "wheel_load_rl": 240.0, "wheel_load_rr": 240.0},
        ]
    )

    events = detect_val(
        samples,
        window=4,
        epi_growth_min=0.3,
        active_nodes_delta_min=2,
        active_node_load_min=200.0,
    )

    assert events == []


def test_detect_val_ignores_nan_wheel_loads() -> None:
    payloads = [
        {"nfr": 17.5, "si": 0.17, "wheel_load_rl": 120.0, "wheel_load_rr": 120.0},
        {"nfr": 20.0, "si": 0.22, "wheel_load_rl": nan, "wheel_load_rr": 210.0},
        {"nfr": 24.5, "si": 0.3, "wheel_load_rl": 280.0, "wheel_load_rr": 280.0},
        {"nfr": 28.0, "si": 0.36, "wheel_load_rl": 330.0, "wheel_load_rr": 330.0},
        {"nfr": 29.0, "si": 0.38, "wheel_load_rl": 310.0, "wheel_load_rr": 320.0},
    ]

    samples = _series(payloads)

    events = detect_val(
        samples,
        window=4,
        epi_growth_min=0.3,
        active_nodes_delta_min=2,
        active_node_load_min=200.0,
    )

    assert events
    assert events[0]["active_nodes_delta"] >= 2
