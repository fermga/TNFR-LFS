from __future__ import annotations

from tnfr_core.operators.operator_detection import detect_nul

from tests.helpers import build_telemetry_record


_BASE_PAYLOAD = dict(
    vertical_load=5200.0,
    slip_ratio=0.0,
    lateral_accel=0.6,
    longitudinal_accel=-0.2,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    brake_pressure=0.2,
    locking=0.0,
    nfr=22.0,
    si=0.25,
    speed=45.0,
    yaw_rate=0.02,
    slip_angle=0.01,
    steer=0.08,
    throttle=0.2,
    gear=4,
    vertical_load_front=2700.0,
    vertical_load_rear=2500.0,
    mu_eff_front=1.0,
    mu_eff_rear=1.0,
    suspension_velocity_front=0.01,
    suspension_velocity_rear=0.01,
    wheel_load_fl=360.0,
    wheel_load_fr=360.0,
    wheel_load_rl=350.0,
    wheel_load_rr=350.0,
    rpm=5800.0,
)


def _series(payloads: list[dict]) -> list:
    samples = []
    for index, overrides in enumerate(payloads):
        payload = dict(_BASE_PAYLOAD)
        payload.update(overrides)
        payload.setdefault("timestamp", index * 0.1)
        samples.append(build_telemetry_record(**payload))
    return samples


def test_detect_nul_detects_support_contraction() -> None:
    samples = _series(
        [
            {"nfr": 22.0, "si": 0.25, "wheel_load_fl": 360.0, "wheel_load_fr": 360.0},
            {"nfr": 24.0, "si": 0.28, "wheel_load_fl": 320.0, "wheel_load_fr": 320.0},
            {"nfr": 26.0, "si": 0.31, "wheel_load_fl": 250.0, "wheel_load_fr": 250.0},
            {"nfr": 28.0, "si": 0.34, "wheel_load_fl": 180.0, "wheel_load_fr": 180.0},
            {"nfr": 29.0, "si": 0.36, "wheel_load_fl": 160.0, "wheel_load_fr": 160.0},
            {"nfr": 27.0, "si": 0.33, "wheel_load_fl": 150.0, "wheel_load_fr": 150.0},
        ]
    )

    events = detect_nul(
        samples,
        window=4,
        active_nodes_delta_max=-2,
        epi_concentration_min=0.55,
        active_node_load_min=250.0,
    )

    assert events
    event = events[0]
    assert event["name"] == "Contraction"
    assert event["active_nodes_delta"] <= -2
    assert event["epi_concentration_peak"] >= 0.55


def test_detect_nul_requires_concentration_threshold() -> None:
    samples = _series(
        [
            {"nfr": 0.32, "si": 0.3, "wheel_load_fl": 300.0, "wheel_load_fr": 300.0},
            {"nfr": 0.31, "si": 0.29, "wheel_load_fl": 240.0, "wheel_load_fr": 240.0},
            {"nfr": 0.3, "si": 0.28, "wheel_load_fl": 230.0, "wheel_load_fr": 230.0},
            {"nfr": 0.29, "si": 0.27, "wheel_load_fl": 220.0, "wheel_load_fr": 220.0},
            {"nfr": 0.28, "si": 0.26, "wheel_load_fl": 210.0, "wheel_load_fr": 210.0},
        ]
    )

    events = detect_nul(
        samples,
        window=4,
        active_nodes_delta_max=-2,
        epi_concentration_min=0.55,
        active_node_load_min=250.0,
    )

    assert events == []
