from __future__ import annotations

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
    nfr=100.0,
    si=0.5,
    speed=40.0,
    yaw_rate=0.0,
    slip_angle=0.0,
    steer=0.0,
    throttle=0.35,
    gear=3,
    vertical_load_front=2500.0,
    vertical_load_rear=2700.0,
    mu_eff_front=1.0,
    mu_eff_rear=1.0,
    suspension_travel_front=0.03,
    suspension_travel_rear=0.03,
    suspension_velocity_front=0.01,
    suspension_velocity_rear=0.01,
)


def _series(payloads: list[dict]) -> list:
    samples = []
    for index, overrides in enumerate(payloads):
        payload = dict(_BASE_PAYLOAD)
        payload.update(overrides)
        payload.setdefault("timestamp", index * 0.1)
        samples.append(build_telemetry_record(**payload))
    return samples


def test_detect_um_identifies_coupled_slip_and_yaw() -> None:
    samples = []
    for index in range(8):
        steer = 0.02 * index
        yaw_rate = 0.018 * index + 0.002
        slip_ratio = 0.015 * index
        slip_angle = 0.012 * index
        samples.append(
            {
                "steer": steer,
                "yaw_rate": yaw_rate,
                "slip_ratio": slip_ratio,
                "slip_angle": slip_angle,
            }
        )
    samples.extend(
        [
            {
                "steer": 0.0,
                "yaw_rate": 0.0,
                "slip_ratio": 0.0,
                "slip_angle": 0.0,
                "timestamp": (8 + idx) * 0.1,
            }
            for idx in range(2)
        ]
    )
    series = _series(samples)

    events = detect_um(
        series,
        window=6,
        rho_min=0.6,
        phase_max=0.12,
        min_duration=0.3,
    )

    assert events
    event = events[0]
    assert event["name"] == "Coupling"
    assert event["max_coupling"] >= 0.6
    assert event["phase_lag"] <= 0.12
    assert event["duration"] >= 0.3


def test_detect_um_rejects_large_phase_lag() -> None:
    samples = []
    for index in range(10):
        steer = 0.02 * index
        yaw_rate = 0.0
        slip_ratio = 0.015 * index
        slip_angle = 0.012 * index
        samples.append(
            {
                "steer": steer,
                "yaw_rate": yaw_rate,
                "slip_ratio": slip_ratio,
                "slip_angle": slip_angle,
            }
        )
    series = _series(samples)

    events = detect_um(
        series,
        window=6,
        rho_min=0.6,
        phase_max=0.12,
        min_duration=0.3,
    )

    assert events == []
