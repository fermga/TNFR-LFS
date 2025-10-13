from __future__ import annotations

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
    nfr=20.0,
    si=0.2,
    speed=40.0,
    yaw_rate=0.05,
    slip_angle=0.0,
    steer=0.0,
    throttle=0.3,
    gear=3,
    vertical_load_front=2600.0,
    vertical_load_rear=2600.0,
    mu_eff_front=1.02,
    mu_eff_rear=1.01,
    suspension_travel_front=0.03,
    suspension_travel_rear=0.03,
    suspension_velocity_front=0.0,
    suspension_velocity_rear=0.0,
    wheel_load_fl=650.0,
    wheel_load_fr=650.0,
    wheel_load_rl=650.0,
    wheel_load_rr=650.0,
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


def _series(payloads: list[dict]) -> list:
    samples = []
    for index, overrides in enumerate(payloads):
        payload = dict(_BASE_PAYLOAD)
        payload.update(overrides)
        payload.setdefault("timestamp", index * 0.1)
        samples.append(build_telemetry_record(**payload))
    return samples


def test_detect_en_integrates_psi_flux_and_epi_growth() -> None:
    samples = _series(
        [
            {
                "suspension_velocity_front": 0.28,
                "suspension_velocity_rear": 0.26,
                "steer": 0.02,
                "nfr": 18.0,
                "si": 0.21,
            },
            {
                "suspension_velocity_front": 0.3,
                "suspension_velocity_rear": 0.29,
                "steer": 0.05,
                "nfr": 19.5,
                "si": 0.23,
            },
            {
                "suspension_velocity_front": 0.34,
                "suspension_velocity_rear": 0.31,
                "steer": 0.08,
                "nfr": 21.0,
                "si": 0.25,
                "vertical_load": 5205.0,
            },
            {
                "suspension_velocity_front": 0.36,
                "suspension_velocity_rear": 0.34,
                "steer": 0.11,
                "nfr": 22.5,
                "si": 0.27,
                "vertical_load": 5208.0,
            },
            {
                "suspension_velocity_front": 0.33,
                "suspension_velocity_rear": 0.32,
                "steer": 0.14,
                "nfr": 24.0,
                "si": 0.28,
                "vertical_load": 5210.0,
            },
            {
                "suspension_velocity_front": 0.3,
                "suspension_velocity_rear": 0.28,
                "steer": 0.16,
                "nfr": 25.5,
                "si": 0.3,
                "vertical_load": 5211.0,
            },
            {
                "suspension_velocity_front": 0.1,
                "suspension_velocity_rear": 0.09,
                "steer": 0.05,
                "nfr": 40.0,
                "si": 0.18,
            },
        ]
    )

    events = detect_en(samples, window=6, psi_threshold=0.6, epi_norm_max=80.0)

    assert events
    event = events[0]
    assert event["name"] == "Reception"
    assert event["start_index"] == 0
    assert event["end_index"] >= 5
    assert event["psi_integral"] >= 0.6
    assert event["epi_norm_end"] >= event["epi_norm_start"]
    assert event["epi_norm_peak"] <= 80.0


def test_detect_en_requires_rising_epi_norm() -> None:
    samples = _series(
        [
            {
                "suspension_velocity_front": 0.3,
                "suspension_velocity_rear": 0.28,
                "steer": 0.04,
                "nfr": 30.0,
                "si": 0.26,
            },
            {
                "suspension_velocity_front": 0.29,
                "suspension_velocity_rear": 0.27,
                "steer": 0.06,
                "nfr": 28.0,
                "si": 0.24,
            },
            {
                "suspension_velocity_front": 0.28,
                "suspension_velocity_rear": 0.26,
                "steer": 0.08,
                "nfr": 26.0,
                "si": 0.22,
            },
            {
                "suspension_velocity_front": 0.27,
                "suspension_velocity_rear": 0.25,
                "steer": 0.1,
                "nfr": 24.0,
                "si": 0.2,
            },
            {
                "suspension_velocity_front": 0.26,
                "suspension_velocity_rear": 0.24,
                "steer": 0.11,
                "nfr": 22.0,
                "si": 0.18,
            },
            {
                "suspension_velocity_front": 0.25,
                "suspension_velocity_rear": 0.23,
                "steer": 0.12,
                "nfr": 20.0,
                "si": 0.16,
            },
        ]
    )

    events = detect_en(samples, window=6, psi_threshold=0.55, epi_norm_max=80.0)

    assert events == []
