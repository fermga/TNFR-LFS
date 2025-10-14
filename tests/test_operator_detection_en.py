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


def _reference_en_metrics(samples: list, window: int) -> dict[str, list[float]]:
    times = [float(getattr(sample, "timestamp", index * 0.1)) for index, sample in enumerate(samples)]
    psi_flux: list[float] = []
    epi_norms: list[float] = []

    for index, sample in enumerate(samples):
        dt = times[index] - times[index - 1] if index > 0 else 0.0
        susp_front = abs(float(getattr(sample, "suspension_velocity_front", 0.0) or 0.0))
        susp_rear = abs(float(getattr(sample, "suspension_velocity_rear", 0.0) or 0.0))

        steer_value = float(getattr(sample, "steer", 0.0) or 0.0)
        prev_steer = (
            float(getattr(samples[index - 1], "steer", steer_value) or steer_value)
            if index > 0
            else steer_value
        )
        steer_rate = abs((steer_value - prev_steer) / dt) if dt > 0.0 else 0.0

        yaw_rate = abs(float(getattr(sample, "yaw_rate", 0.0) or 0.0))

        vertical = getattr(sample, "vertical_load", None)
        prev_vertical = getattr(samples[index - 1], "vertical_load", None) if index > 0 else vertical
        micro_vibration = 0.0
        if (
            vertical is not None
            and prev_vertical is not None
            and dt > 0.0
            and math.isfinite(vertical)
            and math.isfinite(prev_vertical)
        ):
            micro_vibration = abs(float(vertical) - float(prev_vertical)) / dt

        flux = (0.5 * (susp_front + susp_rear)) + steer_rate + yaw_rate + micro_vibration
        psi_flux.append(flux)

        nfr = float(getattr(sample, "nfr", 0.0) or 0.0)
        si = float(getattr(sample, "si", 0.0) or 0.0)
        epi_norms.append(math.hypot(nfr, si))

    psi_integral: list[float] = []
    epi_peak: list[float] = []
    start_indices: list[int] = []
    epi_start: list[float] = []

    for index in range(len(samples)):
        start_index = max(0, index - window + 1)
        start_indices.append(start_index)
        epi_start.append(epi_norms[start_index])
        window_epi = epi_norms[start_index : index + 1]
        epi_peak.append(max(window_epi))

        total = 0.0
        for j in range(start_index + 1, index + 1):
            dt = times[j] - times[j - 1]
            if dt <= 0.0:
                continue
            total += 0.5 * (psi_flux[j - 1] + psi_flux[j]) * dt
        psi_integral.append(total)

    return {
        "psi_integral": psi_integral,
        "start_indices": start_indices,
        "epi_start": epi_start,
        "epi_end": epi_norms,
        "epi_peak": epi_peak,
    }


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


def test_detect_en_handles_nan_payloads() -> None:
    payloads = [
        {
            "suspension_velocity_front": 0.26,
            "suspension_velocity_rear": 0.25,
            "steer": 0.02,
            "nfr": 18.5,
            "si": 0.2,
        },
        {
            "suspension_velocity_front": 0.28,
            "suspension_velocity_rear": 0.27,
            "steer": 0.05,
            "nfr": 19.0,
            "si": 0.22,
        },
        {
            "suspension_velocity_front": math.nan,
            "suspension_velocity_rear": math.nan,
            "steer": math.nan,
            "nfr": 19.5,
            "si": 0.23,
            "vertical_load": math.nan,
        },
        {
            "suspension_velocity_front": 0.32,
            "suspension_velocity_rear": 0.3,
            "steer": 0.08,
            "nfr": 21.5,
            "si": 0.24,
            "vertical_load": 5204.0,
        },
        {
            "suspension_velocity_front": 0.35,
            "suspension_velocity_rear": 0.33,
            "steer": 0.12,
            "nfr": 23.0,
            "si": 0.27,
            "vertical_load": 5208.0,
        },
        {
            "suspension_velocity_front": 0.34,
            "suspension_velocity_rear": 0.32,
            "steer": 0.15,
            "nfr": 24.5,
            "si": 0.28,
            "vertical_load": 5210.0,
        },
        {
            "suspension_velocity_front": 0.12,
            "suspension_velocity_rear": 0.1,
            "steer": 0.04,
            "nfr": 32.0,
            "si": 0.19,
        },
    ]

    samples = _series(payloads)

    events = detect_en(samples, window=6, psi_threshold=0.58, epi_norm_max=80.0)

    assert events
    assert events[0]["start_index"] == 0


def test_detect_en_vectorised_metrics_match_reference() -> None:
    samples = _series(
        [
            {
                "suspension_velocity_front": 0.25,
                "suspension_velocity_rear": 0.23,
                "steer": 0.03,
                "nfr": 17.5,
                "si": 0.19,
            },
            {
                "suspension_velocity_front": 0.28,
                "suspension_velocity_rear": 0.26,
                "steer": 0.06,
                "nfr": 19.0,
                "si": 0.21,
            },
            {
                "suspension_velocity_front": 0.31,
                "suspension_velocity_rear": 0.29,
                "steer": 0.09,
                "nfr": 20.5,
                "si": 0.24,
                "vertical_load": 5204.0,
            },
            {
                "suspension_velocity_front": 0.34,
                "suspension_velocity_rear": 0.32,
                "steer": 0.12,
                "nfr": 22.0,
                "si": 0.26,
                "vertical_load": 5208.0,
            },
            {
                "suspension_velocity_front": 0.33,
                "suspension_velocity_rear": 0.31,
                "steer": 0.14,
                "nfr": 23.5,
                "si": 0.28,
                "vertical_load": 5211.0,
            },
            {
                "suspension_velocity_front": 0.2,
                "suspension_velocity_rear": 0.19,
                "steer": 0.07,
                "nfr": 24.0,
                "si": 0.24,
                "vertical_load": 5206.0,
            },
            {
                "suspension_velocity_front": 0.12,
                "suspension_velocity_rear": 0.11,
                "steer": 0.05,
                "nfr": 21.0,
                "si": 0.2,
            },
        ]
    )

    window = 6
    psi_threshold = 0.6
    events = detect_en(samples, window=window, psi_threshold=psi_threshold, epi_norm_max=80.0)

    assert events
    event = events[0]
    reference = _reference_en_metrics(samples, window)
    end_index = event["end_index"]

    meets = [
        reference["psi_integral"][idx] >= psi_threshold
        and reference["epi_peak"][idx] <= 80.0
        and reference["epi_end"][idx] >= reference["epi_start"][idx] - 1e-6
        for idx in range(len(samples))
    ]
    first_active = next(idx for idx, flag in enumerate(meets) if flag)
    assert event["start_index"] == reference["start_indices"][first_active]
    assert event["psi_integral"] == pytest.approx(reference["psi_integral"][end_index], rel=1e-9, abs=1e-9)
    assert event["epi_norm_start"] == pytest.approx(reference["epi_start"][end_index], rel=1e-9, abs=1e-9)
    assert event["epi_norm_end"] == pytest.approx(reference["epi_end"][end_index], rel=1e-9, abs=1e-9)
    assert event["epi_norm_peak"] == pytest.approx(reference["epi_peak"][end_index], rel=1e-9, abs=1e-9)
