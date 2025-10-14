from __future__ import annotations

from math import nan, pi, sin

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
    si=0.6,
    speed=42.0,
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


def _series(payloads: list[dict]) -> list:
    samples = []
    for index, overrides in enumerate(payloads):
        payload = dict(_BASE_PAYLOAD)
        payload.update(overrides)
        payload.setdefault("timestamp", index * 0.05)
        samples.append(build_telemetry_record(**payload))
    return samples


def test_detect_ra_identifies_resonant_band_power() -> None:
    samples: list[dict] = []
    for index in range(24):
        t = index * 0.05
        nfr = 60.0 + 12.0 * sin(2.0 * pi * 1.5 * t)
        samples.append({"nfr": nfr, "si": 0.62, "speed": 44.0})
    samples.extend(
        {"nfr": 40.0 + index, "si": 0.4, "speed": 30.0, "timestamp": (24 + idx) * 0.05}
        for idx in range(6)
    )
    series = _series(samples)

    events = detect_ra(
        series,
        window=12,
        nu_band=(1.0, 3.0),
        si_min=0.58,
        delta_nfr_max=15.0,
        k_min=2,
    )

    assert events
    event = events[0]
    assert event["name"] == "Propagation"
    assert event["band_power"] > 0.0
    assert event["si_mean"] >= 0.58
    assert event["delta_nfr_dispersion"] <= 15.0
    assert event["severity"] > 1.0
    assert event["peak_value"] == pytest.approx(event["peak_band_power"])


def test_detect_ra_requires_sustained_windows() -> None:
    samples: list[dict] = []
    for index in range(20):
        t = index * 0.05
        nfr = 60.0 + 12.0 * sin(2.0 * pi * 1.5 * t)
        si = 0.45 + 0.05 * sin(2.0 * pi * 0.3 * t)
        samples.append({"nfr": nfr, "si": si, "speed": 38.0})
    series = _series(samples)

    events = detect_ra(
        series,
        window=12,
        nu_band=(1.0, 3.0),
        si_min=0.58,
        delta_nfr_max=15.0,
        k_min=2,
    )

    assert events == []


def test_detect_ra_severity_tracks_band_power_margin() -> None:
    strong_samples: list[dict] = []
    for index in range(36):
        t = index * 0.05
        nfr = 55.0 + 18.0 * sin(2.0 * pi * 1.4 * t)
        strong_samples.append({"nfr": nfr, "si": 0.68, "speed": 46.0})

    borderline_samples: list[dict] = []
    for index in range(36):
        t = index * 0.05
        nfr = 58.0 + 10.0 * sin(2.0 * pi * 1.4 * t)
        borderline_samples.append({"nfr": nfr, "si": 0.59, "speed": 44.0})

    strong_events = detect_ra(
        _series(strong_samples),
        window=12,
        nu_band=(1.0, 3.0),
        si_min=0.58,
        delta_nfr_max=15.0,
        k_min=2,
    )
    borderline_events = detect_ra(
        _series(borderline_samples),
        window=12,
        nu_band=(1.0, 3.0),
        si_min=0.58,
        delta_nfr_max=15.0,
        k_min=2,
    )

    assert strong_events and borderline_events
    strong_severity = strong_events[0]["severity"]
    borderline_severity = borderline_events[0]["severity"]

    assert strong_severity > 1.1
    assert borderline_severity == pytest.approx(1.0, abs=0.15)


def test_detect_ra_skips_nan_samples_without_crashing() -> None:
    samples: list[dict] = []
    for index in range(30):
        t = index * 0.05
        if index == 10:
            samples.append({"nfr": nan, "si": nan, "speed": 44.0})
            continue
        nfr = 60.0 + 12.0 * sin(2.0 * pi * 1.6 * t)
        si = 0.6 + 0.02 * sin(2.0 * pi * 0.4 * t)
        samples.append({"nfr": nfr, "si": si, "speed": 45.0})

    series = _series(samples)

    events = detect_ra(
        series,
        window=12,
        nu_band=(1.0, 3.0),
        si_min=0.58,
        delta_nfr_max=18.0,
        k_min=2,
    )

    assert events
    assert events[0]["band_power"] > 0.0
