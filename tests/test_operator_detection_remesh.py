from __future__ import annotations

from math import sin, tau

from tnfr_core.operators.operator_detection import detect_remesh

from tests.helpers import build_telemetry_record


_BASE_PAYLOAD = dict(
    vertical_load=5050.0,
    slip_ratio=0.0,
    lateral_accel=0.4,
    longitudinal_accel=0.2,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    brake_pressure=0.12,
    locking=0.0,
    nfr=90.0,
    si=0.3,
    speed=38.0,
    yaw_rate=0.04,
    slip_angle=0.02,
    steer=0.1,
    throttle=0.4,
    gear=3,
    vertical_load_front=2500.0,
    vertical_load_rear=2550.0,
    mu_eff_front=1.0,
    mu_eff_rear=1.0,
    suspension_velocity_front=0.01,
    suspension_velocity_rear=0.02,
    rpm=5600.0,
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


def test_detect_remesh_marks_repeated_patterns() -> None:
    samples = []
    for index in range(20):
        t = index * 0.1
        line = 0.4 * sin(tau * 0.5 * t) + 0.05 * sin(tau * t)
        samples.append({"line_deviation": line})
    samples.extend({"line_deviation": 0.0, "timestamp": (20 + idx) * 0.1} for idx in range(5))
    series = _series(samples)

    events = detect_remesh(
        series,
        window=10,
        tau_candidates=(0.2, 0.4, 0.6),
        acf_min=0.7,
        min_repeats=2,
    )

    assert events
    event = events[0]
    assert event["name"] == "Remeshing"
    assert event["matched_lags"] >= 2
    assert event["best_correlation"] >= 0.7


def test_detect_remesh_requires_repetition() -> None:
    baseline_noise = [
        -0.17616723516683763,
        -0.3491508260754981,
        0.15093447303985374,
        -0.42756371333245724,
        0.0358820043066892,
        -0.13431108308741446,
        -0.4420010752252932,
        0.007435733189420257,
        -0.4625043415580151,
        -0.06635431633761413,
        -0.43014457642538106,
        -0.40928698665613494,
        -0.07548081085748604,
        0.32685212467203806,
        -0.3761980388503544,
        -0.27676103539298547,
        0.1274332224055893,
        0.44770894245700565,
        0.07710294861749867,
        -0.10331952534921984,
    ]
    samples = _series([{ "line_deviation": value } for value in baseline_noise])

    events = detect_remesh(
        samples,
        window=10,
        tau_candidates=(0.2, 0.4, 0.6),
        acf_min=0.7,
        min_repeats=2,
    )

    assert events == []
