"""Tests for the operator detection utilities."""

from __future__ import annotations

from typing import List

import pytest

from tnfr_lfs.core.epi import TelemetryRecord
from tnfr_lfs.core.operator_detection import (
    canonical_operator_label,
    detect_al,
    detect_il,
    detect_oz,
    detect_silence,
    normalize_structural_operator_identifier,
    silence_event_payloads,
)

from tests.helpers import build_telemetry_record


_BASE_OPERATOR_PAYLOAD = dict(
    vertical_load=5000.0,
    slip_ratio=0.0,
    lateral_accel=0.0,
    longitudinal_accel=0.0,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    brake_pressure=0.0,
    locking=0.0,
    nfr=0.0,
    si=0.0,
    speed=30.0,
    yaw_rate=0.0,
    slip_angle=0.0,
    steer=0.0,
    throttle=0.0,
    gear=3,
    vertical_load_front=2500.0,
    vertical_load_rear=2500.0,
    mu_eff_front=0.0,
    mu_eff_rear=0.0,
    mu_eff_front_lateral=0.0,
    mu_eff_front_longitudinal=0.0,
    mu_eff_rear_lateral=0.0,
    mu_eff_rear_longitudinal=0.0,
    suspension_travel_front=0.0,
    suspension_travel_rear=0.0,
    suspension_velocity_front=0.0,
    suspension_velocity_rear=0.0,
    tyre_temp_fl=0.0,
    tyre_temp_fr=0.0,
    tyre_temp_rl=0.0,
    tyre_temp_rr=0.0,
    tyre_pressure_fl=0.0,
    tyre_pressure_fr=0.0,
    tyre_pressure_rl=0.0,
    tyre_pressure_rr=0.0,
    rpm=5000.0,
    line_deviation=0.0,
)


def _build_series(samples: List[dict]) -> List[TelemetryRecord]:
    records: List[TelemetryRecord] = []
    for index, payload in enumerate(samples):
        entry = dict(_BASE_OPERATOR_PAYLOAD)
        entry.update(payload)
        entry.setdefault("timestamp", index * 0.1)
        records.append(build_telemetry_record(**entry))
    return records


def test_detect_al_tracks_duration_and_severity() -> None:
    steady = _build_series(
        [
            {"lateral_accel": 0.2},
            {"lateral_accel": 0.3},
            {"lateral_accel": 2.1, "vertical_load": 5300.0},
            {"lateral_accel": 2.0, "vertical_load": 5450.0},
            {"lateral_accel": 1.9, "vertical_load": 5600.0},
            {"lateral_accel": 2.2, "vertical_load": 5700.0},
            {"lateral_accel": 0.5},
            {"lateral_accel": 0.3},
        ]
    )
    intense = _build_series(
        [
            {"lateral_accel": 0.2},
            {"lateral_accel": 0.3},
            {"lateral_accel": 2.8, "vertical_load": 5400.0},
            {"lateral_accel": 2.6, "vertical_load": 5600.0},
            {"lateral_accel": 2.9, "vertical_load": 5750.0},
            {"lateral_accel": 0.4},
        ]
    )

    events = detect_al(steady, window=3, lateral_threshold=1.5, load_threshold=200.0)
    assert len(events) == 1
    event = events[0]
    assert event["name"] == canonical_operator_label("AL")
    assert event["duration"] > 0.0

    stronger = detect_al(intense, window=3, lateral_threshold=1.5, load_threshold=200.0)
    assert len(stronger) == 1
    assert stronger[0]["severity"] > event["severity"]

    # Very short spikes should be ignored when they do not fill the window.
    short_window = _build_series(
        [
            {"lateral_accel": 2.5, "vertical_load": 5500.0},
            {"lateral_accel": 0.2},
        ]
    )
    assert detect_al(short_window, window=3, lateral_threshold=1.5, load_threshold=200.0) == []


def test_detect_oz_requires_slip_and_yaw_alignment() -> None:
    baseline = _build_series(
        [
            {"slip_angle": 0.05, "yaw_rate": 0.1},
            {"slip_angle": 0.06, "yaw_rate": 0.12},
            {"slip_angle": 0.08, "yaw_rate": 0.2},
            {"slip_angle": 0.05, "yaw_rate": 0.1},
        ]
    )
    oversteer = _build_series(
        [
            {"slip_angle": 0.05, "yaw_rate": 0.1},
            {"slip_angle": 0.18, "yaw_rate": 0.32},
            {"slip_angle": 0.2, "yaw_rate": 0.35},
            {"slip_angle": 0.22, "yaw_rate": 0.38},
            {"slip_angle": 0.07, "yaw_rate": 0.12},
        ]
    )

    assert detect_oz(baseline, window=3, slip_threshold=0.12, yaw_threshold=0.25) == []

    events = detect_oz(oversteer, window=3, slip_threshold=0.12, yaw_threshold=0.25)
    assert len(events) == 1
    event = events[0]
    assert event["name"] == canonical_operator_label("OZ")
    assert event["severity"] > 1.0

    milder = _build_series(
        [
            {"slip_angle": 0.05, "yaw_rate": 0.1},
            {"slip_angle": 0.16, "yaw_rate": 0.29},
            {"slip_angle": 0.15, "yaw_rate": 0.28},
            {"slip_angle": 0.07, "yaw_rate": 0.2},
        ]
    )
    weaker = detect_oz(milder, window=3, slip_threshold=0.12, yaw_threshold=0.25)
    assert weaker and weaker[0]["severity"] < event["severity"]


def test_detect_il_uses_speed_weighted_threshold() -> None:
    slow_series = _build_series(
        [
            {"speed": 10.0, "line_deviation": 0.45},
            {"speed": 10.0, "line_deviation": 0.46},
            {"speed": 10.0, "line_deviation": 0.47},
            {"speed": 10.0, "line_deviation": 0.2},
        ]
    )
    fast_series = _build_series(
        [
            {"speed": 40.0, "line_deviation": 0.3},
            {"speed": 40.0, "line_deviation": 1.02},
            {"speed": 40.0, "line_deviation": 1.05},
            {"speed": 40.0, "line_deviation": 1.1},
            {"speed": 40.0, "line_deviation": 0.4},
        ]
    )

    slow_events = detect_il(slow_series, window=3, base_threshold=0.3, speed_gain=0.015)
    assert len(slow_events) == 1
    fast_events = detect_il(fast_series, window=3, base_threshold=0.3, speed_gain=0.015)
    assert len(fast_events) == 1
    assert fast_events[0]["severity"] > slow_events[0]["severity"]

    below_threshold = _build_series(
        [
            {"speed": 25.0, "line_deviation": 0.2},
            {"speed": 25.0, "line_deviation": 0.25},
            {"speed": 25.0, "line_deviation": 0.24},
        ]
    )
    assert detect_il(below_threshold, window=3, base_threshold=0.3, speed_gain=0.02) == []


def test_detect_silence_flags_quiet_structural_intervals() -> None:
    quiet_series = _build_series(
        [
            {
                "lateral_accel": 1.25,
                "longitudinal_accel": 0.05,
                "vertical_load": 4800.0,
                "nfr": 102.0,
                "brake_pressure": 0.04,
                "throttle": 0.18,
                "yaw_rate": 0.02,
                "steer": 0.03,
            }
            for _ in range(16)
        ]
    )
    events = detect_silence(
        quiet_series,
        window=8,
        load_threshold=150.0,
        accel_threshold=0.85,
        delta_nfr_threshold=5.0,
        structural_density_threshold=0.05,
        min_duration=0.4,
    )
    assert events
    event = events[0]
    assert event["name"] == "Structural silence"
    assert event["duration"] >= 0.4
    assert event["structural_duration"] >= event["duration"]
    assert event["load_span"] <= 150.0
    assert event["structural_density_mean"] <= 0.05
    assert event["slack"] > 0.0

    noisy_series = _build_series(
        [
            {
                "lateral_accel": 2.2,
                "longitudinal_accel": 0.4,
                "vertical_load": 5200.0 + (index * 50.0),
                "nfr": 120.0 + index * 3.0,
                "throttle": 0.6,
                "yaw_rate": 0.12,
                "steer": 0.2,
            }
            for index in range(16)
        ]
    )
    assert (
        detect_silence(
            noisy_series,
            window=8,
            load_threshold=150.0,
            accel_threshold=0.85,
            delta_nfr_threshold=5.0,
            structural_density_threshold=0.05,
            min_duration=0.4,
        )
        == []
    )


@pytest.mark.parametrize(
    ("func", "alias", "expected"),
    [
        (normalize_structural_operator_identifier, "silence", "SILENCE"),
        (normalize_structural_operator_identifier, "SILENCIO", "SILENCE"),
        (normalize_structural_operator_identifier, "silencio", "SILENCE"),
        (canonical_operator_label, "silencio", "Structural silence"),
    ],
)
def test_structural_operator_aliases(
    func, alias: str, expected: str
) -> None:
    assert func(alias) == expected


def test_silence_event_payloads_accepts_case_insensitive_identifier() -> None:
    payload = {"duration": 1.2}
    events = {"silence": (payload,)}
    result = silence_event_payloads(events)
    assert result == (payload,)


def test_silence_event_payloads_accepts_spanish_alias() -> None:
    payload = {"duration": 0.6}
    events = {"SILENCIO": payload}
    result = silence_event_payloads(events)
    assert result == (payload,)
