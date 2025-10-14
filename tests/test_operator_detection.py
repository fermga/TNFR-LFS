"""Tests for the operator detection utilities."""

from __future__ import annotations

import importlib
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Callable, List, Mapping, Sequence

from _pytest.mark.structures import ParameterSet

import pytest

from tnfr_core.epi import TelemetryRecord
from tnfr_core.operator_detection import (
    canonical_operator_label,
    detect_al,
    detect_il,
    detect_oz,
    detect_silence,
    detect_nav,
    normalize_structural_operator_identifier,
    silence_event_payloads,
)

from tests.helpers import build_telemetry_record


OperatorVerifier = Callable[[List[dict], Callable[..., List[dict]]], None]


@dataclass(frozen=True)
class OperatorCase:
    series: List[TelemetryRecord]
    kwargs: dict
    expected_count: int
    verifiers: Sequence[OperatorVerifier] = ()


def _run_operator_case(
    func: Callable[..., List[dict]],
    case: OperatorCase,
) -> None:
    events = func(case.series, **case.kwargs)
    if case.expected_count == 0:
        assert events == []
    else:
        assert len(events) == case.expected_count
    for verifier in case.verifiers:
        verifier(events, func)


def _expect_event_name(name: str) -> OperatorVerifier:
    def _verify(events: List[dict], _: Callable[..., List[dict]]) -> None:
        assert events[0]["name"] == name

    return _verify


def _expect_duration_positive(events: List[dict], _: Callable[..., List[dict]]) -> None:
    assert events[0]["duration"] > 0.0


def _expect_duration_at_least(minimum: float) -> OperatorVerifier:
    def _verify(events: List[dict], _: Callable[..., List[dict]]) -> None:
        assert events[0]["duration"] >= minimum

    return _verify


def _expect_field_at_most(field: str, threshold: float) -> OperatorVerifier:
    def _verify(events: List[dict], _: Callable[..., List[dict]]) -> None:
        assert events[0][field] <= threshold

    return _verify


def _expect_field_at_least(field: str, threshold: float) -> OperatorVerifier:
    def _verify(events: List[dict], _: Callable[..., List[dict]]) -> None:
        assert events[0][field] >= threshold

    return _verify


def _expect_field_greater_than(field: str, threshold: float) -> OperatorVerifier:
    def _verify(events: List[dict], _: Callable[..., List[dict]]) -> None:
        assert events[0][field] > threshold

    return _verify


def _expect_severity_above(threshold: float) -> OperatorVerifier:
    return _expect_field_greater_than("severity", threshold)


def _expect_severity_comparison(
    reference_series: List[TelemetryRecord],
    reference_kwargs: dict,
    comparator: Callable[[float, float], bool],
) -> OperatorVerifier:
    def _verify(events: List[dict], func: Callable[..., List[dict]]) -> None:
        reference_events = func(reference_series, **reference_kwargs)
        assert reference_events
        assert comparator(events[0]["severity"], reference_events[0]["severity"])

    return _verify


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


_AL_COMMON_KWARGS = dict(window=3, lateral_threshold=1.5, load_threshold=200.0)
_AL_STEADY_SAMPLES = [
    {"lateral_accel": 0.2},
    {"lateral_accel": 0.3},
    {"lateral_accel": 2.1, "vertical_load": 5300.0},
    {"lateral_accel": 2.0, "vertical_load": 5450.0},
    {"lateral_accel": 1.9, "vertical_load": 5600.0},
    {"lateral_accel": 2.2, "vertical_load": 5700.0},
    {"lateral_accel": 0.5},
    {"lateral_accel": 0.3},
]
_AL_STEADY_SERIES = _build_series(_AL_STEADY_SAMPLES)
_AL_INTENSE_SAMPLES = [
    {"lateral_accel": 0.2},
    {"lateral_accel": 0.3},
    {"lateral_accel": 2.8, "vertical_load": 5400.0},
    {"lateral_accel": 2.6, "vertical_load": 5600.0},
    {"lateral_accel": 2.9, "vertical_load": 5750.0},
    {"lateral_accel": 0.4},
]
_AL_INTENSE_SERIES = _build_series(_AL_INTENSE_SAMPLES)
_AL_SHORT_WINDOW_SAMPLES = [
    {"lateral_accel": 2.5, "vertical_load": 5500.0},
    {"lateral_accel": 0.2},
]
_AL_SHORT_WINDOW_SERIES = _build_series(_AL_SHORT_WINDOW_SAMPLES)


_AL_CASES = [
    pytest.param(
        OperatorCase(
            series=_AL_STEADY_SERIES,
            kwargs=_AL_COMMON_KWARGS,
            expected_count=1,
            verifiers=(
                _expect_event_name(canonical_operator_label("AL")),
                _expect_duration_positive,
            ),
        ),
        id="steady-turn-produces-operator",
    ),
    pytest.param(
        OperatorCase(
            series=_AL_INTENSE_SERIES,
            kwargs=_AL_COMMON_KWARGS,
            expected_count=1,
            verifiers=(
                _expect_severity_comparison(
                    reference_series=_AL_STEADY_SERIES,
                    reference_kwargs=_AL_COMMON_KWARGS,
                    comparator=lambda current, reference: current > reference,
                ),
            ),
        ),
        id="intense-turn-has-greater-severity",
    ),
    # Very short spikes should be ignored when they do not fill the window.
    pytest.param(
        OperatorCase(
            series=_AL_SHORT_WINDOW_SERIES,
            kwargs=_AL_COMMON_KWARGS,
            expected_count=0,
        ),
        id="short-window-ignored",
    ),
]


_OZ_COMMON_KWARGS = dict(window=3, slip_threshold=0.12, yaw_threshold=0.25)
_OZ_BASELINE_SAMPLES = [
    {"slip_angle": 0.05, "yaw_rate": 0.1},
    {"slip_angle": 0.06, "yaw_rate": 0.12},
    {"slip_angle": 0.08, "yaw_rate": 0.2},
    {"slip_angle": 0.05, "yaw_rate": 0.1},
]
_OZ_BASELINE_SERIES = _build_series(_OZ_BASELINE_SAMPLES)
_OZ_OVERSTEER_SAMPLES = [
    {"slip_angle": 0.05, "yaw_rate": 0.1},
    {"slip_angle": 0.18, "yaw_rate": 0.32},
    {"slip_angle": 0.2, "yaw_rate": 0.35},
    {"slip_angle": 0.22, "yaw_rate": 0.38},
    {"slip_angle": 0.07, "yaw_rate": 0.12},
]
_OZ_OVERSTEER_SERIES = _build_series(_OZ_OVERSTEER_SAMPLES)
_OZ_MILDER_SAMPLES = [
    {"slip_angle": 0.05, "yaw_rate": 0.1},
    {"slip_angle": 0.16, "yaw_rate": 0.29},
    {"slip_angle": 0.15, "yaw_rate": 0.28},
    {"slip_angle": 0.07, "yaw_rate": 0.2},
]
_OZ_MILDER_SERIES = _build_series(_OZ_MILDER_SAMPLES)


_OZ_CASES = [
    pytest.param(
        OperatorCase(
            series=_OZ_BASELINE_SERIES,
            kwargs=_OZ_COMMON_KWARGS,
            expected_count=0,
        ),
        id="baseline-no-oversteer",
    ),
    pytest.param(
        OperatorCase(
            series=_OZ_OVERSTEER_SERIES,
            kwargs=_OZ_COMMON_KWARGS,
            expected_count=1,
            verifiers=(
                _expect_event_name(canonical_operator_label("OZ")),
                _expect_severity_above(1.0),
            ),
        ),
        id="aligned-slip-and-yaw",
    ),
    pytest.param(
        OperatorCase(
            series=_OZ_MILDER_SERIES,
            kwargs=_OZ_COMMON_KWARGS,
            expected_count=1,
            verifiers=(
                _expect_severity_comparison(
                    reference_series=_OZ_OVERSTEER_SERIES,
                    reference_kwargs=_OZ_COMMON_KWARGS,
                    comparator=lambda current, reference: current < reference,
                ),
            ),
        ),
        id="milder-oversteer-lower-severity",
    ),
]


_IL_COMMON_KWARGS = dict(window=3, base_threshold=0.3, speed_gain=0.015)
_IL_SLOW_SAMPLES = [
    {"speed": 10.0, "line_deviation": 0.45},
    {"speed": 10.0, "line_deviation": 0.46},
    {"speed": 10.0, "line_deviation": 0.47},
    {"speed": 10.0, "line_deviation": 0.2},
]
_IL_SLOW_SERIES = _build_series(_IL_SLOW_SAMPLES)
_IL_FAST_SAMPLES = [
    {"speed": 40.0, "line_deviation": 0.3},
    {"speed": 40.0, "line_deviation": 1.02},
    {"speed": 40.0, "line_deviation": 1.05},
    {"speed": 40.0, "line_deviation": 1.1},
    {"speed": 40.0, "line_deviation": 0.4},
]
_IL_FAST_SERIES = _build_series(_IL_FAST_SAMPLES)
_IL_BELOW_THRESHOLD_SAMPLES = [
    {"speed": 25.0, "line_deviation": 0.2},
    {"speed": 25.0, "line_deviation": 0.25},
    {"speed": 25.0, "line_deviation": 0.24},
]
_IL_BELOW_THRESHOLD_SERIES = _build_series(_IL_BELOW_THRESHOLD_SAMPLES)


_IL_CASES = [
    pytest.param(
        OperatorCase(
            series=_IL_SLOW_SERIES,
            kwargs=_IL_COMMON_KWARGS,
            expected_count=1,
        ),
        id="slow-line-deviation",
    ),
    pytest.param(
        OperatorCase(
            series=_IL_FAST_SERIES,
            kwargs=_IL_COMMON_KWARGS,
            expected_count=1,
            verifiers=(
                _expect_severity_comparison(
                    reference_series=_IL_SLOW_SERIES,
                    reference_kwargs=_IL_COMMON_KWARGS,
                    comparator=lambda current, reference: current > reference,
                ),
            ),
        ),
        id="faster-speed-higher-severity",
    ),
    pytest.param(
        OperatorCase(
            series=_IL_BELOW_THRESHOLD_SERIES,
            kwargs=dict(window=3, base_threshold=0.3, speed_gain=0.02),
            expected_count=0,
        ),
        id="below-threshold-ignored",
    ),
]


_SILENCE_COMMON_KWARGS = dict(
    window=8,
    load_threshold=150.0,
    accel_threshold=0.85,
    delta_nfr_threshold=5.0,
    structural_density_threshold=0.05,
    min_duration=0.4,
)
_SILENCE_QUIET_SAMPLES = [
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
_SILENCE_QUIET_SERIES = _build_series(_SILENCE_QUIET_SAMPLES)
_SILENCE_NOISY_SAMPLES = [
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
_SILENCE_NOISY_SERIES = _build_series(_SILENCE_NOISY_SAMPLES)

_THOL_BASELINE_SAMPLES = [
    {
        "lateral_accel": 0.2,
        "longitudinal_accel": 0.05,
        "vertical_load": 5000.0,
        "slip_angle": 0.04,
        "yaw_rate": 0.08,
        "line_deviation": 0.05,
        "speed": 22.0,
        "nfr": 100.0,
        "throttle": 0.2,
        "brake_pressure": 0.05,
    }
    for _ in range(6)
]


def _stack_samples(*segments: Sequence[dict]) -> List[TelemetryRecord]:
    combined: List[dict] = []
    timestamp_index = 0
    for segment in segments:
        for payload in segment:
            entry = dict(payload)
            entry["timestamp"] = timestamp_index * 0.1
            combined.append(entry)
            timestamp_index += 1
    return _build_series(combined)


def _build_thol_series() -> List[TelemetryRecord]:
    return _stack_samples(
        _THOL_BASELINE_SAMPLES,
        _AL_STEADY_SAMPLES,
        _THOL_BASELINE_SAMPLES,
        _OZ_OVERSTEER_SAMPLES,
        _THOL_BASELINE_SAMPLES,
        _IL_FAST_SAMPLES,
        _THOL_BASELINE_SAMPLES,
        _SILENCE_QUIET_SAMPLES,
    )


def _resolve_detect_thol() -> Callable[[Sequence[TelemetryRecord]], SequenceABC[Mapping[str, object]]]:
    module_candidates = (
        "tnfr_core.thol_detection",
        "tnfr_core.thol",
        "tnfr_core.operator_detection_thol",
        "tnfr_lfs.plugins.thol",
        "tnfr_lfs.tools.thol",
    )
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        detector = getattr(module, "detect_thol", None)
        if callable(detector):
            return detector
    pytest.skip("THOL detector is not available in this environment")


def _collect_event_names(result: object) -> set[str]:
    names: set[str] = set()

    def _extract_items(value: object) -> List[object]:
        if isinstance(value, MappingABC):
            items: List[object] = []
            for payload in value.values():
                items.extend(_extract_items(payload))
            return items
        if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
            items = []
            for item in value:
                items.extend(_extract_items(item))
            return items
        return [value]

    for entry in _extract_items(result):
        if isinstance(entry, MappingABC):
            name = entry.get("name")
            if isinstance(name, str):
                names.add(name)

    return names


_SILENCE_CASES = [
    pytest.param(
        OperatorCase(
            series=_SILENCE_QUIET_SERIES,
            kwargs=_SILENCE_COMMON_KWARGS,
            expected_count=1,
            verifiers=(
                _expect_event_name("Structural silence"),
                _expect_duration_at_least(0.4),
                _expect_field_at_least("structural_duration", 0.4),
                _expect_field_at_most("load_span", 150.0),
                _expect_field_at_most("structural_density_mean", 0.05),
                _expect_field_greater_than("slack", 0.0),
            ),
        ),
        id="quiet-interval-detected",
    ),
    pytest.param(
        OperatorCase(
            series=_SILENCE_NOISY_SERIES,
            kwargs=_SILENCE_COMMON_KWARGS,
            expected_count=0,
        ),
        id="noisy-interval-ignored",
    ),
]


def _structural_operator_params() -> List[ParameterSet]:
    suites = (
        ("AL", detect_al, _AL_CASES),
        ("OZ", detect_oz, _OZ_CASES),
        ("IL", detect_il, _IL_CASES),
        ("SILENCE", detect_silence, _SILENCE_CASES),
    )
    parameters: List[ParameterSet] = []
    for operator_id, detector, cases in suites:
        for entry in cases:
            if isinstance(entry, ParameterSet):
                (case,) = entry.values
                case_id = entry.id or "case"
            else:  # pragma: no cover - defensive path for plain OperatorCase
                case = entry
                case_id = "case"
            parameters.append(
                pytest.param(
                    operator_id,
                    detector,
                    case,
                    id=f"{operator_id}:{case_id}",
                )
            )
    return parameters


@pytest.mark.parametrize(
    "operator_id, detector, case",
    _structural_operator_params(),
)
def test_structural_operator_behaviour(
    operator_id: str,
    detector: Callable[..., List[dict]],
    case: OperatorCase,
) -> None:
    _run_operator_case(detector, case)


_THOL_DETECTOR_CASES = [
    pytest.param(
        detect_al,
        _AL_COMMON_KWARGS,
        canonical_operator_label("AL"),
        id="al",
    ),
    pytest.param(
        detect_oz,
        _OZ_COMMON_KWARGS,
        canonical_operator_label("OZ"),
        id="oz",
    ),
    pytest.param(
        detect_il,
        _IL_COMMON_KWARGS,
        canonical_operator_label("IL"),
        id="il",
    ),
    pytest.param(
        detect_silence,
        _SILENCE_COMMON_KWARGS,
        canonical_operator_label("SILENCE"),
        id="silence",
    ),
]


@pytest.mark.parametrize("detector, kwargs, expected_name", _THOL_DETECTOR_CASES)
def test_thol_series_triggers_structural_detectors(
    detector: Callable[..., List[Mapping[str, object]]],
    kwargs: Mapping[str, object],
    expected_name: str,
) -> None:
    series = _build_thol_series()
    events = detector(series, **kwargs)
    assert any(event.get("name") == expected_name for event in events)


def test_detect_thol_aggregates_structural_events() -> None:
    detect_thol = _resolve_detect_thol()
    series = _build_thol_series()
    result = detect_thol(series)
    names = _collect_event_names(result)
    expected = {
        canonical_operator_label("AL"),
        canonical_operator_label("OZ"),
        canonical_operator_label("IL"),
        canonical_operator_label("SILENCE"),
    }
    assert expected <= names


def test_structural_operator_aliases() -> None:
    expected_labels = {
        "AL": "Support",
        "EN": "Reception",
        "IL": "Coherence",
        "OZ": "Dissonance",
        "UM": "Coupling",
        "RA": "Propagation",
        "SILENCE": "Structural silence",
        "VAL": "Amplification",
        "NUL": "Contraction",
        "THOL": "Auto-organisation",
        "ZHIR": "Transformation",
        "NAV": "Transition",
        "REMESH": "Remeshing",
    }

    assert len(expected_labels) == 13

    for identifier, label in expected_labels.items():
        assert canonical_operator_label(identifier) == label

    alias_expectations = [
        (normalize_structural_operator_identifier, "silence", "SILENCE"),
        (normalize_structural_operator_identifier, "SHA", "SILENCE"),
        (normalize_structural_operator_identifier, "AUTOORGANISATION", "THOL"),
        (normalize_structural_operator_identifier, "TRANSICIÓN", "NAV"),
        (canonical_operator_label, "silence", "Structural silence"),
        (canonical_operator_label, "SHA", "Structural silence"),
        (canonical_operator_label, "AUTO ORGANIZATION", "Auto-organisation"),
        (canonical_operator_label, "NA'V", "Transition"),
    ]

    for func, alias, expected in alias_expectations:
        assert func(alias) == expected


def _silence_tuple_event(payload: dict) -> dict:
    return {"silence": (payload,)}


def _silence_event(payload: dict) -> dict:
    return {"SILENCE": payload}


@pytest.mark.parametrize(
    ("events_factory", "duration"),
    [
        pytest.param(
            _silence_tuple_event,
            1.2,
            id="english-alias-case-insensitive",
        ),
        pytest.param(
            _silence_event,
            0.6,
            id="english-identifier",
        ),
    ],
)
def test_silence_event_payloads_accepts_aliases(
    events_factory, duration: float
) -> None:
    payload = {"duration": duration}
    events = events_factory(payload)
    result = silence_event_payloads(events)
    assert result == (payload,)


# ---------------------------------------------------------------------------
# Transition (NA'V) detection


@pytest.fixture
def nav_global_in_phase() -> tuple[float, Sequence[float]]:
    nu_f = 0.5
    series = [0.498, 0.502, 0.501, 0.499, 0.500, 0.501]
    return nu_f, series


@pytest.fixture
def nav_global_out_of_band() -> tuple[float, Sequence[float]]:
    nu_f = 0.5
    series = [0.45, 0.46, 0.44, 0.60, 0.62, 0.58]
    return nu_f, series


@pytest.fixture
def nav_per_node_in_phase() -> tuple[Mapping[str, float], Sequence[Mapping[str, float]]]:
    nu_map: Mapping[str, float] = {"front": 0.30, "rear": 0.42}
    series = [
        {"front": 0.301, "rear": 0.420},
        {"front": 0.300, "rear": 0.421},
        {"front": 0.299, "rear": 0.419},
        {"front": 0.301, "rear": 0.421},
    ]
    return nu_map, series


@pytest.fixture
def nav_per_node_out_of_band() -> tuple[Mapping[str, float], Sequence[Mapping[str, float]]]:
    nu_map: Mapping[str, float] = {"front": 0.30, "rear": 0.42}
    series = [
        {"front": 0.25, "rear": 0.36},
        {"front": 0.24, "rear": 0.35},
        {"front": 0.55, "rear": 0.50},
        {"front": 0.60, "rear": 0.20},
    ]
    return nu_map, series


@pytest.mark.parametrize("window,eps", [(3, 1e-2), (5, 1e-2)], ids=["w3", "w5"])
def test_detect_nav_global_positive(
    nav_global_in_phase: tuple[float, Sequence[float]], window: int, eps: float
) -> None:
    nu_f, series = nav_global_in_phase
    events = detect_nav(series, nu_f=nu_f, window=window, eps=eps)
    assert events
    top = max(events, key=lambda e: e["severity"])
    assert top["severity"] > 0.0
    assert top["duration"] >= window


@pytest.mark.parametrize(
    "fixture_name",
    ["nav_global_out_of_band", "nav_per_node_out_of_band"],
    ids=["global-series", "per-node-series"],
)
def test_detect_nav_negative(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    nu_target, series = request.getfixturevalue(fixture_name)
    events = detect_nav(series, nu_f=nu_target, window=3, eps=1e-2)
    assert events == [] or max(e["severity"] for e in events) == 0.0


@pytest.mark.parametrize("window,eps", [(3, 1e-2), (4, 1e-2)], ids=["w3", "w4"])
def test_detect_nav_per_node_positive(
    nav_per_node_in_phase: tuple[Mapping[str, float], Sequence[Mapping[str, float]]],
    window: int,
    eps: float,
) -> None:
    nu_map, series = nav_per_node_in_phase
    events = detect_nav(series, nu_f=nu_map, window=window, eps=eps)
    assert events
    assert max(e["severity"] for e in events) > 0.0
    assert max(e["duration"] for e in events) >= window




def test_transition_aliases_canonical() -> None:
    for alias in ("TRANSICION", "TRANSICIÓN", "NA'V", "NAV"):
        assert normalize_structural_operator_identifier(alias) == "NAV"
        assert canonical_operator_label(alias) == "Transition"


def test_detect_nav_coexists_with_existing_detectors() -> None:
    assert callable(detect_al)
    assert callable(detect_il)
    assert callable(detect_oz)
    assert callable(detect_silence)


@dataclass
class _SimpleILRecord:
    timestamp: float
    speed: float
    line_deviation: float
    car_model: str | None = None
    track_name: str | None = None
    tyre_compound: str | None = None
    structural_timestamp: float | None = None


def _configure_detection(monkeypatch: pytest.MonkeyPatch, config: Mapping[str, object]):
    import tnfr_core.operators.operator_detection as module

    monkeypatch.setattr(module, "load_detection_config", lambda: config)
    module._load_detection_table.cache_clear()
    return module


def _build_il_records() -> list[_SimpleILRecord]:
    values = [0.1, 0.12, 0.18, 0.5, 0.52, 0.15]
    records: list[_SimpleILRecord] = []
    for index, deviation in enumerate(values):
        records.append(
            _SimpleILRecord(
                timestamp=float(index),
                speed=50.0,
                line_deviation=deviation,
                car_model="XFG",
                track_name="BL1",
                tyre_compound="R2",
            )
        )
    return records


def test_detect_il_uses_detection_config_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _configure_detection(
        monkeypatch,
        {
            "detect_il": {
                "defaults": {
                    "window": 5,
                    "base_threshold": 0.35,
                    "speed_gain": 0.012,
                },
                "tracks": {
                    "BL1": {
                        "defaults": {"window": 3, "base_threshold": 0.2},
                    }
                },
                "cars": {
                    "XFG": {
                        "compounds": {
                            "R2": {"speed_gain": 0.005},
                        }
                    }
                },
            }
        },
    )

    events = module.detect_il(_build_il_records())
    module._load_detection_table.cache_clear()
    assert events, "expected IL event triggered by detection overrides"


def test_detect_il_explicit_kwargs_override_config(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _configure_detection(
        monkeypatch,
        {
            "detect_il": {
                "defaults": {
                    "window": 5,
                    "base_threshold": 0.35,
                    "speed_gain": 0.012,
                },
                "tracks": {
                    "BL1": {
                        "defaults": {"window": 3, "base_threshold": 0.2},
                    }
                },
                "cars": {
                    "XFG": {
                        "compounds": {
                            "R2": {"speed_gain": 0.005},
                        }
                    }
                },
            }
        },
    )

    events = module.detect_il(_build_il_records(), base_threshold=0.7)
    module._load_detection_table.cache_clear()
    assert not events
