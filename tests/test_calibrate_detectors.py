from __future__ import annotations

import json
import argparse
from pathlib import Path
from types import SimpleNamespace

from dataclasses import dataclass

import pytest

from tools.calibrate_detectors import (
    EvaluationResult,
    LabelledMicrosector,
    MicrosectorSample,
    ParameterSet,
    _evaluate_detector,
    _filter_samples,
    _materialise_best_params,
    _load_labels,
    _normalise_car_identifier,
    _normalise_compound_token,
    _normalise_operator_labels,
    _normalise_track_identifier,
    normalize_structural_operator_identifier,
    _group_combinations,
    calibrate_detectors,
)
from tnfr_core.config.loader import load_detection_config
from tnfr_core.operators import operator_detection


def test_load_labels_jsonl(tmp_path: Path) -> None:
    raf_root = tmp_path / "raf"
    raf_root.mkdir()

    jsonl_path = tmp_path / "labels.jsonl"
    entries = [
        {
            "id": "capture-a",
            "raf": "capture_a.raf",
            "track": "Kyoto National ",
            "car": "xFg",
            "car_class": "Std",
            "compound": "R2",
            "microsectors": [
                {"index": 1, "operators": {"operator.alpha": True}},
                {"index": 2, "operators": ["operator.beta"]},
            ],
        },
        {
            "captures": [
                {
                    "id": "capture-b",
                    "raf": "capture_b.raf",
                    "track": "South City Classic",
                    "car": "fz5",
                    "car_class": "lrf",
                    "compound": "Road Super",
                    "microsectors": {
                        "3": {
                            "operators": {"operator.gamma": "true"},
                        }
                    },
                }
            ]
        },
    ]

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")

    labels = _load_labels(jsonl_path, raf_root=raf_root)

    assert [type(item) for item in labels] == [LabelledMicrosector] * 3
    assert [label.capture_id for label in labels] == [
        "capture-a",
        "capture-a",
        "capture-b",
    ]
    assert [label.microsector_index for label in labels] == [1, 2, 3]

    operator_alpha = normalize_structural_operator_identifier("operator.alpha")
    operator_beta = normalize_structural_operator_identifier("operator.beta")
    operator_gamma = normalize_structural_operator_identifier("operator.gamma")

    assert labels[0].operators == {operator_alpha: True}
    assert labels[1].operators == {operator_beta: True}
    assert labels[2].operators == {operator_gamma: True}
    assert labels[0].operator_intervals == {operator_alpha: ()}
    assert labels[1].operator_intervals == {operator_beta: ()}
    assert labels[2].operator_intervals == {operator_gamma: ()}

    assert labels[0].track == "KY2"
    assert labels[0].car == "XFG"
    assert labels[0].car_class == "STD"
    assert labels[0].compound == "r2"
    assert labels[1].track == "KY2"
    assert labels[1].car == "XFG"
    assert labels[1].car_class == "STD"
    assert labels[1].compound == "r2"
    assert labels[2].track == "SO1"
    assert labels[2].car == "FZ5"
    assert labels[2].car_class == "LRF"
    assert labels[2].compound == "road_super"

    assert labels[0].raf_path == (raf_root / "capture_a.raf").resolve()
    assert labels[2].raf_path == (raf_root / "capture_b.raf").resolve()


def test_normalise_operator_labels_mapping_string_values() -> None:
    operator_alpha = normalize_structural_operator_identifier("operator.alpha")
    operator_beta = normalize_structural_operator_identifier("operator.beta")
    operator_gamma = normalize_structural_operator_identifier("operator.gamma")
    operator_delta = normalize_structural_operator_identifier("operator.delta")

    payload = {
        "operator.alpha": {"active": "false"},
        "operator.beta": {"label": "0"},
        "operator.gamma": {"positive": "no"},
        "operator.delta": {"active": " true "},
    }

    entries = _normalise_operator_labels(payload)

    assert entries == {
        operator_alpha: False,
        operator_beta: False,
        operator_gamma: False,
        operator_delta: True,
    }


def test_filter_samples_case_insensitive() -> None:
    sample = MicrosectorSample(
        capture_id="capture-a",
        microsector_index=1,
        track="KY2",
        car="XFG",
        car_class="STD",
        compound="r2",
        start_index=0,
        end_index=10,
        start_time=0.0,
        end_time=1.0,
        records=(),
        labels={},
        label_intervals={},
    )

    filtered = _filter_samples(
        [sample],
        cars={_normalise_car_identifier("xf gti")},
        compounds={_normalise_compound_token("R2")},
        tracks={_normalise_track_identifier("Kyoto National")},
    )

    assert filtered == [sample]


def test_materialise_best_params_integration(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "best_params.yaml"
    selections = [
        EvaluationResult(
            operator_id="NAV",
            combination=("__default__", "__unknown__", "__default__"),
            parameter_set=ParameterSet(
                identifier="default",
                parameters={"window": 2},
            ),
            precision=1.0,
            recall=1.0,
            f1=1.0,
            fp_per_minute=0.0,
            support=1,
            tp=1,
            fp=0,
            tn=0,
            fn=0,
            duration_minutes=1.0,
        ),
        EvaluationResult(
            operator_id="NAV",
            combination=("GT3", "__unknown__", "r2"),
            parameter_set=ParameterSet(
                identifier="class-compound",
                parameters={"window": 5},
            ),
            precision=1.0,
            recall=1.0,
            f1=1.0,
            fp_per_minute=0.0,
            support=1,
            tp=1,
            fp=0,
            tn=0,
            fn=0,
            duration_minutes=1.0,
        ),
        EvaluationResult(
            operator_id="NAV",
            combination=("__default__", "XFG", "__default__"),
            parameter_set=ParameterSet(
                identifier="car-default",
                parameters={"window": 7},
            ),
            precision=1.0,
            recall=1.0,
            f1=1.0,
            fp_per_minute=0.0,
            support=1,
            tp=1,
            fp=0,
            tn=0,
            fn=0,
            duration_minutes=1.0,
        ),
        EvaluationResult(
            operator_id="NAV",
            combination=("__default__", "__unknown__", "r3"),
            parameter_set=ParameterSet(
                identifier="compound",
                parameters={"window": 4},
            ),
            precision=1.0,
            recall=1.0,
            f1=1.0,
            fp_per_minute=0.0,
            support=1,
            tp=1,
            fp=0,
            tn=0,
            fn=0,
            duration_minutes=1.0,
        ),
    ]

    _materialise_best_params(selections, output_path=output_path)

    config = load_detection_config(path=output_path)
    nav_config = config.get("detect_nav")
    assert isinstance(nav_config, dict)

    assert nav_config["defaults"] == {"window": 2}
    assert nav_config["classes"]["GT3"]["compounds"]["r2"] == {"window": 5}
    assert nav_config["cars"]["XFG"]["defaults"] == {"window": 7}
    assert nav_config["compounds"]["r3"] == {"window": 4}

    operator_detection._load_detection_table.cache_clear()
    try:
        baseline_events = operator_detection.detect_nav([0.0, 0.0], nu_f=0.0)
        assert baseline_events == []

        monkeypatch.setattr(operator_detection, "load_detection_config", lambda: config)
        operator_detection._load_detection_table.cache_clear()

        overridden_events = operator_detection.detect_nav([0.0, 0.0], nu_f=0.0)
        assert overridden_events and overridden_events[0]["duration"] == 2
    finally:
        operator_detection._load_detection_table.cache_clear()


@dataclass
class _StubRecord:
    timestamp: float


def _make_sample(*, car: str, compound: str | None) -> MicrosectorSample:
    return MicrosectorSample(
        capture_id="capture",
        microsector_index=1,
        track="track",
        car=car,
        car_class=None,
        compound=compound,
        start_index=0,
        end_index=0,
        start_time=0.0,
        end_time=0.0,
        records=(),
        labels={},
        label_intervals={},
    )


def test_nav_evaluation_uses_delta_series(monkeypatch) -> None:
    operator_id = "NAV"
    delta_series = [0.5, 0.5, 0.5, 0.5]
    expected_nu_f = 0.5
    captured: dict[str, object] = {}

    def stub_detector(series, *, nu_f, window, metadata=None, **_kwargs):  # type: ignore[no-untyped-def]
        captured["series"] = list(series)
        captured["nu_f"] = nu_f
        captured["metadata"] = metadata
        if all(abs(float(value) - expected_nu_f) < 1e-6 for value in series):
            events = [{"start_index": 0, "end_index": len(series) - 1}]
            captured["events"] = events
            return events
        captured["events"] = []
        return []

    monkeypatch.setattr(
        "tools.calibrate_detectors._detector_callable", lambda _: stub_detector
    )

    records = [_StubRecord(float(index)) for index in range(len(delta_series))]
    interval_end = float(len(delta_series))
    sample = MicrosectorSample(
        capture_id="capture",
        microsector_index=1,
        track="track",
        car="car",
        car_class=None,
        compound=None,
        start_index=0,
        end_index=len(delta_series) - 1,
        start_time=0.0,
        end_time=interval_end,
        records=records,
        delta_nfr_series=tuple(delta_series),
        nav_nu_f=expected_nu_f,
        labels={operator_id: True},
        label_intervals={operator_id: ((0.0, interval_end),)},
    )

    result = _evaluate_detector(
        operator_id,
        [sample],
        ParameterSet(identifier="nav", parameters={"window": 3}),
        fold_assignments={},
    )

    assert result is not None
    assert captured["series"] == delta_series
    assert captured["nu_f"] == expected_nu_f
    assert captured["events"]
    assert captured["metadata"] == list(records)


def test_detect_nav_metadata_overrides(monkeypatch) -> None:
    series = [1.0] * 5
    metadata = [
        SimpleNamespace(
            car_class="GT3",
            car_model="XFG",
            track_name="kyoto",
            tyre_compound="r2",
        )
    ]
    config = {
        "detect_nav": {
            "defaults": {"window": 3},
            "classes": {
                "GT3": {
                    "defaults": {"window": 6},
                }
            },
        }
    }

    operator_detection._load_detection_table.cache_clear()
    monkeypatch.setattr(operator_detection, "load_detection_config", lambda: config)
    operator_detection._load_detection_table.cache_clear()

    try:
        baseline = operator_detection.detect_nav(series, nu_f=1.0)
        assert baseline  # window=3 should yield an event

        overridden = operator_detection.detect_nav(series, nu_f=1.0, metadata=metadata)
        assert overridden == []
    finally:
        operator_detection._load_detection_table.cache_clear()


@pytest.mark.parametrize(
    "detector_events, expected",
    [
        pytest.param(
            [{"start_time": 1.0, "end_time": 4.0}],
            {"tp": 1, "fp": 0, "fn": 0, "tn": 0, "support": 1},
            id="overlap generates true positive",
        ),
        pytest.param(
            [{"start_time": 6.0, "end_time": 8.0}],
            {"tp": 0, "fp": 1, "fn": 1, "tn": 0, "support": 1},
            id="non-overlap registers false positive",
        ),
    ],
)
def test_interval_matching_outcomes(monkeypatch, detector_events, expected) -> None:
    operator_id = "NAV"

    def stub_detector(window, **_kwargs):  # type: ignore[no-untyped-def]
        return detector_events

    monkeypatch.setattr(
        "tools.calibrate_detectors._detector_callable", lambda _: stub_detector
    )

    records = [_StubRecord(timestamp) for timestamp in (0.0, 5.0, 10.0)]
    sample = MicrosectorSample(
        capture_id="capture",
        microsector_index=1,
        track="track",
        car="car",
        car_class=None,
        compound=None,
        start_index=0,
        end_index=2,
        start_time=0.0,
        end_time=10.0,
        records=records,
        labels={operator_id: True},
        label_intervals={operator_id: ((0.0, 5.0),)},
    )

    result = _evaluate_detector(
        operator_id,
        [sample],
        ParameterSet(identifier="stub", parameters={}),
        fold_assignments={},
    )

    assert result is not None
    for metric, expected_value in expected.items():
        assert getattr(result, metric) == expected_value


def test_interval_matching_negative_sample(monkeypatch) -> None:
    operator_id = "NAV"

    def stub_detector(window, **_kwargs):  # type: ignore[no-untyped-def]
        return []

    monkeypatch.setattr(
        "tools.calibrate_detectors._detector_callable", lambda _: stub_detector
    )

    records = [_StubRecord(timestamp) for timestamp in (0.0, 5.0, 10.0)]
    sample = MicrosectorSample(
        capture_id="capture",
        microsector_index=1,
        track="track",
        car="car",
        car_class=None,
        compound=None,
        start_index=0,
        end_index=2,
        start_time=0.0,
        end_time=10.0,
        records=records,
        labels={},
        label_intervals={},
    )

    result = _evaluate_detector(
        operator_id,
        [sample],
        ParameterSet(identifier="stub", parameters={}),
        fold_assignments={},
    )

    assert result is not None
    assert result.tp == 0
    assert result.fp == 0
    assert result.fn == 0
    assert result.tn == 1
    assert result.support == 0


def test_group_combinations_accepts_known_compounds() -> None:
    sample = _make_sample(car="XFG", compound="R2")

    grouped, invalid = _group_combinations([sample])

    assert grouped[("__default__", "XFG", "R2")] == [sample]
    assert invalid == set()


def test_group_combinations_collects_invalid_pairs() -> None:
    sample = _make_sample(car="XFG", compound="soft")

    grouped, invalid = _group_combinations([sample])

    assert grouped[("__default__", "XFG", "soft")] == [sample]
    assert invalid == {("XFG", "soft")}


def test_calibrate_detectors_aborts_on_invalid_pairs(tmp_path: Path, monkeypatch) -> None:
    sample = _make_sample(car="XFG", compound="soft")

    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text("{}\n", encoding="utf-8")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    args = argparse.Namespace(
        raf_root=str(tmp_path),
        labels=str(labels_path),
        out=str(output_dir),
        operators=["NAV"],
        operator_grid=None,
        cars=[],
        compounds=[],
        tracks=[],
        kfold=1,
        fp_per_min_max=0.5,
    )

    monkeypatch.setattr(
        "tools.calibrate_detectors._load_labels",
        lambda _path, *, raf_root: [object()],
    )
    monkeypatch.setattr(
        "tools.calibrate_detectors._build_microsector_dataset",
        lambda _labels, *, raf_root: [sample],
    )

    with pytest.raises(SystemExit) as excinfo:
        calibrate_detectors(args)

    message = str(excinfo.value)
    assert "XFG" in message
    assert "soft" in message
