from __future__ import annotations

import json
from pathlib import Path

from dataclasses import dataclass

from tools.calibrate_detectors import (
    EvaluationResult,
    LabelledMicrosector,
    MicrosectorSample,
    ParameterSet,
    _evaluate_detector,
    _materialise_best_params,
    _load_labels,
    normalize_structural_operator_identifier,
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
                    "microsectors": {
                        "3": {"operators": {"operator.gamma": "true"}}
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

    assert labels[0].raf_path == (raf_root / "capture_a.raf").resolve()
    assert labels[2].raf_path == (raf_root / "capture_b.raf").resolve()


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


def test_interval_matching_overlap(monkeypatch) -> None:
    operator_id = "NAV"

    def stub_detector(window, **_kwargs):  # type: ignore[no-untyped-def]
        return [{"start_time": 1.0, "end_time": 4.0}]

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
    assert result.tp == 1
    assert result.fp == 0
    assert result.fn == 0
    assert result.tn == 0
    assert result.support == 1


def test_interval_matching_non_overlap(monkeypatch) -> None:
    operator_id = "NAV"

    def stub_detector(window, **_kwargs):  # type: ignore[no-untyped-def]
        return [{"start_time": 6.0, "end_time": 8.0}]

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
    assert result.tp == 0
    assert result.fp == 1
    assert result.fn == 1
    assert result.tn == 0
    assert result.support == 1


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
