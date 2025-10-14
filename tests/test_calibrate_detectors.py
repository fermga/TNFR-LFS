from __future__ import annotations

import json
from pathlib import Path

from tools.calibrate_detectors import (
    EvaluationResult,
    LabelledMicrosector,
    ParameterSet,
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
