from __future__ import annotations

import json
from pathlib import Path

from tools.calibrate_detectors import (
    LabelledMicrosector,
    _load_labels,
    normalize_structural_operator_identifier,
)


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
