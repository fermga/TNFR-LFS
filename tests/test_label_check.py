from __future__ import annotations

import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import tools.label_check as label_check


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_label_check(path: Path, raf_root: Path):
    stdout = StringIO()
    stderr = StringIO()
    root_logger = label_check.logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    for handler in previous_handlers:
        root_logger.removeHandler(handler)
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            exit_code = label_check.main([str(path), "--raf-root", str(raf_root)])
        finally:
            root_logger.handlers = previous_handlers
    return SimpleNamespace(returncode=exit_code, stdout=stdout.getvalue(), stderr=stderr.getvalue())


def test_label_check_reports_contiguous_coverage(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.json"
    payload = {
        "captures": [
            {
                "id": "capture-a",
                "raf": "capture_a.raf",
                "track": "Blackwood",
                "car": "XFG",
                "compound": "r2",
                "microsectors": [
                    {"index": 1, "operators": {"NAV": True}},
                    {"index": 2, "operators": {"NAV": True}},
                    {
                        "index": 3,
                        "operators": {
                            "NAV": {
                                "label": True,
                                "intervals": [
                                    {"start": 12.3, "end": 12.9},
                                    {"start": 13.1, "end": 13.8},
                                ],
                            }
                        },
                    },
                ],
            }
        ]
    }
    labels_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_label_check(labels_path, tmp_path)

    assert result.returncode == 0
    assert "Lap 1: indices 1-3 (3 microsectors)" in result.stdout
    assert "gaps: none" in result.stdout
    assert "overlaps: none" in result.stdout
    assert "ERROR" not in result.stderr


def test_label_check_flags_invalid_interval(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels_invalid.json"
    payload = {
        "captures": [
            {
                "id": "capture-b",
                "raf": "capture_b.raf",
                "microsectors": [
                    {
                        "index": 5,
                        "operators": {
                            "NAV": {
                                "label": True,
                                "intervals": [{"start": 2.0, "end": 1.5}],
                            }
                        },
                    }
                ],
            }
        ]
    }
    labels_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_label_check(labels_path, tmp_path)

    assert result.returncode == 1
    assert "invalid intervals" in result.stderr.lower()


def test_label_check_detects_overlaps_and_metadata_issues(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels_overlaps.json"
    payload = {
        "captures": [
            {
                "id": "capture-c",
                "raf": "capture_c.raf",
                "track": "Blackwood",
                "car": "XFG",
                "microsectors": [
                    {"index": 1, "operators": {"NAV": True}},
                    {"index": 1, "operators": {"NAV": True}},
                    {"index": 3, "operators": {"NAV": True}},
                ],
            },
            {
                "id": "capture-c",
                "raf": "capture_c.raf",
                "track": "FernBay",
                "car": "XRG",
                "microsectors": [
                    {"index": 4, "operators": {"NAV": True}},
                ],
            },
        ]
    }
    labels_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_label_check(labels_path, tmp_path)

    assert result.returncode == 1
    stderr = result.stderr
    assert "duplicate microsector indices" in stderr.lower()
    assert "missing microsector indices" in stderr.lower()
    assert "metadata" in stderr.lower()
