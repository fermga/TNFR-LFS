from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_label_check(path: Path, raf_root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_path = PROJECT_ROOT / "src"
    existing = env.get("PYTHONPATH")
    if existing:
        env["PYTHONPATH"] = os.pathsep.join([str(src_path), existing])
    else:
        env["PYTHONPATH"] = str(src_path)
    return subprocess.run(
        [sys.executable, "-m", "tools.label_check", str(path), "--raf-root", str(raf_root)],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=env,
    )


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
                                    {"start": 0.1, "end": 0.4},
                                    {"start": 0.4, "end": 0.7},
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
                                "intervals": [{"start": 0.3, "end": 0.3}],
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
