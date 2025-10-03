from __future__ import annotations

import json
from pathlib import Path

import pytest

from tnfr_lfs.cli.tnfr_lfs_cli import run_cli


def test_baseline_simulation_jsonl(
    tmp_path: Path, capsys, synthetic_stint_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    baseline_path = tmp_path / "baseline.jsonl"

    result = run_cli([
        "baseline",
        str(baseline_path),
        "--simulate",
        str(synthetic_stint_path),
    ])

    captured = capsys.readouterr()
    assert baseline_path.exists()
    lines = [line for line in baseline_path.read_text(encoding="utf8").splitlines() if line.strip()]
    assert len(lines) == 17
    assert "Baseline saved" in result
    assert str(baseline_path) in captured.out


def test_analyze_pipeline_json_export(
    tmp_path: Path,
    capsys,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    baseline_path = tmp_path / "baseline.jsonl"
    run_cli([
        "baseline",
        str(baseline_path),
        "--simulate",
        str(synthetic_stint_path),
    ])

    output = run_cli([
        "analyze",
        str(baseline_path),
        "--export",
        "json",
    ])

    captured = capsys.readouterr()
    payload = json.loads(output)
    assert payload["telemetry_samples"] == 17
    assert len(payload["microsectors"]) == 2
    assert "microsectors" in captured.out
    assert payload["phase_messages"]
    report_paths = payload["reports"]
    sense_path = Path(report_paths["sense_index_map"]["path"])
    spectrum_path = Path(report_paths["yaw_roll_spectrum"]["path"])
    assert sense_path.exists()
    assert spectrum_path.exists()
    assert payload["reports"]["sense_index_map"]["data"]


def test_suggest_pipeline(
    tmp_path: Path,
    capsys,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    baseline_path = tmp_path / "baseline.jsonl"
    run_cli([
        "baseline",
        str(baseline_path),
        "--simulate",
        str(synthetic_stint_path),
    ])

    output = run_cli([
        "suggest",
        str(baseline_path),
        "--export",
        "json",
        "--car-model",
        "generic_gt",
    ])

    payload = json.loads(output)
    assert "recommendations" in payload
    assert isinstance(payload["recommendations"], list)
    assert payload["phase_messages"]
    assert Path(payload["reports"]["yaw_roll_spectrum"]["path"]).exists()


def test_report_generation(
    tmp_path: Path,
    capsys,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    baseline_path = tmp_path / "baseline.jsonl"
    run_cli([
        "baseline",
        str(baseline_path),
        "--simulate",
        str(synthetic_stint_path),
    ])

    output = run_cli([
        "report",
        str(baseline_path),
        "--export",
        "json",
    ])

    payload = json.loads(output)
    assert "delta_nfr" in payload
    assert "sense_index" in payload
    assert Path(payload["reports"]["sense_index_map"]["path"]).exists()


def test_write_set_markdown_export(
    tmp_path: Path,
    capsys,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    baseline_path = tmp_path / "baseline.jsonl"
    run_cli([
        "baseline",
        str(baseline_path),
        "--simulate",
        str(synthetic_stint_path),
    ])

    output = run_cli([
        "write-set",
        str(baseline_path),
        "--export",
        "markdown",
        "--car-model",
        "generic_gt",
        "--session",
        "stint-1",
    ])

    assert "| Cambio |" in output


def test_cli_end_to_end_pipeline(
    tmp_path: Path,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    baseline_path = tmp_path / "baseline.jsonl"

    run_cli([
        "baseline",
        str(baseline_path),
        "--simulate",
        str(synthetic_stint_path),
    ])

    analysis = json.loads(
        run_cli([
            "analyze",
            str(baseline_path),
            "--export",
            "json",
            "--target-delta",
            "1.0",
            "--target-si",
            "0.8",
        ])
    )

    assert analysis["telemetry_samples"] == 17
    assert len(analysis["microsectors"]) == 2
    metrics = analysis["metrics"]
    assert 0.0 <= metrics["sense_index"] <= 1.0
    assert analysis["phase_messages"]

    suggestions = json.loads(
        run_cli([
            "suggest",
            str(baseline_path),
            "--export",
            "json",
            "--car-model",
            "generic_gt",
            "--track",
            "valencia",
        ])
    )

    assert suggestions["car_model"] == "generic_gt"
    assert suggestions["track"] == "valencia"
    assert len(suggestions["recommendations"]) >= 1
    assert suggestions["phase_messages"]


def test_configuration_defaults_are_applied(
    tmp_path: Path,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "tnfr-lfs.toml"
    config_path.write_text(
        """
[telemetry]
outsim_port = 4567

[suggest]
car_model = "config_gt"
track = "config_track"

[paths]
output_dir = "artifacts"

[limits.delta_nfr]
entry = 0.2
apex = 0.2
exit = 0.2
"""
    ,
        encoding="utf8",
    )

    baseline_path = tmp_path / "baseline.jsonl"
    run_cli([
        "baseline",
        str(baseline_path),
        "--simulate",
        str(synthetic_stint_path),
    ])

    payload = json.loads(
        run_cli([
            "suggest",
            str(baseline_path),
            "--export",
            "json",
        ])
    )

    assert payload["car_model"] == "config_gt"
    assert payload["track"] == "config_track"
    report_root = Path(payload["reports"]["sense_index_map"]["path"]).parent
    assert report_root.parent.name == "artifacts"
