from __future__ import annotations

import json
from pathlib import Path

import pytest

from tnfr_lfs.cli.tnfr_lfs_cli import run_cli


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "telemetry.csv"
    csv_path.write_text(
        "\n".join(
            [
                "timestamp,vertical_load,slip_ratio,lateral_accel,longitudinal_accel,nfr,si",
                "0.0,450.0,0.10,0.5,-0.1,100.0,0.72",
                "0.5,470.0,0.12,1.5,-0.5,103.0,0.65",
                "1.0,480.0,0.15,1.6,-0.6,108.0,0.55",
                "1.5,490.0,0.20,1.4,-0.2,110.0,0.62",
                "2.0,460.0,0.18,0.6,0.1,105.0,0.75",
                "2.5,455.0,0.10,0.3,0.2,102.0,0.80",
            ]
        ),
        encoding="utf8",
    )
    return csv_path


def test_baseline_simulation_jsonl(tmp_path: Path, capsys, sample_csv: Path) -> None:
    baseline_path = tmp_path / "baseline.jsonl"

    result = run_cli([
        "baseline",
        str(baseline_path),
        "--simulate",
        str(sample_csv),
    ])

    captured = capsys.readouterr()
    assert baseline_path.exists()
    lines = [line for line in baseline_path.read_text(encoding="utf8").splitlines() if line.strip()]
    assert len(lines) == 6
    assert "Baseline saved" in result
    assert str(baseline_path) in captured.out


def test_analyze_pipeline_json_export(tmp_path: Path, capsys, sample_csv: Path) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    run_cli([
        "baseline",
        str(baseline_path),
        "--simulate",
        str(sample_csv),
    ])

    output = run_cli([
        "analyze",
        str(baseline_path),
        "--export",
        "json",
    ])

    captured = capsys.readouterr()
    payload = json.loads(output)
    assert payload["telemetry_samples"] == 6
    assert payload["microsectors"]
    assert "microsectors" in captured.out


def test_suggest_pipeline(tmp_path: Path, capsys, sample_csv: Path) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    run_cli([
        "baseline",
        str(baseline_path),
        "--simulate",
        str(sample_csv),
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


def test_report_generation(tmp_path: Path, capsys, sample_csv: Path) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    run_cli([
        "baseline",
        str(baseline_path),
        "--simulate",
        str(sample_csv),
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


def test_write_set_markdown_export(tmp_path: Path, capsys, sample_csv: Path) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    run_cli([
        "baseline",
        str(baseline_path),
        "--simulate",
        str(sample_csv),
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
