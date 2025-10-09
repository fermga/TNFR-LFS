from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from tnfr_lfs.cli.tnfr_lfs_cli import run_cli


def test_quickstart_dataset_available(quickstart_dataset_path: Path) -> None:
    """The quickstart dataset is shipped with the repository."""

    assert quickstart_dataset_path.exists()
    header = quickstart_dataset_path.read_text(encoding="utf8").splitlines()[0]
    columns = header.split(",")
    assert columns[:5] == [
        "timestamp",
        "vertical_load",
        "slip_ratio",
        "lateral_accel",
        "longitudinal_accel",
    ]


def test_quickstart_script_points_to_dataset(quickstart_dataset_path: Path) -> None:
    """The shell helper keeps the dataset path in sync with the docs/tests."""

    script_path = Path(__file__).resolve().parents[1] / "examples" / "quickstart.sh"
    content = script_path.read_text(encoding="utf8")
    assert "tnfr_lfs._pack_resources import data_root" in content
    assert quickstart_dataset_path.name in content


@pytest.mark.parametrize("command", ["baseline", "analyze", "suggest"])
def test_quickstart_dataset_pipeline(
    tmp_path: Path,
    quickstart_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    """Core quickstart commands accept the shared dataset without errors."""

    monkeypatch.chdir(tmp_path)
    baseline_path = tmp_path / "baseline.jsonl"

    run_cli([
        "baseline",
        str(baseline_path),
        "--simulate",
        str(quickstart_dataset_path),
    ])

    if command == "baseline":
        assert baseline_path.exists()
        return

    args = [command, str(baseline_path), "--export", "json"]
    if command == "suggest":
        args.extend(["--car-model", "FZR"])

    payload = json.loads(run_cli(args))
    assert payload
    if command == "analyze":
        assert payload["telemetry_samples"] == 17
    if command == "suggest":
        assert payload["recommendations"]


def test_analyze_reports_microsector_variability_by_lap(
    tmp_path: Path, synthetic_records
) -> None:
    samples_per_lap = 12
    laps = []
    offset = 0.0
    for lap_index in range(2):
        lap_records = []
        for record in synthetic_records[:samples_per_lap]:
            payload = asdict(record)
            payload["timestamp"] = float(payload["timestamp"]) + offset
            payload["lap"] = f"lap-{lap_index + 1}"
            lap_records.append(payload)
        laps.extend(lap_records)
        offset += 100.0
    dataset_path = tmp_path / "multi_lap.json"
    dataset_path.write_text(json.dumps(laps), encoding="utf8")

    payload = json.loads(
        run_cli(["analyze", str(dataset_path), "--export", "json"])
    )

    metrics = payload["metrics"]
    variability = metrics["microsector_variability"]
    assert variability
    lap_labels = set()
    for entry in variability:
        lap_labels.update(entry.get("laps", {}).keys())
        overall = entry.get("overall", {})
        assert overall.get("sense_index", {}).get("stability_score") is not None
        assert "delta_nfr_integral" in overall
        for lap_stats in entry.get("laps", {}).values():
            assert "phase_synchrony" in lap_stats
    assert {"lap-1", "lap-2"} <= lap_labels
    reports = payload["reports"]
    assert "microsector_variability" in reports
    assert reports["microsector_variability"]["data"]
