from __future__ import annotations

import json
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
    assert "data/BL1_XFG_baseline.csv" in content
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
        args.extend(["--car-model", "generic_gt"])

    payload = json.loads(run_cli(args))
    assert payload
    if command == "analyze":
        assert payload["telemetry_samples"] == 17
    if command == "suggest":
        assert payload["recommendations"]
