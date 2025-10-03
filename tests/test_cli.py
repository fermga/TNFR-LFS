from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tnfr_lfs.cli.tnfr_lfs_cli import run_cli

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    import tomli as tomllib  # type: ignore


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


def test_baseline_generates_timestamped_run(
    tmp_path: Path, capsys, synthetic_stint_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = run_cli([
        "baseline",
        "--simulate",
        str(synthetic_stint_path),
    ])

    captured = capsys.readouterr()
    runs_dir = tmp_path / "runs"
    assert runs_dir.is_dir()
    runs = list(runs_dir.glob("*.jsonl"))
    assert len(runs) == 1
    run_path = runs[0]
    assert re.match(r"generic_generic_\d{8}_\d{6}_\d{6}\.jsonl$", run_path.name)
    assert "Baseline saved" in result
    assert str(run_path.relative_to(tmp_path)) in captured.out


def test_template_command_emits_phase_presets() -> None:
    output = run_cli([
        "template",
        "--car",
        "generic_gt",
        "--track",
        "valencia",
    ])
    data = tomllib.loads(output)
    assert data["limits"]["delta_nfr"]["entry"] == pytest.approx(0.8, rel=1e-3)
    analyze_templates = data["analyze"]["phase_templates"]
    entry_template = analyze_templates["entry"]
    assert entry_template["target_delta_nfr"] == pytest.approx(1.2, rel=1e-3)
    assert entry_template["slip_lat_window"][0] == pytest.approx(-0.045, rel=1e-3)
    report_templates = data["report"]["phase_templates"]
    assert report_templates["apex"]["yaw_rate_window"][1] == pytest.approx(0.24, rel=1e-3)


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
    assert "intermediate_metrics" in payload
    epi_metrics = payload["intermediate_metrics"]["epi_evolution"]
    assert len(epi_metrics["integrated"]) == len(payload["series"])
    nodal_metrics = payload["intermediate_metrics"]["nodal_metrics"]
    assert set(nodal_metrics["delta_by_node"]) == {"tyres", "suspension", "chassis"}
    memory = payload["intermediate_metrics"]["sense_memory"]
    assert len(memory["memory"]) == len(payload["series"])
    report_paths = payload["reports"]
    sense_path = Path(report_paths["sense_index_map"]["path"])
    resonance_path = Path(report_paths["modal_resonance"]["path"])
    breakdown_path = Path(report_paths["delta_breakdown"]["path"])
    assert sense_path.exists()
    assert resonance_path.exists()
    assert breakdown_path.exists()
    assert payload["reports"]["sense_index_map"]["data"]
    resonance_data = payload["reports"]["modal_resonance"]["data"]
    assert set(resonance_data) >= {"yaw", "roll", "pitch"}
    for axis, axis_data in resonance_data.items():
        assert "sample_rate" in axis_data
        assert "total_energy" in axis_data
        assert isinstance(axis_data.get("peaks", []), list)
    breakdown_data = payload["reports"]["delta_breakdown"]["data"]
    assert breakdown_data["samples"] == len(payload["series"])
    tyres_summary = breakdown_data["per_node"]["tyres"]
    tyres_total = sum(tyres_summary["breakdown"].values())
    assert tyres_total == pytest.approx(tyres_summary["delta_nfr_total"], rel=1e-6, abs=1e-9)


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
    assert Path(payload["reports"]["modal_resonance"]["path"]).exists()
    assert Path(payload["reports"]["delta_breakdown"]["path"]).exists()
    assert any(
        "BAS-" in rec.get("rationale", "") or "ADV-" in rec.get("rationale", "")
        for rec in payload["recommendations"]
    )


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
    assert Path(payload["reports"]["delta_breakdown"]["path"]).exists()
    assert "intermediate_metrics" in payload
    assert "epi_evolution" in payload["intermediate_metrics"]
    assert payload["intermediate_metrics"]["sense_memory"]["memory"] == payload["recursive_trace"]


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


def test_write_set_lfs_export(
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

    message = run_cli([
        "write-set",
        str(baseline_path),
        "--export",
        "set",
        "--car-model",
        "generic_gt",
        "--session",
        "stint-1",
        "--set-output",
        "GEN_race",
    ])

    destination = tmp_path / "LFS/data/setups/GEN_race.set"
    assert destination.exists()
    assert "GEN_race" in message
    assert "TNFR-LFS setup export" in destination.read_text(encoding="utf8")


def test_write_set_lfs_rejects_invalid_name(
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

    with pytest.raises(ValueError):
        run_cli([
            "write-set",
            str(baseline_path),
            "--export",
            "set",
            "--car-model",
            "generic_gt",
            "--set-output",
            "bad_name",
        ])


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


def test_repository_template_configures_default_ports_and_profiles() -> None:
    config_path = Path(__file__).resolve().parents[1] / "tnfr-lfs.toml"
    data = tomllib.loads(config_path.read_text(encoding="utf8"))

    telemetry = data["telemetry"]
    assert telemetry["host"] == "127.0.0.1"
    assert telemetry["outsim_port"] == 4123
    assert telemetry["outgauge_port"] == 3000
    assert telemetry["insim_port"] == 29999

    suggestion_defaults = data["suggest"]
    assert suggestion_defaults["car_model"] == "generic_gt"
    assert suggestion_defaults["track"] == "valencia"

    limits = data["limits"]["delta_nfr"]
    assert limits["entry"] == pytest.approx(0.5, rel=1e-3)
    assert limits["apex"] == pytest.approx(0.4, rel=1e-3)
    assert limits["exit"] == pytest.approx(0.6, rel=1e-3)


class _FakeSocket:
    def __init__(self) -> None:
        self.bound: list[tuple[str, int]] = []
        self.connected: list[tuple[str, int]] = []

    def __enter__(self) -> "_FakeSocket":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[override]
        return False

    def setsockopt(self, *args, **kwargs) -> None:
        return None

    def settimeout(self, timeout: float) -> None:
        return None

    def bind(self, address: tuple[str, int]) -> None:
        self.bound.append(address)

    def connect(self, address: tuple[str, int]) -> None:
        self.connected.append(address)


def test_diagnose_reports_success(tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "cfg.txt"
    cfg_path.write_text(
        """
OutSim Mode 1
OutSim IP 127.0.0.1
OutSim Port 4123
OutGauge Mode 1
OutGauge IP 127.0.0.1
OutGauge Port 3000
InSim IP 127.0.0.1
InSim Port 29999
""",
        encoding="utf8",
    )

    def fake_socket(*args, **kwargs):
        return _FakeSocket()

    monkeypatch.setattr("tnfr_lfs.cli.tnfr_lfs_cli.socket.socket", fake_socket)

    result = run_cli(["diagnose", str(cfg_path), "--timeout", "0.05"])
    captured = capsys.readouterr()
    assert "Estado: correcto" in captured.out
    assert "OutSim listo" in result
    assert "OutGauge listo" in result
    assert "InSim alcanzable" in result


def test_diagnose_detects_disabled_modes(tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "cfg.txt"
    cfg_path.write_text(
        """
OutSim Mode 0
OutSim Port 4123
OutGauge Mode 0
OutGauge Port 3000
""",
        encoding="utf8",
    )

    with pytest.raises(ValueError) as excinfo:
        run_cli(["diagnose", str(cfg_path)])

    captured = capsys.readouterr()
    assert "OutSim Mode" in captured.out
    assert "OutGauge Mode" in str(excinfo.value)


class _BusySocket(_FakeSocket):
    def bind(self, address: tuple[str, int]) -> None:
        raise OSError("address already in use")

    def connect(self, address: tuple[str, int]) -> None:
        raise OSError("address already in use")


def test_diagnose_detects_socket_errors(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = tmp_path / "cfg.txt"
    cfg_path.write_text(
        """
OutSim Mode 1
OutSim IP 127.0.0.1
OutSim Port 4123
OutGauge Mode 1
OutGauge IP 127.0.0.1
OutGauge Port 3000
""",
        encoding="utf8",
    )

    monkeypatch.setattr("tnfr_lfs.cli.tnfr_lfs_cli.socket.socket", lambda *a, **k: _BusySocket())

    with pytest.raises(ValueError) as excinfo:
        run_cli(["diagnose", str(cfg_path)])

    captured = capsys.readouterr()
    assert "OutSim falló" in captured.out
    assert "OutGauge falló" in str(excinfo.value)
