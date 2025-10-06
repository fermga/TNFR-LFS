from __future__ import annotations

import csv
import itertools
import json
import re
from collections.abc import Iterable, Mapping
from textwrap import dedent
from pathlib import Path

import pytest

from tnfr_lfs.cli import compare as compare_module
from tnfr_lfs.cli import tnfr_lfs_cli as cli_module
from tnfr_lfs.cli.tnfr_lfs_cli import run_cli
from tnfr_lfs.io.profiles import ProfileManager
from tnfr_lfs.recommender.rules import RecommendationEngine

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
    assert re.match(r"xfg_generic_\d{8}_\d{6}_\d{6}\.jsonl$", run_path.name)
    assert "Baseline saved" in result
    assert str(run_path.relative_to(tmp_path)) in captured.out


def test_cli_analyze_accepts_raf_sample(
    tmp_path: Path,
    raf_sample_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    captured_records: dict[str, list] = {}

    original = cli_module.raf_to_telemetry_records

    def _capture_records(raf_file):
        records = original(raf_file)
        captured_records["records"] = records
        return records

    monkeypatch.setattr(cli_module, "raf_to_telemetry_records", _capture_records)

    class _StubThresholds:
        phase_weights: Mapping[str, float] = {}
        robustness: Mapping[str, float] | None = None

    monkeypatch.setattr(
        cli_module,
        "_compute_insights",
        lambda *args, **kwargs: ([], [], _StubThresholds(), None),
    )

    monkeypatch.setattr(
        cli_module,
        "orchestrate_delta_metrics",
        lambda *args, **kwargs: {
            "delta_nfr": 0.0,
            "sense_index": 0.0,
            "bundles": [],
            "microsector_variability": [],
            "stages": {},
            "objectives": {},
            "lap_sequence": [],
        },
    )

    monkeypatch.setattr(
        cli_module,
        "_generate_out_reports",
        lambda *args, **kwargs: {},
    )

    monkeypatch.setattr(
        cli_module,
        "_phase_deviation_messages",
        lambda *args, **kwargs: [],
    )

    monkeypatch.setattr(
        cli_module,
        "compute_session_robustness",
        lambda *args, **kwargs: {},
    )

    payload = json.loads(
        run_cli(
            [
                "analyze",
                str(raf_sample_path),
                "--export",
                "json",
                "--target-delta",
                "0.5",
                "--target-si",
                "0.75",
            ]
        )
    )

    assert payload["telemetry_samples"] == 9586
    records = captured_records.get("records")
    assert records is not None and len(records) == 9586

    record = records[0]
    assert record.wheel_load_fl == pytest.approx(3056.1862793, rel=1e-6)
    assert record.wheel_longitudinal_force_rr == pytest.approx(901.4348755, rel=1e-6)
    assert record.wheel_lateral_force_fl == pytest.approx(1046.0095215, rel=1e-6)


@pytest.fixture()
def cli_config_pack(tmp_path: Path) -> Path:
    pack_root = tmp_path / "pack"
    config_dir = pack_root / "config"
    cars_dir = pack_root / "data" / "cars"
    profiles_dir = pack_root / "data" / "profiles"
    profiles_dir.mkdir(parents=True)
    cars_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    cars_dir.joinpath("ABC.toml").write_text(
        dedent(
            """
            abbrev = "ABC"
            name = "Alpha"
            license = "demo"
            engine_layout = "front"
            drive = "RWD"
            weight_kg = 900
            wheel_rotation_group_deg = 30
            profile = "custom-profile"
            """
        ),
        encoding="utf8",
    )

    profiles_dir.joinpath("custom.toml").write_text(
        dedent(
            """
            [meta]
            id = "custom-profile"
            category = "road"

            [targets.balance]
            delta_nfr = 0.42
            sense_index = 0.83

            [policy.steering]
            aggressiveness = 0.5

            [recommender.steering]
            kp = 1.0
            """
        ),
        encoding="utf8",
    )

    config_dir.joinpath("global.toml").write_text(
        dedent(
            """
            [analyze]
            car_model = "ABC"

            [suggest]
            car_model = "ABC"
            track = "AS5"

            [write_set]
            car_model = "ABC"

            [osd]
            car_model = "ABC"
            track = "AS5"
            """
        ),
        encoding="utf8",
    )

    return pack_root


def test_template_command_emits_phase_presets() -> None:
    output = run_cli([
        "template",
        "--car",
        "FZR",
        "--track",
        "AS5",
    ])
    data = tomllib.loads(output)
    assert data["limits"]["delta_nfr"]["entry"] == pytest.approx(0.8, rel=1e-3)
    analyze_templates = data["analyze"]["phase_templates"]
    entry_template = analyze_templates["entry"]
    assert entry_template["target_delta_nfr"] == pytest.approx(1.2, rel=1e-3)
    assert entry_template["slip_lat_window"][0] == pytest.approx(-0.045, rel=1e-3)
    report_templates = data["report"]["phase_templates"]
    assert report_templates["apex"]["yaw_rate_window"][1] == pytest.approx(0.24, rel=1e-3)


def test_compare_command_attaches_abtest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline_path = tmp_path / "baseline.csv"
    variant_path = tmp_path / "variant.csv"
    baseline_path.write_text("", encoding="utf8")
    variant_path.write_text("", encoding="utf8")

    class DummyBundle:
        def __init__(self, value: float) -> None:
            self.sense_index = value
            self.delta_nfr = value
            self.coherence_index = value
            self.delta_nfr_proj_longitudinal = value
            self.delta_nfr_proj_lateral = value

    metrics_a = {
        "stages": {
            "coherence": {
                "bundles": [
                    DummyBundle(0.60),
                    DummyBundle(0.62),
                    DummyBundle(0.61),
                    DummyBundle(0.63),
                ]
            },
            "recepcion": {"lap_indices": [0, 0, 1, 1]},
        }
    }
    metrics_b = {
        "stages": {
            "coherence": {
                "bundles": [
                    DummyBundle(0.65),
                    DummyBundle(0.67),
                    DummyBundle(0.66),
                    DummyBundle(0.68),
                ]
            },
            "recepcion": {"lap_indices": [0, 0, 1, 1]},
        }
    }
    metrics_iter = itertools.cycle([metrics_a, metrics_b])

    monkeypatch.setattr(cli_module, "_load_records", lambda path: [])
    monkeypatch.setattr(cli_module, "_group_records_by_lap", lambda records: [[], []])
    monkeypatch.setattr(
        compare_module,
        "orchestrate_delta_metrics",
        lambda *args, **kwargs: next(metrics_iter),
    )
    monkeypatch.setattr(cli_module, "_load_pack_cars", lambda pack: {})
    monkeypatch.setattr(cli_module, "_load_pack_track_profiles", lambda pack: {})
    monkeypatch.setattr(cli_module, "_load_pack_modifiers", lambda pack: {})
    monkeypatch.setattr(
        cli_module,
        "_assemble_session_payload",
        lambda *args, **kwargs: {
            "car_model": "XFG",
            "track_profile": "generic",
            "weights": {},
            "hints": {},
        },
    )

    output = run_cli(
        [
            "compare",
            str(baseline_path),
            str(variant_path),
            "--metric",
            "sense_index",
            "--export",
            "json",
        ]
    )
    payload = json.loads(output)
    session = payload["session"]
    assert "abtest" in session
    abtest = session["abtest"]
    assert abtest["metric"] == "sense_index"
    assert abtest["baseline_laps"]
    assert abtest["variant_laps"]


def test_track_argument_resolves_session_payload(mini_track_pack) -> None:
    config: dict[str, object] = {}
    selection = cli_module._resolve_track_argument(
        mini_track_pack.layout_code,
        config,
        pack_root=mini_track_pack.root,
    )

    assert selection.layout == mini_track_pack.layout_code
    assert selection.config is not None
    assert selection.config.name == "Mini Aston Historic"
    assert selection.config.track_profile == mini_track_pack.track_profile
    assert selection.config.length_km == pytest.approx(5.2)

    cars = cli_module._load_pack_cars(mini_track_pack.root)
    track_profiles = cli_module._load_pack_track_profiles(mini_track_pack.root)
    modifiers = cli_module._load_pack_modifiers(mini_track_pack.root)

    payload = cli_module._assemble_session_payload(
        mini_track_pack.car_model,
        selection,
        cars=cars,
        track_profiles=track_profiles,
        modifiers=modifiers,
    )

    assert payload is not None
    assert payload["car_profile"] == mini_track_pack.car_profile
    assert payload["layout_name"] == "Mini Aston Historic"
    assert payload["weights"]["entry"]["brakes"] == pytest.approx(1.05 * 1.4)
    assert payload["hints"]["slip_ratio_bias"] == "aggressive"

    engine = RecommendationEngine(
        car_model=mini_track_pack.car_model,
        track_name=selection.name,
    )
    engine.session = payload
    context = engine._resolve_context(mini_track_pack.car_model, selection.name)
    assert context.session_weights["entry"]["brakes"] == pytest.approx(1.05 * 1.4)
    assert context.session_hints["surface"] == "asphalt"


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
    assert payload["car"]["abbrev"] == "XFG"
    assert payload["tnfr_targets"]["meta"]["category"] == "road"
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
    coherence_entry = payload["reports"]["coherence_map"]
    assert Path(coherence_entry["path"]).exists()
    assert coherence_entry["data"]["microsectors"]
    operator_entry = payload["reports"]["operator_trajectories"]
    assert Path(operator_entry["path"]).exists()
    assert operator_entry["data"]["events"]
    bifurcation_entry = payload["reports"]["delta_bifurcations"]
    assert Path(bifurcation_entry["path"]).exists()
    assert bifurcation_entry["data"]["series"]
    thermal_entry = payload["reports"]["tyre_thermal"]
    assert Path(thermal_entry["path"]).exists()
    assert "temperature" in thermal_entry["data"]
    assert "pressure" in thermal_entry["data"]


def test_analyze_reports_note_missing_tyre_data(
    tmp_path: Path,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    truncated_path = tmp_path / "synthetic_no_tyres.csv"
    with synthetic_stint_path.open(encoding="utf8") as source, truncated_path.open(
        "w", encoding="utf8", newline=""
    ) as destination:
        reader = csv.DictReader(source)
        assert reader.fieldnames is not None
        fieldnames = [
            name
            for name in reader.fieldnames
            if not name.startswith("tyre_temp_") and not name.startswith("tyre_pressure_")
        ]
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            writer.writerow({name: row[name] for name in fieldnames})

    baseline_path = tmp_path / "baseline.jsonl"
    run_cli([
        "baseline",
        str(baseline_path),
        "--simulate",
        str(truncated_path),
    ])

    output = run_cli([
        "analyze",
        str(baseline_path),
        "--export",
        "json",
    ])

    payload = json.loads(output)
    summary_text = payload["reports"]["metrics_summary"]["data"]
    assert "Temperatura (°C): sin datos" in summary_text
    assert "Presión (bar): sin datos" in summary_text
    thermal_entry = payload["reports"]["tyre_thermal"]
    assert Path(thermal_entry["path"]).exists()
    assert thermal_entry["data"]["temperature"] is None
    assert thermal_entry["data"]["pressure"] is None


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
        "FZR",
    ])

    payload = json.loads(output)
    if "car" in payload:
        assert isinstance(payload["car"], dict)
    if "tnfr_targets" in payload:
        assert "targets" in payload["tnfr_targets"]
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
        "--report-format",
        "visual",
    ])

    payload = json.loads(output)
    assert "delta_nfr" in payload
    assert "sense_index" in payload
    assert Path(payload["reports"]["sense_index_map"]["path"]).exists()
    assert Path(payload["reports"]["delta_breakdown"]["path"]).exists()
    assert "intermediate_metrics" in payload
    assert "epi_evolution" in payload["intermediate_metrics"]
    assert payload["intermediate_metrics"]["sense_memory"]["memory"] == payload["recursive_trace"]
    coherence_report = payload["reports"]["coherence_map"]
    assert coherence_report["path"].endswith(".viz")
    assert "rendered" in coherence_report
    operator_report = payload["reports"]["operator_trajectories"]
    assert operator_report["path"].endswith(".viz")
    assert "rendered" in operator_report
    bifurcation_report = payload["reports"]["delta_bifurcations"]
    assert bifurcation_report["path"].endswith(".viz")
    assert "rendered" in bifurcation_report


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
        "FZR",
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
        "FZR",
        "--session",
        "stint-1",
        "--set-output",
        "FZR_race",
    ])

    destination = tmp_path / "LFS/data/setups/FZR_race.set"
    assert destination.exists()
    assert "FZR_race" in message
    assert "TNFR-LFS setup export" in destination.read_text(encoding="utf8")


def test_write_set_combined_export_outputs(
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
    capsys.readouterr()

    result = run_cli([
        "write-set",
        str(baseline_path),
        "--export",
        "set",
        "--export",
        "lfs-notes",
        "--car-model",
        "FZR",
        "--set-output",
        "FZR_test",
    ])

    captured = capsys.readouterr()
    destination = tmp_path / "LFS/data/setups/FZR_test.set"
    assert destination.exists()
    assert "Setup guardado" in result
    assert "Instrucciones rápidas TNFR" in result
    assert "| Cambio | Δ | Acción |" in result
    assert "| Cambio | Δ | Acción |" in captured.out


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
            "FZR",
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
            "FZR",
            "--track",
            "AS5",
        ])
    )

    assert suggestions["car_model"] == "FZR"
    assert suggestions["track"] == "AS5"
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
    assert suggestion_defaults["car_model"] == "FZR"
    assert suggestion_defaults["track"] == "AS5"

    limits = data["limits"]["delta_nfr"]
    assert limits["entry"] == pytest.approx(0.5, rel=1e-3)
    assert limits["apex"] == pytest.approx(0.4, rel=1e-3)
    assert limits["exit"] == pytest.approx(0.6, rel=1e-3)


def test_diagnose_reports_success(tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    lfs_root = tmp_path / "LFS"
    cfg_dir = lfs_root / "cfg"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "cfg.txt"
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

    monkeypatch.setattr(
        "tnfr_lfs.cli.tnfr_lfs_cli._outsim_ping",
        lambda host, port, timeout: (True, f"OutSim respondió desde {host}:{port}"),
    )
    monkeypatch.setattr(
        "tnfr_lfs.cli.tnfr_lfs_cli._outgauge_ping",
        lambda host, port, timeout: (True, f"OutGauge respondió desde {host}:{port}"),
    )
    monkeypatch.setattr(
        "tnfr_lfs.cli.tnfr_lfs_cli._insim_handshake",
        lambda host, port, timeout: (True, "InSim respondió con versión 9"),
    )
    monkeypatch.setattr(
        "tnfr_lfs.cli.tnfr_lfs_cli._check_setups_directory",
        lambda path: (True, "Permisos de escritura confirmados en /fake/setups"),
    )

    result = run_cli(["diagnose", str(cfg_path), "--timeout", "0.05"])
    captured = capsys.readouterr()
    assert "Estado: correcto" in captured.out
    assert "OutSim respondió" in result
    assert "OutGauge respondió" in result
    assert "InSim respondió" in result
    assert "Permisos de escritura confirmados" in result


def test_diagnose_detects_disabled_modes(tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    lfs_root = tmp_path / "LFS"
    cfg_dir = lfs_root / "cfg"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "cfg.txt"
    cfg_path.write_text(
        """
OutSim Mode 0
OutSim IP 127.0.0.1
OutSim Port 4123
OutGauge Mode 0
OutGauge IP 127.0.0.1
OutGauge Port 3000
""",
        encoding="utf8",
    )

    copied: list[list[str]] = []

    def fake_copy(commands: Iterable[str]) -> bool:
        copied.append(list(commands))
        return True

    monkeypatch.setattr("tnfr_lfs.cli.tnfr_lfs_cli._copy_to_clipboard", fake_copy)
    monkeypatch.setattr(
        "tnfr_lfs.cli.tnfr_lfs_cli._check_setups_directory",
        lambda path: (True, "Permisos de escritura confirmados en /fake/setups"),
    )

    with pytest.raises(ValueError) as excinfo:
        run_cli(["diagnose", str(cfg_path)])

    captured = capsys.readouterr()
    assert "/outsim 1 127.0.0.1 4123" in captured.out
    assert "/outgauge 1 127.0.0.1 3000" in captured.out
    assert "copiado" in captured.out
    assert copied and "/outsim 1 127.0.0.1 4123" in copied[0]
    assert "OutSim Mode" in str(excinfo.value)


def test_profiles_persist_and_adjust(
    tmp_path: Path,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "tnfr-lfs.toml"
    config_path.write_text(
        """
[suggest]
car_model = "FZR"
track = "AS5"

[write_set]
car_model = "FZR"

[paths]
profiles = "profiles.toml"
""".strip()
        + "\n",
        encoding="utf8",
    )

    baseline_path = tmp_path / "baseline.csv"
    baseline_path.write_bytes(synthetic_stint_path.read_bytes())

    improved_path = tmp_path / "improved.csv"
    with synthetic_stint_path.open("r", encoding="utf8", newline="") as source, improved_path.open(
        "w",
        encoding="utf8",
        newline="",
    ) as destination:
        reader = csv.reader(source)
        writer = csv.writer(destination)
        header = next(reader)
        writer.writerow(header)
        for row in reader:
            if not row:
                continue
            values = list(row)
            try:
                nfr_value = float(values[10])
                si_value = float(values[11])
            except (ValueError, IndexError):
                writer.writerow(values)
                continue
            values[10] = f"{nfr_value * 0.5:.6f}"
            values[11] = f"{min(0.99, si_value + 0.1):.6f}"
            writer.writerow(values)

    run_cli(["suggest", str(baseline_path), "--export", "json"])

    profiles_path = tmp_path / "profiles.toml"
    assert profiles_path.exists()

    run_cli(["write-set", str(baseline_path), "--car-model", "FZR"])

    def entry_weight() -> float:
        manager = ProfileManager(profiles_path)
        engine = RecommendationEngine(
            car_model="FZR",
            track_name="AS5",
            profile_manager=manager,
        )
        weights = engine._resolve_context("FZR", "AS5").thresholds.weights_for_phase("entry")
        if isinstance(weights, Mapping):
            return float(weights.get("__default__", 1.0))
        return float(weights)

    before_weight = entry_weight()

    run_cli(["suggest", str(improved_path), "--export", "json"])

    after_weight = entry_weight()
    assert after_weight > before_weight

    manager = ProfileManager(profiles_path)
    engine = RecommendationEngine(
        car_model="FZR",
        track_name="AS5",
        profile_manager=manager,
    )
    base_profile = engine._lookup_profile("FZR", "AS5")
    snapshot = manager.resolve("FZR", "AS5", base_profile)
    assert not snapshot.pending_plan
    assert snapshot.last_result is not None
    assert "last_result" in profiles_path.read_text(encoding="utf8")


def test_analyze_uses_pack_root_metadata(
    tmp_path: Path,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli_config_pack: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    baseline_path = tmp_path / "baseline.jsonl"
    config_path = cli_config_pack / "config" / "global.toml"
    pack_arg = str(cli_config_pack)

    run_cli(
        [
            "baseline",
            str(baseline_path),
            "--simulate",
            str(synthetic_stint_path),
            "--config",
            str(config_path),
            "--pack-root",
            pack_arg,
        ]
    )

    payload = json.loads(
        run_cli(
            [
                "analyze",
                str(baseline_path),
                "--export",
                "json",
                "--config",
                str(config_path),
                "--pack-root",
                pack_arg,
            ]
        )
    )

    assert payload["car"]["abbrev"] == "ABC"
    assert payload["tnfr_targets"]["meta"]["id"] == "custom-profile"
    assert payload["tnfr_targets"]["targets"]["balance"]["delta_nfr"] == pytest.approx(
        0.42, rel=1e-3
    )
    assert (tmp_path / "profiles.toml").exists()


def test_diagnose_reports_missing_udp_response(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    lfs_root = tmp_path / "LFS"
    cfg_dir = lfs_root / "cfg"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "cfg.txt"
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

    monkeypatch.setattr(
        "tnfr_lfs.cli.tnfr_lfs_cli._outsim_ping",
        lambda host, port, timeout: (False, "Sin respuesta de OutSim 127.0.0.1:4123 tras 0.05s"),
    )
    monkeypatch.setattr(
        "tnfr_lfs.cli.tnfr_lfs_cli._outgauge_ping",
        lambda host, port, timeout: (True, "OutGauge respondió"),
    )
    monkeypatch.setattr(
        "tnfr_lfs.cli.tnfr_lfs_cli._insim_handshake",
        lambda host, port, timeout: (True, "InSim respondió"),
    )
    monkeypatch.setattr(
        "tnfr_lfs.cli.tnfr_lfs_cli._check_setups_directory",
        lambda path: (True, "Permisos ok"),
    )

    with pytest.raises(ValueError) as excinfo:
        run_cli(["diagnose", str(cfg_path)])

    captured = capsys.readouterr()
    assert "Sin respuesta de OutSim" in captured.out
    assert "Sin respuesta de OutSim" in str(excinfo.value)


def test_diagnose_reports_permission_error(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    lfs_root = tmp_path / "LFS"
    cfg_dir = lfs_root / "cfg"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "cfg.txt"
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

    monkeypatch.setattr(
        "tnfr_lfs.cli.tnfr_lfs_cli._outsim_ping",
        lambda host, port, timeout: (True, "OutSim respondió"),
    )
    monkeypatch.setattr(
        "tnfr_lfs.cli.tnfr_lfs_cli._outgauge_ping",
        lambda host, port, timeout: (True, "OutGauge respondió"),
    )
    monkeypatch.setattr(
        "tnfr_lfs.cli.tnfr_lfs_cli._insim_handshake",
        lambda host, port, timeout: (True, "InSim respondió"),
    )
    monkeypatch.setattr(
        "tnfr_lfs.cli.tnfr_lfs_cli._check_setups_directory",
        lambda path: (False, "No hay permisos de escritura en setups"),
    )

    with pytest.raises(ValueError) as excinfo:
        run_cli(["diagnose", str(cfg_path)])

    captured = capsys.readouterr()
    assert "No hay permisos" in captured.out
    assert "No hay permisos" in str(excinfo.value)
