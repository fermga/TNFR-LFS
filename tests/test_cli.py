from __future__ import annotations

import csv
import itertools
import json
import re
import argparse
from collections.abc import Callable, Iterable, Mapping
from textwrap import dedent
from pathlib import Path
from types import SimpleNamespace

import logging
import pytest

from tnfr_lfs.cli import compare as compare_module
from tnfr_lfs.cli import app as cli_module
from tnfr_lfs.cli import io as cli_io_module
from tnfr_lfs.cli import workflows as workflows_module
from tnfr_lfs.cli import run_cli
from tnfr_lfs.cli.common import CliError
from tnfr_lfs.ingestion.offline import ProfileManager
from tnfr_lfs.recommender.rules import RecommendationEngine
from tnfr_lfs.core.cache_settings import DEFAULT_DYNAMIC_CACHE_SIZE
from tnfr_lfs.configuration import load_project_config
from tests.helpers import (
    DummyBundle,
    create_cli_config_pack,
    instrument_prepare_pack_context,
    run_cli_in_tmp,
)

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    import tomli as tomllib  # type: ignore


@pytest.fixture
def prepare_diagnose_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[Mapping[str, tuple[bool, str]] | None], Path]:
    def _prepare(stub_overrides: Mapping[str, tuple[bool, str]] | None = None) -> Path:
        lfs_root = tmp_path / "LFS"
        cfg_dir = lfs_root / "cfg"
        cfg_dir.mkdir(parents=True)
        cfg_path = cfg_dir / "cfg.txt"
        cfg_path.write_text(
            """\
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

        defaults: dict[str, tuple[bool, str]] = {
            "_outsim_ping": (True, "OutSim responded"),
            "_outgauge_ping": (True, "OutGauge responded"),
            "_insim_handshake": (True, "InSim responded"),
            "_check_setups_directory": (True, "Permissions ok"),
        }
        if stub_overrides:
            defaults.update(stub_overrides)

        for module in (cli_module, workflows_module):
            monkeypatch.setattr(
                module,
                "_outsim_ping",
                lambda host, port, timeout, result=defaults["_outsim_ping"]: result,
            )
            monkeypatch.setattr(
                module,
                "_outgauge_ping",
                lambda host, port, timeout, result=defaults["_outgauge_ping"]: result,
            )
            monkeypatch.setattr(
                module,
                "_insim_handshake",
                lambda host, port, timeout, result=defaults["_insim_handshake"]: result,
            )
            monkeypatch.setattr(
                module,
                "_check_setups_directory",
                lambda path, result=defaults["_check_setups_directory"]: result,
            )

        return cfg_path

    return _prepare


def test_run_cli_dispatches_registered_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Mapping[str, object]]] = []

    def dummy_handler(
        namespace: argparse.Namespace, *, config: Mapping[str, object]
    ) -> str:
        calls.append((str(namespace.command), config))
        return "dummy-result"

    def register_dummy(
        subparsers: argparse._SubParsersAction[argparse.ArgumentParser], *, config: Mapping[str, object]
    ) -> None:
        parser = subparsers.add_parser("dummy", help="Dummy command used for testing.")
        parser.set_defaults(handler=dummy_handler)

    def build_parser_stub(config: Mapping[str, object] | None = None) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        register_dummy(subparsers, config=config or {})
        return parser

    monkeypatch.setattr(cli_module, "build_parser", build_parser_stub)
    monkeypatch.setattr(cli_module, "load_cli_config", lambda path: {})

    result = run_cli(["dummy"])

    assert result == "dummy-result"
    assert len(calls) == 1
    command, config = calls[0]
    assert command == "dummy"
    assert config.get("logging") == {
        "level": "info",
        "output": "stderr",
        "format": "json",
    }


def test_run_cli_loads_pyproject_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        dedent(
            """
            [tool.tnfr_lfs.logging]
            level = "warning"
            format = "text"
            output = "stdout"

            [tool.tnfr_lfs.performance]
            cache_enabled = false
            max_cache_size = 12
            """
        ),
        encoding="utf8",
    )

    captured_configs: list[Mapping[str, object]] = []

    def dummy_handler(
        namespace: argparse.Namespace, *, config: Mapping[str, object]
    ) -> str:
        captured_configs.append(config)
        return "ok"

    def build_parser_stub(
        config: Mapping[str, object] | None = None,
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        parser_dummy = subparsers.add_parser("dummy")
        parser_dummy.set_defaults(handler=dummy_handler)
        return parser

    monkeypatch.setattr(cli_module, "build_parser", build_parser_stub)

    result = run_cli(["dummy"])

    assert result == "ok"
    assert len(captured_configs) == 1
    config = captured_configs[0]
    assert config["_config_path"] == str(pyproject_path.resolve())
    logging_cfg = config["logging"]
    assert logging_cfg["level"] == "warning"
    assert logging_cfg["format"] == "text"
    assert logging_cfg["output"] == "stdout"
    performance_cfg = config["performance"]
    assert performance_cfg["cache_enabled"] is False
    assert performance_cfg["nu_f_cache_size"] == 0
    assert performance_cfg["max_cache_size"] == 0


def test_run_cli_requires_telemetry_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "load_cli_config", lambda path: {})

    with pytest.raises(SystemExit) as excinfo:
        run_cli(["analyze"])

    assert excinfo.value.code == 1
    assert isinstance(excinfo.value.__cause__, CliError)
    assert str(excinfo.value.__cause__) == (
        "A telemetry baseline path is required unless --replay-csv-bundle is provided."
    )


def test_cli_exports_helper_attributes() -> None:
    assert hasattr(cli_module, "_load_pack_profiles")
    assert callable(cli_module._load_pack_profiles)
    assert hasattr(cli_module, "_load_pack_lfs_class_overrides")
    assert callable(cli_module._load_pack_lfs_class_overrides)


def test_baseline_simulation_jsonl(
    tmp_path: Path, capsys, synthetic_stint_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = tmp_path / "baseline.jsonl"

    result, captured = run_cli_in_tmp(
        [
            "baseline",
            str(baseline_path),
            "--simulate",
            str(synthetic_stint_path),
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        capsys=capsys,
        capture_output=True,
    )
    assert baseline_path.exists()
    lines = [line for line in baseline_path.read_text(encoding="utf8").splitlines() if line.strip()]
    assert len(lines) == 17
    assert "Baseline saved" in result
    assert str(baseline_path) in captured.err


def test_baseline_generates_timestamped_run(
    tmp_path: Path, capsys, synthetic_stint_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, captured = run_cli_in_tmp(
        [
            "baseline",
            "--simulate",
            str(synthetic_stint_path),
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        capsys=capsys,
        capture_output=True,
    )
    runs_dir = tmp_path / "runs"
    assert runs_dir.is_dir()
    runs = list(runs_dir.glob("*.jsonl"))
    assert len(runs) == 1
    run_path = runs[0]
    assert re.match(r"xfg_generic_\d{8}_\d{6}_\d{6}\.jsonl$", run_path.name)
    assert "Baseline saved" in result
    assert str(run_path.relative_to(tmp_path)) in captured.err


def test_baseline_simulation_accepts_optional_flags(
    tmp_path: Path,
    capsys,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "custom_runs"

    result, captured = run_cli_in_tmp(
        [
            "baseline",
            "--simulate",
            str(synthetic_stint_path),
            "--duration",
            "12",
            "--limit",
            "5",
            "--output",
            "session.jsonl",
            "--output-dir",
            str(output_dir),
            "--insim-keepalive",
            "2.5",
            "--force",
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        capsys=capsys,
        capture_output=True,
    )
    destination = output_dir / "session.jsonl"
    assert destination.exists()
    lines = [line for line in destination.read_text(encoding="utf8").splitlines() if line.strip()]
    assert len(lines) == 5
    assert "Baseline saved" in result
    assert str(destination.relative_to(tmp_path)) in captured.err


def test_baseline_overlay_uses_keepalive_and_overlay(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    created_clients: list[dict[str, float | str]] = []
    overlay_calls: dict[str, object] = {}

    class _DummyInSimClient:
        def __init__(self, *, host, port, timeout, keepalive_interval, app_name):  # type: ignore[no-untyped-def]
            created_clients.append(
                {
                    "host": host,
                    "port": port,
                    "timeout": timeout,
                    "keepalive_interval": keepalive_interval,
                    "app_name": app_name,
                }
            )

        def close(self) -> None:
            pass

    class _DummyOverlay:
        def __init__(self, client, layout):  # type: ignore[no-untyped-def]
            overlay_calls["client"] = client
            overlay_calls["layout"] = layout
            overlay_calls["tick"] = 0

        def connect(self) -> None:
            overlay_calls["connected"] = True

        def show(self, lines) -> None:  # type: ignore[no-untyped-def]
            overlay_calls.setdefault("messages", []).append(tuple(lines))

        def tick(self) -> None:
            overlay_calls["tick"] = overlay_calls.get("tick", 0) + 1

        def close(self) -> None:
            overlay_calls["closed"] = True

    captured_args: dict[str, object] = {}

    def _fake_capture(**kwargs):  # type: ignore[no-untyped-def]
        heartbeat = kwargs.get("heartbeat")
        if callable(heartbeat):
            heartbeat()
        captured_args["buffer_size"] = kwargs.get("buffer_size")
        return workflows_module.CaptureResult(
            records=[],
            metrics=workflows_module.CaptureMetrics(
                attempts=1,
                samples=0,
                dropped_pairs=1,
                duration=0.0,
                outsim_timeouts=1,
                outgauge_timeouts=1,
                outsim_ignored_hosts=0,
                outgauge_ignored_hosts=0,
                outsim_loss_events=0,
                outgauge_loss_events=0,
                outsim_recovered_packets=0,
                outgauge_recovered_packets=0,
            ),
        )

    monkeypatch.setattr(workflows_module, "InSimClient", _DummyInSimClient)
    monkeypatch.setattr(workflows_module, "OverlayManager", _DummyOverlay)
    monkeypatch.setattr(workflows_module, "_capture_udp_samples", _fake_capture)

    result, _ = run_cli_in_tmp(
        [
            "baseline",
            "--overlay",
            "--duration",
            "3",
            "--insim-keepalive",
            "1.25",
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        capsys=capsys,
        capture_output=True,
    )
    assert "No telemetry samples captured" in result
    assert created_clients and created_clients[0]["keepalive_interval"] == pytest.approx(1.25)
    assert overlay_calls.get("connected")
    assert overlay_calls.get("tick") == 1
    assert overlay_calls.get("closed")
    assert captured_args.get("buffer_size") is None


def test_baseline_threads_configured_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(
        "[tool.tnfr_lfs.performance]\ntelemetry_buffer_size = 12\n",
        encoding="utf8",
    )

    captured: dict[str, object] = {}

    def _fake_capture(**kwargs):  # type: ignore[no-untyped-def]
        captured["buffer_size"] = kwargs.get("buffer_size")
        return workflows_module.CaptureResult(
            records=[],
            metrics=workflows_module.CaptureMetrics(
                attempts=1,
                samples=0,
                dropped_pairs=0,
                duration=0.0,
                outsim_timeouts=0,
                outgauge_timeouts=0,
                outsim_ignored_hosts=0,
                outgauge_ignored_hosts=0,
                outsim_loss_events=0,
                outgauge_loss_events=0,
                outsim_recovered_packets=0,
                outgauge_recovered_packets=0,
            ),
        )

    monkeypatch.setattr(workflows_module, "_capture_udp_samples", _fake_capture)

    result = run_cli_in_tmp(
        ["baseline", "--duration", "1"],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    assert "No telemetry samples captured" in result
    assert captured.get("buffer_size") == 12


def test_cli_analyze_accepts_raf_sample(
    tmp_path: Path,
    raf_sample_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with instrument_prepare_pack_context(monkeypatch) as prepare_calls:
        captured_records: dict[str, list] = {}

        original = cli_io_module.raf_to_telemetry_records

        def _capture_records(raf_file):
            records = original(raf_file)
            captured_records["records"] = records
            return records

        monkeypatch.setattr(cli_io_module, "raf_to_telemetry_records", _capture_records)
        monkeypatch.setattr(cli_module, "raf_to_telemetry_records", _capture_records)

        class _StubThresholds:
            phase_weights: Mapping[str, float] = {}
            robustness: Mapping[str, float] | None = None

        from tnfr_lfs.analysis.insights import InsightsResult

        stub_insights = lambda *args, **kwargs: InsightsResult([], [], _StubThresholds(), None, {})

        monkeypatch.setattr(workflows_module, "compute_insights", stub_insights)
        monkeypatch.setattr(cli_module, "compute_insights", stub_insights)

        monkeypatch.setattr(
            workflows_module,
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
        monkeypatch.setattr(cli_module, "orchestrate_delta_metrics", workflows_module.orchestrate_delta_metrics)

        monkeypatch.setattr(
            workflows_module,
            "_generate_out_reports",
            lambda *args, **kwargs: {},
        )
        monkeypatch.setattr(cli_module, "_generate_out_reports", workflows_module._generate_out_reports)

        monkeypatch.setattr(
            workflows_module,
            "_phase_deviation_messages",
            lambda *args, **kwargs: [],
        )
        monkeypatch.setattr(cli_module, "_phase_deviation_messages", workflows_module._phase_deviation_messages)

        import tnfr_lfs.analysis.insights as insights_module

        monkeypatch.setattr(
            insights_module,
            "compute_session_robustness",
            lambda *args, **kwargs: {},
        )

        payload = json.loads(
            run_cli_in_tmp(
                [
                    "analyze",
                    str(raf_sample_path),
                    "--export",
                    "json",
                    "--target-delta",
                    "0.5",
                    "--target-si",
                    "0.75",
                ],
                tmp_path=tmp_path,
                monkeypatch=monkeypatch,
            )
        )

        assert prepare_calls["count"] == 1

        assert payload["telemetry_samples"] == 9586
        records = captured_records.get("records")
        assert records is not None and len(records) == 9586

        record = records[0]
        assert record.wheel_load_fl == pytest.approx(3056.1862793, rel=1e-6)
        assert record.wheel_longitudinal_force_rr == pytest.approx(901.4348755, rel=1e-6)
        assert record.wheel_lateral_force_fl == pytest.approx(1046.0095215, rel=1e-6)


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
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(
        dedent(
            """
            [tool.tnfr_lfs.compare]
            car_model = "XFG"
            target_delta = 0.42
            target_si = 0.86
            coherence_window = 5
            recursion_decay = 0.65
            """
        ),
        encoding="utf8",
    )
    baseline_path = tmp_path / "baseline.csv"
    variant_path = tmp_path / "variant.csv"
    baseline_path.write_text("", encoding="utf8")
    variant_path.write_text("", encoding="utf8")

    metrics_a = {
        "stages": {
            "coherence": {
                "bundles": [
                    DummyBundle(
                        sense_index=value,
                        delta_nfr=value,
                        coherence_index=value,
                        delta_nfr_proj_longitudinal=value,
                        delta_nfr_proj_lateral=value,
                    )
                    for value in (0.60, 0.62, 0.61, 0.63)
                ]
            },
            "reception": {"lap_indices": [0, 0, 1, 1]},
        }
    }
    metrics_b = {
        "stages": {
            "coherence": {
                "bundles": [
                    DummyBundle(
                        sense_index=value,
                        delta_nfr=value,
                        coherence_index=value,
                        delta_nfr_proj_longitudinal=value,
                        delta_nfr_proj_lateral=value,
                    )
                    for value in (0.65, 0.67, 0.66, 0.68)
                ]
            },
            "reception": {"lap_indices": [0, 0, 1, 1]},
        }
    }
    metrics_iter = itertools.cycle([metrics_a, metrics_b])
    captured_objectives: list[dict[str, float]] = []

    def _capture_orchestrate(
        lap_segments,
        target_delta_nfr,
        target_sense_index,
        *,
        coherence_window,
        recursion_decay,
        **kwargs,
    ):
        captured_objectives.append(
            {
                "target_delta": float(target_delta_nfr),
                "target_si": float(target_sense_index),
                "coherence_window": int(coherence_window),
                "recursion_decay": float(recursion_decay),
            }
        )
        return next(metrics_iter)

    monkeypatch.setattr(cli_module, "_load_records", lambda path: [])
    monkeypatch.setattr(cli_module, "_group_records_by_lap", lambda records: [[], []])
    monkeypatch.setattr(compare_module, "orchestrate_delta_metrics", _capture_orchestrate)
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

    output = run_cli_in_tmp(
        [
            "--config",
            str(config_path),
            "compare",
            str(baseline_path),
            str(variant_path),
            "--metric",
            "sense_index",
            "--export",
            "json",
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    payload = json.loads(output)
    session = payload["session"]
    assert "abtest" in session
    abtest = session["abtest"]
    assert abtest["metric"] == "sense_index"
    assert abtest["baseline_laps"]
    assert abtest["variant_laps"]
    assert len(captured_objectives) == 2
    for captured in captured_objectives:
        assert captured["target_delta"] == pytest.approx(0.42)
        assert captured["target_si"] == pytest.approx(0.86)
        assert captured["coherence_window"] == 5
        assert captured["recursion_decay"] == pytest.approx(0.65)


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


def test_prepare_pack_context_matches_loaders() -> None:
    namespace = SimpleNamespace(pack_root=None)
    config: dict[str, object] = {}
    pack_context = workflows_module._prepare_pack_context(
        namespace,
        config,
        car_model="FZR",
    )

    assert pack_context.pack_root is None
    assert pack_context.cars == workflows_module._load_pack_cars(None)
    assert pack_context.track_profiles == workflows_module._load_pack_track_profiles(None)
    assert pack_context.modifiers == workflows_module._load_pack_modifiers(None)
    assert pack_context.class_overrides == workflows_module._load_pack_lfs_class_overrides(None)
    assert pack_context.profile_manager.path == pack_context.profiles_ctx.storage_path
    assert pack_context.tnfr_targets == workflows_module._resolve_tnfr_targets(
        "FZR",
        pack_context.cars,
        pack_context.profiles_ctx.pack_profiles,
        overrides=pack_context.class_overrides,
    )


def test_analyze_pipeline_json_export(
    tmp_path: Path,
    capsys,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with instrument_prepare_pack_context(monkeypatch) as prepare_calls:
        baseline_path = tmp_path / "baseline.jsonl"
        run_cli_in_tmp(
            [
                "baseline",
                str(baseline_path),
                "--simulate",
                str(synthetic_stint_path),
            ],
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )

        output, captured = run_cli_in_tmp(
            [
                "analyze",
                str(baseline_path),
                "--export",
                "json",
            ],
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            capsys=capsys,
            capture_output=True,
        )
        payload = json.loads(output)
        assert prepare_calls["count"] == 1
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


def test_analyze_debug_logging_safe(
    tmp_path: Path,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    run_cli_in_tmp(
        [
            "baseline",
            str(baseline_path),
            "--simulate",
            str(synthetic_stint_path),
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    original_level = workflows_module.logger.level
    workflows_module.logger.setLevel(logging.DEBUG)
    try:
        result = run_cli_in_tmp(
            [
                "analyze",
                str(baseline_path),
            ],
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    finally:
        workflows_module.logger.setLevel(original_level)

    assert isinstance(result, str)


def test_analyze_reports_note_missing_tyre_data(
    tmp_path: Path,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    run_cli_in_tmp(
        [
            "baseline",
            str(baseline_path),
            "--simulate",
            str(truncated_path),
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    output = run_cli_in_tmp(
        [
            "analyze",
            str(baseline_path),
            "--export",
            "json",
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    payload = json.loads(output)
    summary_text = payload["reports"]["metrics_summary"]["data"]
    assert "Temperature (°C): no data" in summary_text
    assert "Pressure (bar): no data" in summary_text
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
    with instrument_prepare_pack_context(monkeypatch) as prepare_calls:
        baseline_path = tmp_path / "baseline.jsonl"
        run_cli_in_tmp(
            [
                "baseline",
                str(baseline_path),
                "--simulate",
                str(synthetic_stint_path),
            ],
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )

        output = run_cli_in_tmp(
            [
                "suggest",
                str(baseline_path),
                "--export",
                "json",
                "--car-model",
                "FZR",
            ],
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )

        payload = json.loads(output)
    assert prepare_calls["count"] == 1
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
    with instrument_prepare_pack_context(monkeypatch) as prepare_calls:
        baseline_path = tmp_path / "baseline.jsonl"
        run_cli_in_tmp(
            [
                "baseline",
                str(baseline_path),
                "--simulate",
                str(synthetic_stint_path),
            ],
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )

        output = run_cli_in_tmp(
            [
                "report",
                str(baseline_path),
                "--export",
                "json",
                "--report-format",
                "visual",
            ],
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )

        payload = json.loads(output)
    assert prepare_calls["count"] == 1
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
    baseline_path = tmp_path / "baseline.jsonl"
    run_cli_in_tmp(
        [
            "baseline",
            str(baseline_path),
            "--simulate",
            str(synthetic_stint_path),
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    output = run_cli_in_tmp(
        [
            "write-set",
            str(baseline_path),
            "--export",
            "markdown",
            "--car-model",
            "FZR",
            "--session",
            "stint-1",
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    assert "| Change |" in output


def test_write_set_lfs_export(
    tmp_path: Path,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    run_cli_in_tmp(
        [
            "baseline",
            str(baseline_path),
            "--simulate",
            str(synthetic_stint_path),
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    message = run_cli_in_tmp(
        [
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
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

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
    baseline_path = tmp_path / "baseline.jsonl"
    run_cli_in_tmp(
        [
            "baseline",
            str(baseline_path),
            "--simulate",
            str(synthetic_stint_path),
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        capsys=capsys,
        capture_output=True,
    )

    result, captured = run_cli_in_tmp(
        [
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
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        capsys=capsys,
        capture_output=True,
    )
    destination = tmp_path / "LFS/data/setups/FZR_test.set"
    assert destination.exists()
    assert "Setup saved" in result
    assert "Quick TNFR notes" in result
    assert "| Change | Δ | Action |" in result
    assert "| Change | Δ | Action |" in captured.out


def test_write_set_lfs_rejects_invalid_name(
    tmp_path: Path,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    run_cli_in_tmp(
        [
            "baseline",
            str(baseline_path),
            "--simulate",
            str(synthetic_stint_path),
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(ValueError):
        run_cli_in_tmp(
            [
                "write-set",
                str(baseline_path),
                "--export",
                "set",
                "--car-model",
                "FZR",
                "--set-output",
                "bad_name",
            ],
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )


def test_cli_end_to_end_pipeline(
    tmp_path: Path,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    run_cli_in_tmp(
        [
            "baseline",
            str(baseline_path),
            "--simulate",
            str(synthetic_stint_path),
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    analysis = json.loads(
        run_cli_in_tmp(
            [
                "analyze",
                str(baseline_path),
                "--export",
                "json",
                "--target-delta",
                "1.0",
                "--target-si",
                "0.8",
            ],
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    )

    assert analysis["telemetry_samples"] == 17
    assert len(analysis["microsectors"]) == 2
    metrics = analysis["metrics"]
    assert 0.0 <= metrics["sense_index"] <= 1.0
    assert analysis["phase_messages"]

    suggestions = json.loads(
        run_cli_in_tmp(
            [
                "suggest",
                str(baseline_path),
                "--export",
                "json",
                "--car-model",
                "FZR",
                "--track",
                "AS5",
            ],
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
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
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(
        """
[tool.tnfr_lfs.core]
outsim_port = 4567

[tool.tnfr_lfs.performance]
telemetry_buffer_size = 8

[tool.tnfr_lfs.suggest]
car_model = "config_gt"
track = "config_track"

[tool.tnfr_lfs.paths]
output_dir = "artifacts"

[tool.tnfr_lfs.limits.delta_nfr]
entry = 0.2
apex = 0.2
exit = 0.2
"""
        ,
        encoding="utf8",
    )

    baseline_path = tmp_path / "baseline.jsonl"
    run_cli_in_tmp(
        [
            "baseline",
            str(baseline_path),
            "--simulate",
            str(synthetic_stint_path),
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    payload = json.loads(
        run_cli_in_tmp(
            [
                "suggest",
                str(baseline_path),
                "--export",
                "json",
            ],
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    )

    assert payload["car_model"] == "config_gt"
    assert payload["track"] == "config_track"
    report_root = Path(payload["reports"]["sense_index_map"]["path"]).parent
    assert report_root.parent.name == "artifacts"


def test_repository_pyproject_configures_default_ports_and_profiles() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    loaded = load_project_config(pyproject_path)
    assert loaded is not None
    data, resolved_path = loaded
    assert resolved_path == pyproject_path

    logging_cfg = data["logging"]
    assert logging_cfg["level"] == "info"
    assert logging_cfg["output"] == "stderr"
    assert logging_cfg["format"] == "json"

    core = data["core"]
    assert core["host"] == "127.0.0.1"
    assert core["outsim_port"] == 4123
    assert core["outgauge_port"] == 3000
    assert core["insim_port"] == 29999
    assert core["udp_timeout"] == pytest.approx(0.01)
    assert core["udp_retries"] == 3

    performance = data["performance"]
    assert performance["telemetry_buffer_size"] == 64
    assert performance["cache_enabled"] is True
    assert performance["max_cache_size"] == DEFAULT_DYNAMIC_CACHE_SIZE

    suggestion_defaults = data["suggest"]
    assert suggestion_defaults["car_model"] == "FZR"
    assert suggestion_defaults["track"] == "AS5"

    paths_cfg = data["paths"]
    assert paths_cfg["output_dir"] == "out"
    assert paths_cfg["pack_root"] == "."

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

    for module in (cli_module, workflows_module):
        monkeypatch.setattr(
            module,
            "_outsim_ping",
            lambda host, port, timeout: (True, f"OutSim responded from {host}:{port}"),
        )
        monkeypatch.setattr(
            module,
            "_outgauge_ping",
            lambda host, port, timeout: (True, f"OutGauge responded from {host}:{port}"),
        )
        monkeypatch.setattr(
            module,
            "_insim_handshake",
            lambda host, port, timeout: (True, "InSim responded with version 9"),
        )
        monkeypatch.setattr(
            module,
            "_check_setups_directory",
            lambda path: (True, "Write permissions confirmed in /fake/setups"),
        )

    result = run_cli(["diagnose", str(cfg_path), "--timeout", "0.05"])
    captured = capsys.readouterr()
    assert "Status: ok" in captured.out
    assert "OutSim responded" in result
    assert "OutGauge responded" in result
    assert "InSim responded" in result
    assert "Write permissions confirmed" in result


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

    monkeypatch.setattr(cli_module, "_copy_to_clipboard", fake_copy)
    monkeypatch.setattr(workflows_module, "_copy_to_clipboard", fake_copy)
    for module in (cli_module, workflows_module):
        monkeypatch.setattr(
            module,
            "_check_setups_directory",
            lambda path: (True, "Write permissions confirmed in /fake/setups"),
        )

    with pytest.raises(SystemExit) as excinfo:
        run_cli(["diagnose", str(cfg_path)])

    captured = capsys.readouterr()
    assert "/outsim 1 127.0.0.1 4123" in captured.out
    assert "/outgauge 1 127.0.0.1 3000" in captured.out
    assert "Commands copied to the clipboard." in captured.out
    assert copied and "/outsim 1 127.0.0.1 4123" in copied[0]
    assert "OutSim Mode" in str(excinfo.value.__cause__)


def test_profiles_persist_and_adjust(
    tmp_path: Path,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(
        """
[tool.tnfr_lfs.suggest]
car_model = "FZR"
track = "AS5"

[tool.tnfr_lfs.write_set]
car_model = "FZR"

[tool.tnfr_lfs.paths]
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

    run_cli_in_tmp(
        ["suggest", str(baseline_path), "--export", "json"],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    profiles_path = tmp_path / "profiles.toml"
    assert profiles_path.exists()

    run_cli_in_tmp(
        ["write-set", str(baseline_path), "--car-model", "FZR"],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

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

    run_cli_in_tmp(
        ["suggest", str(improved_path), "--export", "json"],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

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
) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    pack_root = create_cli_config_pack(tmp_path / "pack")
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(
        """
[tool.tnfr_lfs.analyze]
car_model = "ABC"

[tool.tnfr_lfs.suggest]
car_model = "ABC"
track = "AS5"

[tool.tnfr_lfs.write_set]
car_model = "ABC"

[tool.tnfr_lfs.osd]
car_model = "ABC"
track = "AS5"
""",
        encoding="utf8",
    )
    pack_arg = str(pack_root)

    run_cli_in_tmp(
        [
            "baseline",
            str(baseline_path),
            "--simulate",
            str(synthetic_stint_path),
            "--config",
            str(config_path),
            "--pack-root",
            pack_arg,
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    payload = json.loads(
        run_cli_in_tmp(
            [
                "analyze",
                str(baseline_path),
                "--export",
                "json",
                "--config",
                str(config_path),
                "--pack-root",
                pack_arg,
            ],
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    )

    assert payload["car"]["abbrev"] == "ABC"
    assert payload["tnfr_targets"]["meta"]["id"] == "custom-profile"
    assert payload["tnfr_targets"]["targets"]["balance"]["delta_nfr"] == pytest.approx(
        0.42, rel=1e-3
    )
    assert (tmp_path / "profiles.toml").exists()

@pytest.mark.parametrize(
    ("failing_stub", "stub_result", "expected_message"),
    (
        (
            "_outsim_ping",
            (False, "No response from OutSim 127.0.0.1:4123 after 0.05s"),
            "No response from OutSim",
        ),
        (
            "_check_setups_directory",
            (False, "No write permissions in setups"),
            "No write permissions",
        ),
    ),
)
def test_diagnose_reports_errors(
    failing_stub: str,
    stub_result: tuple[bool, str],
    expected_message: str,
    capsys,
    prepare_diagnose_environment: Callable[[Mapping[str, tuple[bool, str]] | None], Path],
) -> None:
    cfg_path = prepare_diagnose_environment({failing_stub: stub_result})

    with pytest.raises(SystemExit) as excinfo:
        run_cli(["diagnose", str(cfg_path)])

    captured = capsys.readouterr()
    assert expected_message in captured.err
    assert expected_message in str(excinfo.value.__cause__)
