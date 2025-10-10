"""Argument parsing helpers for the TNFR × LFS CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional

from ..exporters import REPORT_ARTIFACT_FORMATS, exporters_registry
from . import compare as compare_command
from . import pareto as pareto_command
from .common import add_export_argument, default_car_model, default_track_name, validated_export
from .workflows import (
    _handle_analyze,
    _handle_baseline,
    _handle_compare,
    _handle_diagnose,
    _handle_osd,
    _handle_pareto,
    _handle_report,
    _handle_suggest,
    _handle_template,
    _handle_write_set,
)


def build_parser(config: Optional[Mapping[str, Any]] = None) -> argparse.ArgumentParser:
    config = dict(config or {})
    logging_cfg = dict(config.get("logging", {}))
    core_cfg_raw = config.get("core", {})
    if isinstance(core_cfg_raw, Mapping):
        core_cfg = dict(core_cfg_raw)
    else:
        core_cfg = {}
    performance_cfg_raw = config.get("performance", {})
    if isinstance(performance_cfg_raw, Mapping):
        performance_cfg = dict(performance_cfg_raw)
    else:
        performance_cfg = {}

    parser = argparse.ArgumentParser(
        description="TNFR × LFS – Live for Speed Load & Force Synthesis"
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=None,
        help="Path to the TOML configuration file to load.",
    )
    parser.add_argument(
        "--pack-root",
        dest="pack_root",
        type=Path,
        default=None,
        help=(
            "Root directory of a TNFR × LFS pack containing config/ and data/. "
            "Overrides paths.pack_root."
        ),
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=logging_cfg.get("level", "info"),
        help="Logging level (e.g. debug, info, warning).",
    )
    parser.add_argument(
        "--log-output",
        dest="log_output",
        default=logging_cfg.get("output", "stderr"),
        help="Logging destination (stdout, stderr or a file path).",
    )
    parser.add_argument(
        "--log-format",
        dest="log_format",
        choices=("json", "text"),
        default=logging_cfg.get("format", "json"),
        help="Logging formatter (json or text).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    template_parser = subparsers.add_parser(
        "template",
        help="Generate ΔNFR, slip and yaw presets for analyze/report workflows.",
    )
    template_parser.add_argument(
        "--car",
        dest="car_model",
        default=default_car_model(config),
        help="Car model used to resolve the preset (default: configured car model).",
    )
    template_parser.add_argument(
        "--track",
        dest="track",
        default=default_track_name(config),
        help="Track identifier used to resolve the preset (default: configured track).",
    )
    template_parser.set_defaults(handler=_handle_template)

    osd_cfg_raw = core_cfg.get("osd", {})
    if isinstance(osd_cfg_raw, Mapping):
        osd_cfg = dict(osd_cfg_raw)
    else:
        osd_cfg = {}
    try:
        udp_timeout_default = float(core_cfg.get("udp_timeout", 2.0))
    except (TypeError, ValueError):
        udp_timeout_default = 2.0
    try:
        udp_retries_default = int(core_cfg.get("udp_retries", core_cfg.get("retries", 3)))
    except (TypeError, ValueError):
        udp_retries_default = 3
    telemetry_buffer_raw = performance_cfg.get("telemetry_buffer_size")
    telemetry_buffer_default: int | None
    try:
        telemetry_buffer_default = (
            None
            if telemetry_buffer_raw is None
            else int(telemetry_buffer_raw)
        )
    except (TypeError, ValueError):
        telemetry_buffer_default = None
    if telemetry_buffer_default is not None and telemetry_buffer_default <= 0:
        telemetry_buffer_default = None
    osd_parser = subparsers.add_parser(
        "osd",
        help="Render the live ΔNFR HUD inside Live for Speed via InSim buttons.",
    )
    osd_parser.add_argument(
        "--host",
        default=str(osd_cfg.get("host", core_cfg.get("host", "127.0.0.1"))),
        help="Host where the OutSim/OutGauge broadcasters are running.",
    )
    osd_parser.add_argument(
        "--outsim-port",
        type=int,
        default=int(osd_cfg.get("outsim_port", core_cfg.get("outsim_port", 4123))),
        help="Port used by the OutSim UDP stream.",
    )
    osd_parser.add_argument(
        "--outgauge-port",
        type=int,
        default=int(osd_cfg.get("outgauge_port", core_cfg.get("outgauge_port", 3000))),
        help="Port used by the OutGauge UDP stream.",
    )
    osd_parser.add_argument(
        "--insim-port",
        type=int,
        default=int(osd_cfg.get("insim_port", core_cfg.get("insim_port", 29999))),
        help="Port used by the InSim TCP control channel.",
    )
    osd_parser.add_argument(
        "--insim-keepalive",
        type=float,
        default=float(osd_cfg.get("insim_keepalive", 5.0)),
        help="Interval in seconds between InSim keepalive packets (default: 5s).",
    )
    osd_parser.add_argument(
        "--update-rate",
        type=float,
        default=float(osd_cfg.get("update_rate", 6.0)),
        help="HUD refresh rate in Hz (default: 6).",
    )
    osd_parser.add_argument(
        "--car-model",
        default=str(osd_cfg.get("car_model", default_car_model(config))),
        help="Car model used to resolve recommendation thresholds.",
    )
    osd_parser.add_argument(
        "--track",
        default=str(osd_cfg.get("track", default_track_name(config))),
        help="Track identifier used to resolve recommendation thresholds.",
    )
    osd_parser.add_argument(
        "--layout-left",
        type=int,
        default=osd_cfg.get("layout_left"),
        help="Override the IS_BTN left coordinate (0-200).",
    )
    osd_parser.add_argument(
        "--layout-top",
        type=int,
        default=osd_cfg.get("layout_top"),
        help="Override the IS_BTN top coordinate (0-200).",
    )
    osd_parser.add_argument(
        "--layout-width",
        type=int,
        default=osd_cfg.get("layout_width"),
        help="Override the IS_BTN width (0-200).",
    )
    osd_parser.add_argument(
        "--layout-height",
        type=int,
        default=osd_cfg.get("layout_height"),
        help="Override the IS_BTN height (0-200).",
    )
    osd_parser.set_defaults(handler=_handle_osd)
    osd_parser.set_defaults(telemetry_buffer_size=telemetry_buffer_default)

    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Validate cfg.txt telemetry configuration and UDP availability.",
    )
    diagnose_parser.add_argument(
        "cfg",
        type=Path,
        help="Path to the Live for Speed cfg.txt file to validate.",
    )
    diagnose_parser.add_argument(
        "--host",
        default=str(core_cfg.get("host", "127.0.0.1")),
        help="Host where the OutSim/OutGauge broadcasters are running.",
    )
    diagnose_parser.add_argument(
        "--outsim-port",
        type=int,
        default=int(core_cfg.get("outsim_port", 4123)),
        help="Port used by the OutSim UDP stream.",
    )
    diagnose_parser.add_argument(
        "--outgauge-port",
        type=int,
        default=int(core_cfg.get("outgauge_port", 3000)),
        help="Port used by the OutGauge UDP stream.",
    )
    diagnose_parser.add_argument(
        "--insim-port",
        type=int,
        default=int(core_cfg.get("insim_port", 29999)),
        help="Port used by the InSim TCP control channel.",
    )
    diagnose_parser.add_argument(
        "--timeout",
        type=float,
        default=udp_timeout_default,
        help="Timeout in seconds to wait for UDP packets before failing.",
    )
    diagnose_parser.add_argument(
        "--retries",
        type=int,
        default=udp_retries_default,
        help="Retries when establishing the InSim TCP control channel.",
    )
    diagnose_parser.add_argument(
        "--car-model",
        default=str(core_cfg.get("car_model", default_car_model(config))),
        help="Car model used to resolve the telemetry profiles.",
    )
    diagnose_parser.add_argument(
        "--track",
        default=str(core_cfg.get("track", default_track_name(config))),
        help="Track identifier used to resolve the telemetry profiles.",
    )
    diagnose_parser.set_defaults(handler=_handle_diagnose)

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Record a telemetry baseline by capturing OutSim/OutGauge streams.",
    )
    simulate_default = core_cfg.get("simulate")
    if isinstance(simulate_default, str) and simulate_default.strip():
        simulate_default = Path(simulate_default).expanduser()
    else:
        simulate_default = None
    duration_default = core_cfg.get("duration", 45.0)
    try:
        duration_default = float(duration_default)
    except (TypeError, ValueError):
        duration_default = 45.0
    limit_default = core_cfg.get("limit")
    try:
        limit_default = int(limit_default)
    except (TypeError, ValueError):
        limit_default = None
    overlay_default = bool(core_cfg.get("overlay", False))
    insim_keepalive_default = core_cfg.get("insim_keepalive", 5.0)
    try:
        insim_keepalive_default = float(insim_keepalive_default)
    except (TypeError, ValueError):
        insim_keepalive_default = 5.0
    output_default = core_cfg.get("output")
    if isinstance(output_default, str) and output_default.strip():
        output_default = Path(output_default).expanduser()
    else:
        output_default = None
    output_dir_default = core_cfg.get("output_dir")
    if isinstance(output_dir_default, str) and output_dir_default.strip():
        output_dir_default = Path(output_dir_default).expanduser()
    else:
        output_dir_default = None
    force_default = bool(core_cfg.get("force", False))
    baseline_parser.add_argument(
        "--simulate",
        dest="simulate",
        type=Path,
        default=simulate_default,
        help="Telemetry source ingested instead of live capture (CSV/RAF/JSONL).",
    )
    baseline_parser.add_argument(
        "telemetry",
        type=Path,
        nargs="?",
        help=(
            "Path to the output file (.jsonl.gz, .jsonl, .parquet). "
            "Defaults to out/baseline.jsonl.gz if omitted."
        ),
    )
    baseline_parser.add_argument(
        "--format",
        choices=("jsonl", "parquet"),
        default=str(core_cfg.get("format", "jsonl")),
        help="Telemetry output format (default: jsonl).",
    )
    baseline_parser.add_argument(
        "--duration",
        type=float,
        default=duration_default,
        help=f"Maximum capture duration in seconds (default: {duration_default:g}).",
    )
    baseline_parser.add_argument(
        "--limit",
        type=int,
        default=limit_default,
        help="Maximum number of records ingested when simulating telemetry.",
    )
    baseline_parser.add_argument(
        "--overlay",
        action="store_true",
        default=overlay_default,
        help="Render a Live for Speed overlay while capturing live telemetry.",
    )
    baseline_parser.add_argument(
        "--insim-keepalive",
        dest="insim_keepalive",
        type=float,
        default=insim_keepalive_default,
        help=(
            "Interval in seconds between InSim keepalive packets used by the overlay"
            f" (default: {insim_keepalive_default:g})."
        ),
    )
    baseline_parser.add_argument(
        "--max-samples",
        type=int,
        default=int(core_cfg.get("max_samples", 3600)),
        help="Maximum number of telemetry samples to record.",
    )
    baseline_parser.add_argument(
        "--timeout",
        type=float,
        default=udp_timeout_default,
        help="Timeout in seconds to wait for UDP packets before failing.",
    )
    baseline_parser.add_argument(
        "--retries",
        type=int,
        default=udp_retries_default,
        help="Retries when establishing the InSim TCP control channel.",
    )
    baseline_parser.add_argument(
        "--host",
        default=str(core_cfg.get("host", "127.0.0.1")),
        help="Host where the OutSim/OutGauge broadcasters are running.",
    )
    baseline_parser.add_argument(
        "--outsim-port",
        type=int,
        default=int(core_cfg.get("outsim_port", 4123)),
        help="Port used by the OutSim UDP stream.",
    )
    baseline_parser.add_argument(
        "--outgauge-port",
        type=int,
        default=int(core_cfg.get("outgauge_port", 3000)),
        help="Port used by the OutGauge UDP stream.",
    )
    baseline_parser.add_argument(
        "--insim-port",
        type=int,
        default=int(core_cfg.get("insim_port", 29999)),
        help="Port used by the InSim TCP control channel.",
    )
    baseline_parser.add_argument(
        "--output",
        dest="output",
        type=Path,
        default=output_default,
        help="Destination file or directory for the recorded baseline.",
    )
    baseline_parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=output_dir_default,
        help="Directory used when auto-generating baseline filenames.",
    )
    baseline_parser.add_argument(
        "--force",
        action="store_true",
        default=force_default,
        help="Overwrite existing files or reuse timestamped runs when colliding.",
    )
    baseline_parser.set_defaults(handler=_handle_baseline)
    baseline_parser.set_defaults(telemetry_buffer_size=telemetry_buffer_default)

    analyze_cfg = dict(config.get("analyze", {}))
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Compute ΔNFR, sense index and operator breakdown for a telemetry run.",
    )
    analyze_parser.add_argument(
        "telemetry",
        type=Path,
        nargs="?",
        help=(
            "Path to a baseline file (.raf, .csv, .jsonl, .json, .parquet). "
            "Required unless --replay-csv-bundle is provided."
        ),
    )
    analyze_parser.add_argument(
        "--replay-csv-bundle",
        dest="replay_csv_bundle",
        type=Path,
        default=None,
        help=(
            "Directory or ZIP bundle with CSV telemetry exported from the replay analyzer."
        ),
    )
    add_export_argument(
        analyze_parser,
        default=validated_export(analyze_cfg.get("export"), fallback="json"),
        help_text="Exporter used to render the analysis payload.",
    )
    analyze_parser.add_argument(
        "--car-model",
        default=str(analyze_cfg.get("car_model", default_car_model(config))),
        help="Car model used to resolve the recommendation thresholds.",
    )
    analyze_parser.add_argument(
        "--track",
        default=str(analyze_cfg.get("track", default_track_name(config))),
        help="Track identifier used to resolve the recommendation thresholds.",
    )
    analyze_parser.add_argument(
        "--target-delta",
        type=float,
        default=float(analyze_cfg.get("target_delta", 0.0)),
        help="Target ΔNFR objective used by the operators orchestration.",
    )
    analyze_parser.add_argument(
        "--target-si",
        type=float,
        default=float(analyze_cfg.get("target_si", 0.75)),
        help="Target sense index objective used by the operators orchestration.",
    )
    analyze_parser.add_argument(
        "--coherence-window",
        type=int,
        default=int(analyze_cfg.get("coherence_window", 3)),
        help="Window length used by the coherence operator when smoothing ΔNFR.",
    )
    analyze_parser.add_argument(
        "--recursion-decay",
        type=float,
        default=float(analyze_cfg.get("recursion_decay", 0.4)),
        help="Decay factor for the recursive operator when computing hysteresis.",
    )
    analyze_parser.set_defaults(handler=_handle_analyze)

    suggest_cfg = dict(config.get("suggest", {}))
    suggest_parser = subparsers.add_parser(
        "suggest",
        help="Generate recommendations for a telemetry baseline using the rule engine.",
    )
    suggest_parser.add_argument(
        "telemetry",
        type=Path,
        nargs="?",
        help=(
            "Path to a baseline file (.raf, .csv, .jsonl, .json, .parquet). "
            "Required unless --replay-csv-bundle is provided."
        ),
    )
    suggest_parser.add_argument(
        "--replay-csv-bundle",
        dest="replay_csv_bundle",
        type=Path,
        default=None,
        help=(
            "Directory or ZIP bundle with CSV telemetry exported from the replay analyzer."
        ),
    )
    add_export_argument(
        suggest_parser,
        default=validated_export(suggest_cfg.get("export"), fallback="json"),
        help_text="Exporter used to render the recommendation payload.",
    )
    suggest_parser.add_argument(
        "--car-model",
        default=str(suggest_cfg.get("car_model", "generic")),
        help="Car model used to resolve the recommendation threshold profile.",
    )
    suggest_parser.add_argument(
        "--track",
        default=str(suggest_cfg.get("track", "generic")),
        help="Track identifier used to resolve the recommendation threshold profile.",
    )
    suggest_parser.set_defaults(handler=_handle_suggest)

    report_cfg = dict(config.get("report", {}))
    report_parser = subparsers.add_parser(
        "report",
        help="Generate ΔNFR and sense index reports linked to the exporter registry.",
    )
    report_parser.add_argument(
        "telemetry",
        type=Path,
        nargs="?",
        help=(
            "Path to a baseline file (.raf, .csv, .jsonl, .json, .parquet). "
            "Required unless --replay-csv-bundle is provided."
        ),
    )
    report_parser.add_argument(
        "--replay-csv-bundle",
        dest="replay_csv_bundle",
        type=Path,
        default=None,
        help=(
            "Directory or ZIP bundle with CSV telemetry exported from the replay analyzer."
        ),
    )
    add_export_argument(
        report_parser,
        default=validated_export(report_cfg.get("export"), fallback="json"),
        help_text="Exporter used to render the report payload.",
    )
    report_parser.add_argument(
        "--target-delta",
        type=float,
        default=float(report_cfg.get("target_delta", 0.0)),
        help="Target ΔNFR objective used by the operators orchestration.",
    )
    report_parser.add_argument(
        "--target-si",
        type=float,
        default=float(report_cfg.get("target_si", 0.75)),
        help="Target sense index objective used by the operators orchestration.",
    )
    report_parser.add_argument(
        "--coherence-window",
        type=int,
        default=int(report_cfg.get("coherence_window", 3)),
        help="Window length used by the coherence operator when smoothing ΔNFR.",
    )
    report_parser.add_argument(
        "--recursion-decay",
        type=float,
        default=float(report_cfg.get("recursion_decay", 0.4)),
        help="Decay factor for the recursive operator when computing hysteresis.",
    )
    report_parser.add_argument(
        "--report-format",
        choices=REPORT_ARTIFACT_FORMATS,
        default=str(report_cfg.get("artifact_format", "json")),
        help=(
            "Formato de los artefactos adicionales generados (json, markdown o visual)."
        ),
    )
    report_parser.set_defaults(handler=_handle_report)

    write_set_cfg = dict(config.get("write_set", {}))
    write_set_parser = subparsers.add_parser(
        "write-set",
        help="Generate setup plans and .set scaffolds from telemetry baselines.",
    )
    write_set_parser.add_argument(
        "telemetry",
        type=Path,
        nargs="?",
        help=(
            "Path to a baseline file (.raf, .csv, .jsonl, .json, .parquet). "
            "Required unless --replay-csv-bundle is provided."
        ),
    )
    write_set_parser.add_argument(
        "--replay-csv-bundle",
        dest="replay_csv_bundle",
        type=Path,
        default=None,
        help=(
            "Directory or ZIP bundle with CSV telemetry exported from the replay analyzer."
        ),
    )
    add_export_argument(
        write_set_parser,
        default=validated_export(write_set_cfg.get("export"), fallback="markdown"),
        help_text="Exporter used to render the setup plan (default: markdown).",
    )
    write_set_parser.add_argument(
        "--car-model",
        default=str(write_set_cfg.get("car_model", default_car_model(config))),
        help="Car model used to select the decision space for optimisation.",
    )
    write_set_parser.add_argument(
        "--session",
        default=None,
        help="Optional session label attached to the generated setup plan.",
    )
    write_set_parser.add_argument(
        "--set-output",
        default=None,
        help=(
            "Base name for the .set file saved under LFS/data/setups/. "
            "Must start with the car prefix."
        ),
    )
    write_set_parser.set_defaults(handler=_handle_write_set)

    compare_command.register_subparser(subparsers, config=config)
    pareto_command.register_subparser(subparsers, config=config)

    return parser
