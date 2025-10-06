"""Argument parsing helpers for the TNFR × LFS CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from ..exporters import REPORT_ARTIFACT_FORMATS, exporters_registry
from . import compare as compare_command
from . import pareto as pareto_command
from .workflows import (
    _default_car_model,
    _default_track_name,
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


def _validated_export(value: Any, *, fallback: str) -> str:
    if isinstance(value, str) and value in exporters_registry:
        return value
    return fallback


def _add_export_argument(
    parser: argparse.ArgumentParser, *, default: str, help_text: str
) -> None:
    parser.add_argument(
        "--export",
        dest="exports",
        choices=sorted(exporters_registry.keys()),
        action="append",
        help=f"{help_text} Puede repetirse para combinar salidas.",
    )
    parser.set_defaults(exports=None, export_default=default)


def build_parser(config: Mapping[str, Any] | None = None) -> argparse.ArgumentParser:
    config = dict(config or {})
    parser = argparse.ArgumentParser(description="TNFR Load & Force Synthesis")
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=None,
        help="Ruta del fichero de configuración TOML a utilizar.",
    )
    parser.add_argument(
        "--pack-root",
        dest="pack_root",
        type=Path,
        default=None,
        help=(
            "Directorio raíz de un pack TNFR × LFS con config/ y data/. "
            "Sobrescribe paths.pack_root."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    template_parser = subparsers.add_parser(
        "template",
        help="Generate ΔNFR, slip and yaw presets for analyze/report workflows.",
    )
    template_parser.add_argument(
        "--car",
        dest="car_model",
        default=_default_car_model(config),
        help="Car model used to resolve the preset (default: perfil actual).",
    )
    template_parser.add_argument(
        "--track",
        dest="track",
        default=_default_track_name(config),
        help="Identificador de pista usado para seleccionar el perfil (default: actual).",
    )
    template_parser.set_defaults(handler=_handle_template)

    telemetry_cfg = dict(config.get("telemetry", {}))
    osd_cfg = dict(config.get("osd", {}))
    osd_parser = subparsers.add_parser(
        "osd",
        help="Render the live ΔNFR HUD inside Live for Speed via InSim buttons.",
    )
    osd_parser.add_argument(
        "--host",
        default=str(osd_cfg.get("host", telemetry_cfg.get("host", "127.0.0.1"))),
        help="Host where the OutSim/OutGauge broadcasters are running.",
    )
    osd_parser.add_argument(
        "--outsim-port",
        type=int,
        default=int(osd_cfg.get("outsim_port", telemetry_cfg.get("outsim_port", 4123))),
        help="Port used by the OutSim UDP stream.",
    )
    osd_parser.add_argument(
        "--outgauge-port",
        type=int,
        default=int(osd_cfg.get("outgauge_port", telemetry_cfg.get("outgauge_port", 3000))),
        help="Port used by the OutGauge UDP stream.",
    )
    osd_parser.add_argument(
        "--insim-port",
        type=int,
        default=int(osd_cfg.get("insim_port", telemetry_cfg.get("insim_port", 29999))),
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
        default=str(osd_cfg.get("car_model", _default_car_model(config))),
        help="Car model used to resolve recommendation thresholds.",
    )
    osd_parser.add_argument(
        "--track",
        default=str(osd_cfg.get("track", _default_track_name(config))),
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

    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Validate cfg.txt telemetry configuration and UDP availability.",
    )
    diagnose_parser.add_argument(
        "--host",
        default=str(telemetry_cfg.get("host", "127.0.0.1")),
        help="Host where the OutSim/OutGauge broadcasters are running.",
    )
    diagnose_parser.add_argument(
        "--outsim-port",
        type=int,
        default=int(telemetry_cfg.get("outsim_port", 4123)),
        help="Port used by the OutSim UDP stream.",
    )
    diagnose_parser.add_argument(
        "--outgauge-port",
        type=int,
        default=int(telemetry_cfg.get("outgauge_port", 3000)),
        help="Port used by the OutGauge UDP stream.",
    )
    diagnose_parser.add_argument(
        "--insim-port",
        type=int,
        default=int(telemetry_cfg.get("insim_port", 29999)),
        help="Port used by the InSim TCP control channel.",
    )
    diagnose_parser.add_argument(
        "--timeout",
        type=float,
        default=float(telemetry_cfg.get("timeout", 2.0)),
        help="Timeout in seconds to wait for UDP packets before failing.",
    )
    diagnose_parser.add_argument(
        "--retries",
        type=int,
        default=int(telemetry_cfg.get("retries", 3)),
        help="Retries when establishing the InSim TCP control channel.",
    )
    diagnose_parser.add_argument(
        "--car-model",
        default=str(telemetry_cfg.get("car_model", _default_car_model(config))),
        help="Car model used to resolve the telemetry profiles.",
    )
    diagnose_parser.add_argument(
        "--track",
        default=str(telemetry_cfg.get("track", _default_track_name(config))),
        help="Track identifier used to resolve the telemetry profiles.",
    )
    diagnose_parser.set_defaults(handler=_handle_diagnose)

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Record a telemetry baseline by capturing OutSim/OutGauge streams.",
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
        default=str(telemetry_cfg.get("format", "jsonl")),
        help="Telemetry output format (default: jsonl).",
    )
    baseline_parser.add_argument(
        "--max-samples",
        type=int,
        default=int(telemetry_cfg.get("max_samples", 3600)),
        help="Maximum number of telemetry samples to record.",
    )
    baseline_parser.add_argument(
        "--timeout",
        type=float,
        default=float(telemetry_cfg.get("timeout", 2.0)),
        help="Timeout in seconds to wait for UDP packets before failing.",
    )
    baseline_parser.add_argument(
        "--retries",
        type=int,
        default=int(telemetry_cfg.get("retries", 3)),
        help="Retries when establishing the InSim TCP control channel.",
    )
    baseline_parser.add_argument(
        "--host",
        default=str(telemetry_cfg.get("host", "127.0.0.1")),
        help="Host where the OutSim/OutGauge broadcasters are running.",
    )
    baseline_parser.add_argument(
        "--outsim-port",
        type=int,
        default=int(telemetry_cfg.get("outsim_port", 4123)),
        help="Port used by the OutSim UDP stream.",
    )
    baseline_parser.add_argument(
        "--outgauge-port",
        type=int,
        default=int(telemetry_cfg.get("outgauge_port", 3000)),
        help="Port used by the OutGauge UDP stream.",
    )
    baseline_parser.add_argument(
        "--insim-port",
        type=int,
        default=int(telemetry_cfg.get("insim_port", 29999)),
        help="Port used by the InSim TCP control channel.",
    )
    baseline_parser.set_defaults(handler=_handle_baseline)

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
    _add_export_argument(
        analyze_parser,
        default=_validated_export(analyze_cfg.get("export"), fallback="json"),
        help_text="Exporter used to render the analysis payload.",
    )
    analyze_parser.add_argument(
        "--car-model",
        default=str(analyze_cfg.get("car_model", _default_car_model(config))),
        help="Car model used to resolve the recommendation thresholds.",
    )
    analyze_parser.add_argument(
        "--track",
        default=str(analyze_cfg.get("track", _default_track_name(config))),
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
    _add_export_argument(
        suggest_parser,
        default=_validated_export(suggest_cfg.get("export"), fallback="json"),
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
    _add_export_argument(
        report_parser,
        default=_validated_export(report_cfg.get("export"), fallback="json"),
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
    _add_export_argument(
        write_set_parser,
        default=_validated_export(write_set_cfg.get("export"), fallback="markdown"),
        help_text="Exporter used to render the setup plan (default: markdown).",
    )
    write_set_parser.add_argument(
        "--car-model",
        default=str(write_set_cfg.get("car_model", _default_car_model(config))),
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
            "Nombre base del archivo .set que se guardará bajo LFS/data/setups. "
            "Debe comenzar por el prefijo del coche."
        ),
    )
    write_set_parser.set_defaults(handler=_handle_write_set)

    compare_command.register_subparser(subparsers, config=config)
    pareto_command.register_subparser(subparsers, config=config)

    return parser
