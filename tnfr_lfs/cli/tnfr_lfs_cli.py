"""Command line entry point for TNFR × LFS."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import monotonic, sleep
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore

from ..acquisition import (
    ButtonLayout,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    InSimClient,
    OverlayManager,
    OutGaugeUDPClient,
    OutSimClient,
    OutSimUDPClient,
    TelemetryFusion,
)
from ..core.epi import EPIExtractor, TelemetryRecord
from ..core.operators import orchestrate_delta_metrics
from ..core.segmentation import Microsector, segment_microsectors
from ..exporters import exporters_registry
from ..exporters.setup_plan import SetupChange, SetupPlan
from ..recommender import RecommendationEngine, SetupPlanner


Records = List[TelemetryRecord]
Bundles = Sequence[Any]

CONFIG_ENV_VAR = "TNFR_LFS_CONFIG"
DEFAULT_CONFIG_FILENAME = "tnfr-lfs.toml"
DEFAULT_OUTPUT_DIR = Path("out")


def _validated_export(value: Any, *, fallback: str) -> str:
    if isinstance(value, str) and value in exporters_registry:
        return value
    return fallback


def load_cli_config(path: Path | None = None) -> Dict[str, Any]:
    """Load CLI defaults from ``tnfr-lfs.toml`` style files."""

    candidates: List[Path] = []
    if path is not None:
        candidates.append(path)
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / DEFAULT_CONFIG_FILENAME)
    candidates.append(Path.home() / ".config" / DEFAULT_CONFIG_FILENAME)

    resolved_candidates: Dict[Path, None] = {}
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.expanduser().resolve()
        if resolved in resolved_candidates:
            continue
        resolved_candidates[resolved] = None
        if not resolved.exists():
            continue
        with resolved.open("rb") as handle:
            data = tomllib.load(handle)
        data["_config_path"] = str(resolved)
        return data
    return {"_config_path": None}


def _resolve_output_dir(config: Mapping[str, Any]) -> Path:
    paths_cfg = config.get("paths")
    if isinstance(paths_cfg, Mapping):
        output_dir = paths_cfg.get("output_dir")
        if isinstance(output_dir, str):
            return Path(output_dir).expanduser()
    return DEFAULT_OUTPUT_DIR


def _default_car_model(config: Mapping[str, Any]) -> str:
    for section in ("analyze", "suggest", "write_set"):
        section_cfg = config.get(section)
        if isinstance(section_cfg, Mapping):
            candidate = section_cfg.get("car_model")
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    return "generic"


def _default_track_name(config: Mapping[str, Any]) -> str:
    for section in ("analyze", "suggest"):
        section_cfg = config.get(section)
        if isinstance(section_cfg, Mapping):
            candidate = section_cfg.get("track")
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    return "generic"


def _parse_lfs_cfg(cfg_path: Path) -> Dict[str, Dict[str, str]]:
    sections: Dict[str, Dict[str, str]] = {"OutSim": {}, "OutGauge": {}, "InSim": {}}
    with cfg_path.open("r", encoding="utf8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue
            normalized = line.replace("=", " ")
            parts = [token for token in normalized.split(" ") if token]
            if len(parts) < 3:
                continue
            prefix, key, value = parts[0], parts[1], " ".join(parts[2:])
            if prefix not in sections:
                continue
            sections[prefix][key] = value
    return sections


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _probe_udp_socket(host: str, port: int, timeout: float) -> Tuple[bool, str]:
    description = f"{host}:{port}"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(timeout)
            try:
                sock.bind((host, port))
            except OSError:
                try:
                    sock.bind(("0.0.0.0", 0))
                except OSError:
                    pass
                try:
                    sock.connect((host, port))
                except OSError as connect_error:
                    return False, f"No se pudo abrir UDP en {description}: {connect_error}"
                return True, f"Socket UDP conectado a {description}"
            return True, f"Socket UDP disponible en {description}"
    except OSError as exc:
        return False, f"No se pudo crear socket UDP para {description}: {exc}"


def _phase_tolerances(
    config: Mapping[str, Any], car_model: str, track_name: str
) -> Dict[str, float]:
    tolerances: Dict[str, float] = {}
    limits_cfg = config.get("limits")
    if isinstance(limits_cfg, Mapping):
        delta_cfg = limits_cfg.get("delta_nfr")
        if isinstance(delta_cfg, Mapping):
            for phase, default_value in (
                ("entry", 1.5),
                ("apex", 1.0),
                ("exit", 2.0),
            ):
                value = delta_cfg.get(phase)
                if value is None:
                    continue
                try:
                    tolerances[phase] = float(value)
                except (TypeError, ValueError):
                    tolerances[phase] = float(default_value)
    if len(tolerances) < 3:
        engine = RecommendationEngine()
        context = engine._resolve_context(car_model, track_name)  # type: ignore[attr-defined]
        profile = context.thresholds
        baseline = {
            "entry": float(profile.entry_delta_tolerance),
            "apex": float(profile.apex_delta_tolerance),
            "exit": float(profile.exit_delta_tolerance),
        }
        baseline.update(tolerances)
        tolerances = baseline
    return tolerances


_PHASE_LABELS = {"entry": "entrada", "apex": "vértice", "exit": "salida"}
_PHASE_RECOMMENDATIONS = {
    "entry": {
        "positive": "liberar presión o adelantar el reparto para estabilizar el eje delantero",
        "negative": "ganar mordida adelantando el apoyo delantero o retrasando el reparto",
    },
    "apex": {
        "positive": "aliviar rigidez lateral (barras/altura) para evitar saturar el vértice",
        "negative": "buscar más rotación aumentando apoyo lateral o ajustando convergencias",
    },
    "exit": {
        "positive": "moderar la entrega de par o endurecer el soporte trasero para contener el sobreviraje",
        "negative": "liberar el eje trasero (altura/damper) para mejorar la tracción a la salida",
    },
}


def _microsector_goal(microsector: Microsector, phase: str):
    for goal in microsector.goals:
        if goal.phase == phase:
            return goal
    return None


def _phase_recommendation(phase: str, deviation: float) -> str:
    recommendations = _PHASE_RECOMMENDATIONS.get(phase, {})
    key = "positive" if deviation > 0 else "negative"
    return recommendations.get(key, "ajustar la puesta a punto para equilibrar ΔNFR")


def _phase_deviation_messages(
    bundles: Bundles,
    microsectors: Sequence[Microsector],
    config: Mapping[str, Any],
    *,
    car_model: str,
    track_name: str,
) -> List[str]:
    if not microsectors or not bundles:
        return [
            "Sin desviaciones ΔNFR↓ relevantes; no se detectaron curvas segmentadas.",
        ]

    tolerances = _phase_tolerances(config, car_model, track_name)
    messages: List[str] = []
    bundle_count = len(bundles)
    for microsector in microsectors:
        for phase in ("entry", "apex", "exit"):
            goal = _microsector_goal(microsector, phase)
            if goal is None:
                continue
            indices = microsector.phase_indices(phase)
            samples = [
                bundles[idx].delta_nfr
                for idx in indices
                if 0 <= idx < bundle_count
            ]
            if not samples:
                continue
            actual_delta = mean(samples)
            deviation = actual_delta - float(goal.target_delta_nfr)
            tolerance = float(tolerances.get(phase, 0.0))
            if abs(deviation) <= tolerance:
                continue
            label = _PHASE_LABELS.get(phase, phase)
            direction = "exceso" if deviation > 0 else "déficit"
            recommendation = _phase_recommendation(phase, deviation)
            messages.append(
                (
                    f"Curva {microsector.index + 1} ({label}): ΔNFR↓ medio "
                    f"{actual_delta:+.2f} vs objetivo {goal.target_delta_nfr:+.2f} "
                    f"({direction} {abs(deviation):.2f}, tolerancia ±{tolerance:.2f}). "
                    f"Sugerencia: {recommendation}."
                )
            )
    if not messages:
        return [
            "Sin desviaciones ΔNFR↓ relevantes por fase; mantener la referencia actual.",
        ]
    return messages


def _sense_index_map(
    bundles: Bundles, microsectors: Sequence[Microsector]
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    bundle_count = len(bundles)
    for microsector in microsectors:
        entry: Dict[str, Any] = {
            "microsector": microsector.index,
            "label": f"Curva {microsector.index + 1}",
            "sense_index": {},
        }
        aggregate: List[float] = []
        for phase in ("entry", "apex", "exit"):
            indices = microsector.phase_indices(phase)
            samples = [
                bundles[idx].sense_index
                for idx in indices
                if 0 <= idx < bundle_count
            ]
            if samples:
                avg_value = mean(samples)
                aggregate.extend(samples)
            else:
                avg_value = 0.0
            entry["sense_index"][phase] = round(float(avg_value), 4)
        entry["sense_index"]["overall"] = round(
            float(mean(aggregate)) if aggregate else 0.0,
            4,
        )
        results.append(entry)
    return results


def _spectrum(values: Sequence[float], bins: int = 12) -> Dict[str, Any]:
    if not values:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "rms": 0.0,
            "centres": [],
            "counts": [],
        }
    minimum = min(values)
    maximum = max(values)
    average = mean(values)
    rms = math.sqrt(sum(value * value for value in values) / len(values))
    if maximum - minimum <= 1e-9:
        centres = [float(minimum)]
        counts = [len(values)]
    else:
        step = (maximum - minimum) / bins
        centres = [float(minimum + (index + 0.5) * step) for index in range(bins)]
        counts = [0 for _ in range(bins)]
        for value in values:
            if step <= 0:
                bucket = 0
            else:
                bucket = int((value - minimum) / step)
                if bucket >= bins:
                    bucket = bins - 1
            counts[bucket] += 1
    return {
        "min": float(minimum),
        "max": float(maximum),
        "mean": float(average),
        "rms": float(rms),
        "centres": centres,
        "counts": counts,
    }


def _yaw_roll_spectrum(records: Records) -> Dict[str, Any]:
    yaw_values = [record.yaw for record in records]
    roll_values = [record.roll for record in records]
    return {
        "yaw": _spectrum(yaw_values),
        "roll": _spectrum(roll_values),
    }


def _generate_out_reports(
    records: Records,
    bundles: Bundles,
    microsectors: Sequence[Microsector],
    destination: Path,
) -> Dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    sense_map = _sense_index_map(bundles, microsectors)
    spectrum = _yaw_roll_spectrum(records)
    sense_path = destination / "sense_index_map.json"
    spectrum_path = destination / "yaw_roll_spectrum.json"
    with sense_path.open("w", encoding="utf8") as handle:
        json.dump(sense_map, handle, indent=2, sort_keys=True)
    with spectrum_path.open("w", encoding="utf8") as handle:
        json.dump(spectrum, handle, indent=2, sort_keys=True)
    return {
        "sense_index_map": {"path": str(sense_path), "data": sense_map},
        "yaw_roll_spectrum": {"path": str(spectrum_path), "data": spectrum},
    }

_TELEMETRY_DEFAULTS: Mapping[str, Any] = {
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
    "brake_pressure": 0.0,
    "locking": 0.0,
    "speed": 0.0,
    "yaw_rate": 0.0,
    "slip_angle": 0.0,
    "steer": 0.0,
    "throttle": 0.0,
    "gear": 0,
    "vertical_load_front": 0.0,
    "vertical_load_rear": 0.0,
    "mu_eff_front": 0.0,
    "mu_eff_rear": 0.0,
    "suspension_travel_front": 0.0,
    "suspension_travel_rear": 0.0,
    "suspension_velocity_front": 0.0,
    "suspension_velocity_rear": 0.0,
}


def _coerce_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    data: Dict[str, Any] = dict(_TELEMETRY_DEFAULTS)
    data.update(payload)
    return data


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

    subparsers = parser.add_subparsers(dest="command", required=True)

    telemetry_cfg = dict(config.get("telemetry", {}))
    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Validate cfg.txt telemetry configuration and UDP availability.",
    )
    diagnose_parser.add_argument(
        "cfg",
        type=Path,
        help="Ruta al fichero cfg.txt de Live for Speed.",
    )
    diagnose_parser.add_argument(
        "--timeout",
        type=float,
        default=0.25,
        help="Tiempo máximo (s) para probar cada socket UDP.",
    )
    diagnose_parser.set_defaults(handler=_handle_diagnose)

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Capture telemetry from UDP clients or simulation data and persist it.",
    )
    baseline_parser.add_argument("output", type=Path, help="Destination file for the baseline")
    baseline_parser.add_argument(
        "--format",
        choices=("jsonl", "parquet"),
        default=str(telemetry_cfg.get("format", "jsonl")),
        help="Persistence format used for the baseline (default: jsonl).",
    )
    baseline_parser.add_argument(
        "--duration",
        type=float,
        default=float(telemetry_cfg.get("duration", 30.0)),
        help="Capture duration in seconds when using live UDP acquisition.",
    )
    baseline_parser.add_argument(
        "--max-samples",
        type=int,
        default=int(telemetry_cfg.get("max_samples", 10_000)),
        help="Maximum number of samples to collect from UDP clients.",
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
    baseline_parser.add_argument(
        "--timeout",
        type=float,
        default=float(telemetry_cfg.get("timeout", DEFAULT_TIMEOUT)),
        help="Polling timeout for the UDP clients.",
    )
    baseline_parser.add_argument(
        "--retries",
        type=int,
        default=int(telemetry_cfg.get("retries", DEFAULT_RETRIES)),
        help="Number of retries performed by the UDP clients while polling.",
    )
    baseline_parser.add_argument(
        "--insim-keepalive",
        type=float,
        default=float(telemetry_cfg.get("insim_keepalive", 5.0)),
        help="Interval in seconds between InSim keepalive packets.",
    )
    baseline_parser.add_argument(
        "--simulate",
        type=Path,
        help="Telemetry CSV file used in simulation mode (bypasses UDP capture).",
    )
    baseline_parser.add_argument(
        "--limit",
        type=int,
        default=(
            None
            if telemetry_cfg.get("limit") is None
            else int(telemetry_cfg.get("limit"))
        ),
        help="Optional limit of samples to persist when using simulation data.",
    )
    baseline_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )
    baseline_parser.add_argument(
        "--overlay",
        action="store_true",
        help="Display a Live for Speed overlay while capturing baselines.",
    )
    baseline_parser.set_defaults(handler=_handle_baseline)

    analyze_cfg = dict(config.get("analyze", {}))
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyse a telemetry baseline and export ΔNFR/Si insights."
    )
    analyze_parser.add_argument("telemetry", type=Path, help="Path to a baseline file or CSV.")
    analyze_parser.add_argument(
        "--export",
        choices=sorted(exporters_registry.keys()),
        default=_validated_export(analyze_cfg.get("export"), fallback="json"),
        help="Exporter used to render the analysis results.",
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
    suggest_parser.add_argument("telemetry", type=Path, help="Path to a baseline file or CSV.")
    suggest_parser.add_argument(
        "--export",
        choices=sorted(exporters_registry.keys()),
        default=_validated_export(suggest_cfg.get("export"), fallback="json"),
        help="Exporter used to render the recommendation payload.",
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
    report_parser.add_argument("telemetry", type=Path, help="Path to a baseline file or CSV.")
    report_parser.add_argument(
        "--export",
        choices=sorted(exporters_registry.keys()),
        default=_validated_export(report_cfg.get("export"), fallback="json"),
        help="Exporter used to render the report payload.",
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
    report_parser.set_defaults(handler=_handle_report)

    write_set_cfg = dict(config.get("write_set", {}))
    write_set_parser = subparsers.add_parser(
        "write-set",
        help="Create a setup plan by combining optimisation with recommendations.",
    )
    write_set_parser.add_argument(
        "telemetry", type=Path, help="Path to a baseline file or CSV containing telemetry."
    )
    write_set_parser.add_argument(
        "--export",
        choices=sorted(exporters_registry.keys()),
        default=_validated_export(write_set_cfg.get("export"), fallback="markdown"),
        help="Exporter used to render the setup plan (default: markdown).",
    )
    write_set_parser.add_argument(
        "--car-model",
        default=str(write_set_cfg.get("car_model", "generic_gt")),
        help="Car model used to select the decision space for optimisation.",
    )
    write_set_parser.add_argument(
        "--session",
        default=None,
        help="Optional session label attached to the generated setup plan.",
    )
    write_set_parser.set_defaults(handler=_handle_write_set)

    return parser


def _handle_diagnose(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    cfg_path: Path = namespace.cfg.expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"No se encontró el fichero cfg.txt en {cfg_path}")

    sections = _parse_lfs_cfg(cfg_path)
    timeout = float(namespace.timeout)
    errors: List[str] = []
    successes: List[str] = []

    outsim_mode = _coerce_int(sections["OutSim"].get("Mode") or sections["OutSim"].get("Enable"))
    if outsim_mode != 1:
        errors.append("OutSim Mode debe ser 1 para habilitar la telemetría")
    outsim_port = _coerce_int(sections["OutSim"].get("Port"))
    outsim_host = sections["OutSim"].get("IP", "127.0.0.1")
    if outsim_port is None:
        errors.append("OutSim Port no está definido en cfg.txt")
    elif outsim_mode == 1:
        ok, message = _probe_udp_socket(outsim_host, outsim_port, timeout)
        if ok:
            successes.append(f"OutSim listo: {message}")
        else:
            errors.append(f"OutSim falló: {message}")

    outgauge_mode = _coerce_int(
        sections["OutGauge"].get("Mode") or sections["OutGauge"].get("Enable")
    )
    if outgauge_mode != 1:
        errors.append("OutGauge Mode debe ser 1 para habilitar la transmisión")
    outgauge_port = _coerce_int(sections["OutGauge"].get("Port"))
    outgauge_host = sections["OutGauge"].get("IP", outsim_host)
    if outgauge_port is None:
        errors.append("OutGauge Port no está definido en cfg.txt")
    elif outgauge_mode == 1:
        ok, message = _probe_udp_socket(outgauge_host, outgauge_port, timeout)
        if ok:
            successes.append(f"OutGauge listo: {message}")
        else:
            errors.append(f"OutGauge falló: {message}")

    insim_port = _coerce_int(sections["InSim"].get("Port"))
    insim_host = sections["InSim"].get("IP", outsim_host)
    if insim_port is not None:
        ok, message = _probe_udp_socket(insim_host, insim_port, timeout)
        if ok:
            successes.append(f"InSim alcanzable: {message}")
        else:
            errors.append(f"InSim falló: {message}")

    header = f"Diagnóstico de cfg.txt en {cfg_path}"
    if errors:
        summary = [header, "Estado: errores detectados"]
        summary.extend(f"- {error}" for error in errors)
        if successes:
            summary.append("Detalles adicionales:")
            summary.extend(f"  * {success}" for success in successes)
        message = "\n".join(summary)
        print(message)
        raise ValueError(message)

    summary = [header, "Estado: correcto"]
    summary.extend(f"- {success}" for success in successes)
    message = "\n".join(summary)
    print(message)
    return message


def _handle_baseline(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    if namespace.output.exists() and not namespace.force:
        raise FileExistsError(
            f"Baseline destination {namespace.output} already exists. Use --force to overwrite."
        )

    overlay: OverlayManager | None = None
    records: Records = []
    if namespace.overlay and namespace.simulate is not None:
        raise ValueError("--overlay solo está disponible con captura en vivo (sin --simulate)")

    try:
        if namespace.overlay:
            overlay_client = InSimClient(
                host=namespace.host,
                port=namespace.insim_port,
                timeout=namespace.timeout,
                keepalive_interval=namespace.insim_keepalive,
                app_name="TNFR Baseline",
            )
            layout = ButtonLayout(left=10, top=10, width=180, height=30, click_id=7)
            overlay = OverlayManager(overlay_client, layout=layout)
            overlay.connect()
            overlay.show(
                [
                    "Capturando baseline",
                    f"Duración máx: {int(namespace.duration)} s",
                    f"Muestras objetivo: {namespace.max_samples}",
                ]
            )

        if namespace.simulate is not None:
            records = OutSimClient().ingest(namespace.simulate)
            if namespace.limit is not None:
                records = records[: namespace.limit]
        else:
            heartbeat = overlay.tick if overlay is not None else None
            records = _capture_udp_samples(
                duration=namespace.duration,
                max_samples=namespace.max_samples,
                host=namespace.host,
                outsim_port=namespace.outsim_port,
                outgauge_port=namespace.outgauge_port,
                timeout=namespace.timeout,
                retries=namespace.retries,
                heartbeat=heartbeat,
            )
    finally:
        if overlay is not None:
            overlay.close()

    if not records:
        message = "No telemetry samples captured."
        print(message)
        return message

    _persist_records(records, namespace.output, namespace.format)
    message = (
        f"Baseline saved {len(records)} samples to {namespace.output} "
        f"({namespace.format})."
    )
    print(message)
    return message


def _handle_analyze(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    records = _load_records(namespace.telemetry)
    bundles, microsectors = _compute_insights(records)
    metrics = orchestrate_delta_metrics(
        [records],
        namespace.target_delta,
        namespace.target_si,
        coherence_window=namespace.coherence_window,
        recursion_decay=namespace.recursion_decay,
    )
    phase_messages = _phase_deviation_messages(
        bundles,
        microsectors,
        config,
        car_model=_default_car_model(config),
        track_name=_default_track_name(config),
    )
    reports = _generate_out_reports(
        records,
        bundles,
        microsectors,
        _resolve_output_dir(config) / namespace.telemetry.stem,
    )
    payload: Dict[str, Any] = {
        "series": bundles,
        "microsectors": microsectors,
        "telemetry_samples": len(records),
        "metrics": {key: value for key, value in metrics.items() if key != "bundles"},
        "smoothed_series": metrics.get("bundles", []),
        "phase_messages": phase_messages,
        "reports": reports,
    }
    return _render_payload(payload, namespace.export)


def _handle_suggest(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    records = _load_records(namespace.telemetry)
    bundles, microsectors = _compute_insights(records)
    engine = RecommendationEngine()
    recommendations = engine.generate(
        bundles, microsectors, car_model=namespace.car_model, track_name=namespace.track
    )
    phase_messages = _phase_deviation_messages(
        bundles,
        microsectors,
        config,
        car_model=namespace.car_model,
        track_name=namespace.track,
    )
    reports = _generate_out_reports(
        records,
        bundles,
        microsectors,
        _resolve_output_dir(config) / namespace.telemetry.stem,
    )
    payload = {
        "series": bundles,
        "microsectors": microsectors,
        "recommendations": recommendations,
        "car_model": namespace.car_model,
        "track": namespace.track,
        "phase_messages": phase_messages,
        "reports": reports,
    }
    return _render_payload(payload, namespace.export)


def _handle_report(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    records = _load_records(namespace.telemetry)
    bundles, microsectors = _compute_insights(records)
    metrics = orchestrate_delta_metrics(
        [records],
        namespace.target_delta,
        namespace.target_si,
        coherence_window=namespace.coherence_window,
        recursion_decay=namespace.recursion_decay,
    )
    reports = _generate_out_reports(
        records,
        bundles,
        microsectors,
        _resolve_output_dir(config) / namespace.telemetry.stem,
    )
    payload: Dict[str, Any] = {
        "objectives": metrics.get("objectives", {}),
        "delta_nfr": metrics.get("delta_nfr", 0.0),
        "sense_index": metrics.get("sense_index", 0.0),
        "dissonance": metrics.get("dissonance", 0.0),
        "coupling": metrics.get("coupling", 0.0),
        "resonance": metrics.get("resonance", 0.0),
        "recursive_trace": metrics.get("recursive_trace", []),
        "series": bundles if bundles else metrics.get("bundles", []),
        "reports": reports,
    }
    return _render_payload(payload, namespace.export)


def _handle_write_set(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    records = _load_records(namespace.telemetry)
    bundles, microsectors = _compute_insights(records)
    planner = SetupPlanner()
    plan = planner.plan(bundles, microsectors, car_model=namespace.car_model)

    aggregated_rationales = [rec.rationale for rec in plan.recommendations if rec.rationale]
    aggregated_effects = [rec.message for rec in plan.recommendations if rec.message]
    if not aggregated_rationales:
        aggregated_rationales = ["Optimización de objetivo Si/ΔNFR"]
    if not aggregated_effects:
        aggregated_effects = ["Mejora equilibrada del coche"]

    changes = [
        SetupChange(
            parameter=name,
            delta=value,
            rationale="; ".join(aggregated_rationales),
            expected_effect="; ".join(aggregated_effects),
        )
        for name, value in sorted(plan.decision_vector.items())
    ]

    setup_plan = SetupPlan(
        car_model=namespace.car_model,
        session=namespace.session,
        changes=tuple(changes),
        rationales=tuple(aggregated_rationales),
        expected_effects=tuple(aggregated_effects),
    )

    payload = {
        "setup_plan": setup_plan,
        "objective_value": plan.objective_value,
        "recommendations": plan.recommendations,
        "series": plan.telemetry,
    }
    return _render_payload(payload, namespace.export)


def _render_payload(payload: Mapping[str, Any], exporter_name: str) -> str:
    exporter = exporters_registry[exporter_name]
    rendered = exporter(dict(payload))
    print(rendered)
    return rendered


def _compute_insights(records: Records) -> tuple[Bundles, Sequence[Microsector]]:
    if not records:
        return [], []
    extractor = EPIExtractor()
    bundles = extractor.extract(records)
    if not bundles:
        return bundles, []
    microsectors = segment_microsectors(records, bundles)
    return bundles, microsectors


def _capture_udp_samples(
    *,
    duration: float,
    max_samples: int,
    host: str,
    outsim_port: int,
    outgauge_port: int,
    timeout: float,
    retries: int,
    heartbeat: Callable[[], None] | None = None,
) -> Records:
    fusion = TelemetryFusion()
    records: Records = []
    deadline = monotonic() + max(duration, 0.0)

    with OutSimUDPClient(
        host=host, port=outsim_port, timeout=timeout, retries=retries
    ) as outsim, OutGaugeUDPClient(
        host=host, port=outgauge_port, timeout=timeout, retries=retries
    ) as outgauge:
        while len(records) < max_samples and monotonic() < deadline:
            if heartbeat is not None:
                heartbeat()
            outsim_packet = outsim.recv()
            outgauge_packet = outgauge.recv()
            if outsim_packet is None or outgauge_packet is None:
                if heartbeat is not None:
                    heartbeat()
                sleep(timeout)
                continue
            record = fusion.fuse(outsim_packet, outgauge_packet)
            records.append(record)

    return records


def _persist_records(records: Records, destination: Path, fmt: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with destination.open("w", encoding="utf8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), sort_keys=True))
                handle.write("\n")
        return

    if fmt == "parquet":
        serialised = [asdict(record) for record in records]
        try:
            import pandas as pd  # type: ignore

            frame = pd.DataFrame(serialised)
            frame.to_parquet(destination, index=False)
            return
        except ModuleNotFoundError:
            with destination.open("w", encoding="utf8") as handle:
                json.dump(serialised, handle, sort_keys=True)
            return

    raise ValueError(f"Unsupported format '{fmt}'.")


def _load_records(source: Path) -> Records:
    if not source.exists():
        raise FileNotFoundError(f"Telemetry source {source} does not exist")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return OutSimClient().ingest(source)
    if suffix == ".jsonl":
        records: Records = []
        with source.open("r", encoding="utf8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                records.append(TelemetryRecord(**_coerce_payload(payload)))
        return records
    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore

            frame = pd.read_parquet(source)
            data = frame.to_dict(orient="records")
        except ModuleNotFoundError:
            with source.open("r", encoding="utf8") as handle:
                data = json.load(handle)
        return [TelemetryRecord(**_coerce_payload(item)) for item in data]
    if suffix == ".json":
        with source.open("r", encoding="utf8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return [TelemetryRecord(**_coerce_payload(item)) for item in data]
        raise ValueError(f"JSON telemetry source {source} must contain a list of samples")

    raise ValueError(f"Unsupported telemetry format: {source}")


def run_cli(args: Sequence[str] | None = None) -> str:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", dest="config_path", type=Path, default=None
    )
    preliminary, remaining = config_parser.parse_known_args(args)
    config = load_cli_config(preliminary.config_path)
    parser = build_parser(config)
    parser.set_defaults(config_path=preliminary.config_path)
    namespace = parser.parse_args(remaining)
    namespace.config = config
    namespace.config_path = (
        getattr(namespace, "config_path", None)
        or preliminary.config_path
        or config.get("_config_path")
    )
    handler: Callable[[argparse.Namespace, Mapping[str, Any]], str] = getattr(
        namespace, "handler"
    )
    return handler(namespace, config=config)


def main() -> None:  # pragma: no cover - thin wrapper
    run_cli()


if __name__ == "__main__":  # pragma: no cover - CLI invocation guard
    main()
