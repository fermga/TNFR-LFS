"""Command line entry point for TNFR × LFS."""

from __future__ import annotations

import argparse
import json
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
from .osd import OSDController
from ..core.epi import EPIExtractor, TelemetryRecord, NU_F_NODE_DEFAULTS
from ..core.resonance import analyse_modal_resonance
from ..core.operators import orchestrate_delta_metrics
from ..core.segmentation import Microsector, segment_microsectors
from ..exporters import exporters_registry
from ..exporters.setup_plan import SetupChange, SetupPlan
from ..io import logs
from ..recommender import RecommendationEngine, SetupPlanner
from ..recommender.rules import ThresholdProfile


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


def _resolve_baseline_destination(
    namespace: argparse.Namespace, config: Mapping[str, Any]
) -> Path:
    fmt = namespace.format
    suffix = ".jsonl" if fmt == "jsonl" else ".parquet"
    output_dir_arg: Path | None = getattr(namespace, "output_dir", None)
    if output_dir_arg is not None:
        output_dir_arg = output_dir_arg.expanduser()
    auto_kwargs = {
        "car_model": _default_car_model(config),
        "track_name": _default_track_name(config),
        "output_dir": output_dir_arg,
        "suffix": suffix,
        "force": namespace.force,
    }

    output_arg: Path | None = namespace.output
    if output_arg is None:
        return logs.prepare_run_destination(**auto_kwargs)

    output_path = output_arg.expanduser()
    if output_path.exists() and output_path.is_dir():
        auto_kwargs["output_dir"] = output_path
        return logs.prepare_run_destination(**auto_kwargs)

    if output_path.suffix == "":
        auto_kwargs["output_dir"] = output_path
        return logs.prepare_run_destination(**auto_kwargs)

    if output_dir_arg is not None and not output_path.is_absolute():
        destination = output_dir_arg.expanduser() / output_path
    else:
        destination = output_path

    if destination.exists() and destination.is_dir():
        auto_kwargs["output_dir"] = destination
        return logs.prepare_run_destination(**auto_kwargs)

    if destination.exists() and not namespace.force:
        raise FileExistsError(
            f"Baseline destination {destination} already exists. Use --force to overwrite."
        )

    return destination


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


def _format_window(window: Tuple[float, float]) -> str:
    return f"[{window[0]:.3f}, {window[1]:.3f}]"


def _profile_phase_templates(profile: ThresholdProfile) -> Dict[str, Dict[str, object]]:
    templates: Dict[str, Dict[str, object]] = {}
    for phase, target in profile.phase_targets.items():
        templates[phase] = {
            "target_delta_nfr": round(float(target.target_delta_nfr), 3),
            "slip_lat_window": [
                round(float(target.slip_lat_window[0]), 3),
                round(float(target.slip_lat_window[1]), 3),
            ],
            "slip_long_window": [
                round(float(target.slip_long_window[0]), 3),
                round(float(target.slip_long_window[1]), 3),
            ],
            "yaw_rate_window": [
                round(float(target.yaw_rate_window[0]), 3),
                round(float(target.yaw_rate_window[1]), 3),
            ],
        }
    return templates


def _phase_templates_from_config(
    config: Mapping[str, Any], section: str
) -> Dict[str, Dict[str, float | List[float]]]:
    section_cfg = config.get(section)
    if not isinstance(section_cfg, Mapping):
        return {}
    templates_cfg = section_cfg.get("phase_templates")
    if not isinstance(templates_cfg, Mapping):
        return {}
    templates: Dict[str, Dict[str, float | List[float]]] = {}
    for phase, payload in templates_cfg.items():
        if not isinstance(payload, Mapping):
            continue
        entry: Dict[str, float | List[float]] = {}
        target_delta = payload.get("target_delta_nfr")
        if target_delta is not None:
            try:
                entry["target_delta_nfr"] = float(target_delta)
            except (TypeError, ValueError):
                pass
        for key in ("slip_lat_window", "slip_long_window", "yaw_rate_window"):
            window = payload.get(key)
            if isinstance(window, Sequence) and len(window) == 2:
                try:
                    entry[key] = [float(window[0]), float(window[1])]
                except (TypeError, ValueError):
                    continue
        if entry:
            templates[str(phase)] = entry
    return templates


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
        filtered_payload = {
            key: round(float(value), 4)
            for key, value in microsector.filtered_measures.items()
        }
        if "grip_rel" not in filtered_payload:
            filtered_payload["grip_rel"] = round(float(microsector.grip_rel), 4)
        entry["filtered_measures"] = filtered_payload
        entry["grip_rel"] = filtered_payload.get("grip_rel", 0.0)
        entry["filtered_style_index"] = filtered_payload.get("style_index")
        occupancy_payload: Dict[str, Dict[str, float]] = {}
        for phase in ("entry", "apex", "exit"):
            phase_occupancy = microsector.window_occupancy.get(phase, {})
            occupancy_payload[phase] = {
                metric: round(float(value), 4)
                for metric, value in phase_occupancy.items()
            }
        entry["window_occupancy"] = occupancy_payload
        mutation = microsector.last_mutation
        if mutation:
            entry["mutation"] = {
                "archetype": str(mutation.get("archetype", "")),
                "mutated": bool(mutation.get("mutated", False)),
                "entropy": round(float(mutation.get("entropy", 0.0)), 4),
                "entropy_delta": round(float(mutation.get("entropy_delta", 0.0)), 4),
                "style_delta": round(float(mutation.get("style_delta", 0.0)), 4),
                "phase": mutation.get("phase"),
            }
        else:
            entry["mutation"] = None
        results.append(entry)
    return results


def _delta_breakdown_summary(bundles: Bundles) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "samples": len(bundles),
        "per_node": {
            node: {"delta_nfr_total": 0.0, "breakdown": {}}
            for node in NU_F_NODE_DEFAULTS
        },
    }
    if not bundles:
        return summary

    for bundle in bundles:
        breakdown = getattr(bundle, "delta_breakdown", {}) or {}
        for node, features in breakdown.items():
            node_entry = summary["per_node"].setdefault(
                node, {"delta_nfr_total": 0.0, "breakdown": {}}
            )
            node_total = sum(features.values())
            node_entry["delta_nfr_total"] += float(node_total)
            feature_map = node_entry["breakdown"]
            for name, value in features.items():
                feature_map[name] = feature_map.get(name, 0.0) + float(value)

    for node, node_entry in summary["per_node"].items():
        node_entry["delta_nfr_total"] = float(node_entry["delta_nfr_total"])
        node_entry["breakdown"] = {
            name: float(value) for name, value in node_entry["breakdown"].items()
        }

    return summary


def _generate_out_reports(
    records: Records,
    bundles: Bundles,
    microsectors: Sequence[Microsector],
    destination: Path,
    *,
    microsector_variability: Sequence[Mapping[str, Any]] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    sense_map = _sense_index_map(bundles, microsectors)
    resonance = analyse_modal_resonance(records)
    breakdown = _delta_breakdown_summary(bundles)
    metrics = dict(metrics or {})

    def _floatify(value: Any, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _floatify_mapping(mapping: Mapping[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for key, value in mapping.items():
            if isinstance(value, Mapping):
                payload[str(key)] = _floatify_mapping(value)
            else:
                payload[str(key)] = _floatify(value, default=0.0)
        return payload

    sense_path = destination / "sense_index_map.json"
    resonance_path = destination / "modal_resonance.json"
    breakdown_path = destination / "delta_breakdown.json"
    occupancy_path = destination / "window_occupancy.json"
    pairwise_path = destination / "pairwise_coupling.json"
    dissonance_path = destination / "dissonance_breakdown.json"
    memory_path = destination / "sense_memory.json"
    summary_path = destination / "metrics_summary.md"

    occupancy_payload: List[Dict[str, Any]] = []
    for entry in sense_map:
        occupancy_payload.append(
            {
                "microsector": entry.get("microsector"),
                "label": entry.get("label"),
                "window_occupancy": {
                    phase: {
                        metric: _floatify(value, default=0.0)
                        for metric, value in (entry.get("window_occupancy", {}).get(phase, {}) or {}).items()
                    }
                    for phase in ("entry", "apex", "exit")
                },
            }
        )

    with sense_path.open("w", encoding="utf8") as handle:
        json.dump(sense_map, handle, indent=2, sort_keys=True)
    resonance_payload = {
        axis: {
            "sample_rate": analysis.sample_rate,
            "total_energy": analysis.total_energy,
            "peaks": [
                {
                    "frequency": peak.frequency,
                    "energy": peak.energy,
                    "classification": peak.classification,
                }
                for peak in analysis.peaks
            ],
        }
        for axis, analysis in resonance.items()
    }
    with resonance_path.open("w", encoding="utf8") as handle:
        json.dump(resonance_payload, handle, indent=2, sort_keys=True)
    with breakdown_path.open("w", encoding="utf8") as handle:
        json.dump(breakdown, handle, indent=2, sort_keys=True)
    with occupancy_path.open("w", encoding="utf8") as handle:
        json.dump(occupancy_payload, handle, indent=2, sort_keys=True)

    pairwise_payload: Dict[str, Any] = {}
    raw_pairwise = metrics.get("pairwise_coupling")
    if isinstance(raw_pairwise, Mapping):
        pairwise_payload = _floatify_mapping(raw_pairwise)
    coupling_payload = {
        "global": {
            "delta_nfr_vs_sense_index": _floatify(metrics.get("coupling")),
            "resonance_index": _floatify(metrics.get("resonance")),
            "dissonance": _floatify(metrics.get("dissonance")),
        },
        "pairwise": pairwise_payload,
    }
    with pairwise_path.open("w", encoding="utf8") as handle:
        json.dump(coupling_payload, handle, indent=2, sort_keys=True)

    dissonance_breakdown = metrics.get("dissonance_breakdown")
    if dissonance_breakdown is None:
        dissonance_payload: Dict[str, Any] = {}
    else:
        try:
            dissonance_payload = {
                key: _floatify(value)
                for key, value in asdict(dissonance_breakdown).items()
            }
        except TypeError:
            if isinstance(dissonance_breakdown, Mapping):
                dissonance_payload = {
                    key: _floatify(value) for key, value in dissonance_breakdown.items()
                }
            else:
                dissonance_payload = {"value": _floatify(dissonance_breakdown)}
    with dissonance_path.open("w", encoding="utf8") as handle:
        json.dump(dissonance_payload, handle, indent=2, sort_keys=True)

    sense_memory = metrics.get("sense_memory")
    if isinstance(sense_memory, Mapping):
        memory_payload = {
            "series": [_floatify(value) for value in sense_memory.get("series", []) if isinstance(value, (int, float))],
            "memory": [_floatify(value) for value in sense_memory.get("memory", []) if isinstance(value, (int, float))],
            "average": _floatify(sense_memory.get("average")),
            "decay": _floatify(sense_memory.get("decay")),
        }
    else:
        memory_payload = {"series": [], "memory": [], "average": 0.0, "decay": 0.0}
    with memory_path.open("w", encoding="utf8") as handle:
        json.dump(memory_payload, handle, indent=2, sort_keys=True)

    summary_lines: List[str] = [
        "# Resumen de métricas avanzadas",
        "",
        "## Resonancia modal",
    ]
    for axis, analysis in sorted(resonance_payload.items()):
        summary_lines.append(
            f"- **{axis}** · energía total {analysis['total_energy']:.3f} (Fs={analysis['sample_rate']:.1f} Hz)"
        )
        if analysis["peaks"]:
            for peak in analysis["peaks"]:
                summary_lines.append(
                    "  - "
                    f"{peak['classification']} @ {peak['frequency']:.3f} Hz "
                    f"({peak['energy']:.3f})"
                )
        else:
            summary_lines.append("  - Sin picos detectados")

    if dissonance_payload:
        summary_lines.extend(
            [
                "",
                "## Disonancia útil",
                f"- Magnitud útil: {dissonance_payload.get('useful_magnitude', 0.0):.3f}",
                f"- Eventos útiles: {int(dissonance_payload.get('useful_events', 0))}",
                f"- Magnitud parasitaria: {dissonance_payload.get('parasitic_magnitude', 0.0):.3f}",
            ]
        )

    summary_lines.extend(
        [
            "",
            "## Acoplamientos",
            f"- Acoplamiento global ΔNFR↔Si: {coupling_payload['global']['delta_nfr_vs_sense_index']:.3f}",
            f"- Índice de resonancia global: {coupling_payload['global']['resonance_index']:.3f}",
        ]
    )
    if pairwise_payload:
        summary_lines.append("- Pares analizados:")
        for domain, pairs in sorted(pairwise_payload.items()):
            summary_lines.append(f"  - {domain}:")
            for pair, value in sorted(pairs.items()):
                summary_lines.append(f"    - {pair}: {value:.3f}")

    if memory_payload["memory"] or memory_payload["series"]:
        summary_lines.extend(
            [
                "",
                "## Memoria del índice de sensibilidad",
                f"- Promedio suavizado: {memory_payload['average']:.3f}",
                f"- Último valor de memoria: {memory_payload['memory'][-1]:.3f}",
                f"- Factor de decaimiento: {memory_payload['decay']:.3f}",
            ]
        )
    else:
        summary_lines.extend(
            [
                "",
                "## Memoria del índice de sensibilidad",
                "- No se registraron muestras para la traza de memoria.",
            ]
        )

    if occupancy_payload:
        summary_lines.extend(["", "## Ocupación de ventanas"])
        phase_totals: Dict[str, List[float]] = {"entry": [], "apex": [], "exit": []}
        for entry in occupancy_payload:
            for phase, values in entry.get("window_occupancy", {}).items():
                total = sum(_floatify(value) for value in values.values())
                phase_totals.setdefault(phase, []).append(total)
        for phase, values in sorted(phase_totals.items()):
            if not values:
                continue
            summary_lines.append(
                f"- {phase}: promedio {sum(values) / len(values):.3f} (ver window_occupancy.json para detalle)"
            )

    summary_text = "\n".join(summary_lines) + "\n"
    with summary_path.open("w", encoding="utf8") as handle:
        handle.write(summary_text)

    variability_data = [dict(entry) for entry in microsector_variability or ()]
    variability_path = destination / "microsector_variability.json"
    with variability_path.open("w", encoding="utf8") as handle:
        json.dump(variability_data, handle, indent=2, sort_keys=True)
    return {
        "sense_index_map": {"path": str(sense_path), "data": sense_map},
        "modal_resonance": {"path": str(resonance_path), "data": resonance_payload},
        "delta_breakdown": {"path": str(breakdown_path), "data": breakdown},
        "window_occupancy": {"path": str(occupancy_path), "data": occupancy_payload},
        "microsector_variability": {
            "path": str(variability_path),
            "data": variability_data,
        },
        "pairwise_coupling": {"path": str(pairwise_path), "data": coupling_payload},
        "dissonance_breakdown": {"path": str(dissonance_path), "data": dissonance_payload},
        "sense_memory": {"path": str(memory_path), "data": memory_payload},
        "metrics_summary": {"path": str(summary_path), "data": summary_text},
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
    "lap": None,
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
    baseline_parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Destination file for the baseline. When omitted or pointing to a directory "
            "a timestamped run is created under runs/."
        ),
    )
    baseline_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory used when auto-generating baseline runs (default: runs/)."
        ),
    )
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
    write_set_parser.add_argument(
        "--set-output",
        default=None,
        help=(
            "Nombre base del archivo .set que se guardará bajo LFS/data/setups. "
            "Debe comenzar por el prefijo del coche."
        ),
    )
    write_set_parser.set_defaults(handler=_handle_write_set)

    return parser


def _handle_template(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    car_model = str(namespace.car_model or _default_car_model(config))
    track_name = str(namespace.track or _default_track_name(config))
    engine = RecommendationEngine(car_model=car_model, track_name=track_name)
    context = engine._resolve_context(car_model, track_name)
    profile = context.thresholds
    templates = _profile_phase_templates(profile)
    average_delta = (
        mean(entry["target_delta_nfr"] for entry in templates.values())
        if templates
        else 0.0
    )

    lines = [
        f"# Preset generado para {context.profile_label}",
        "",
        "[limits.delta_nfr]",
        f"entry = {profile.entry_delta_tolerance:.3f}",
        f"apex = {profile.apex_delta_tolerance:.3f}",
        f"exit = {profile.exit_delta_tolerance:.3f}",
        f"piano = {profile.piano_delta_tolerance:.3f}",
        "",
    ]

    for section in ("analyze", "report"):
        lines.append(f"[{section}]")
        lines.append(f"target_delta = {average_delta:.3f}")
        lines.append("")
        for phase, payload in templates.items():
            lines.append(f"[{section}.phase_templates.{phase}]")
            lines.append(f"target_delta_nfr = {payload['target_delta_nfr']:.3f}")
            lat = payload["slip_lat_window"]
            lines.append(f"slip_lat_window = [{lat[0]:.3f}, {lat[1]:.3f}]")
            lng = payload["slip_long_window"]
            lines.append(f"slip_long_window = [{lng[0]:.3f}, {lng[1]:.3f}]")
            yaw = payload["yaw_rate_window"]
            lines.append(f"yaw_rate_window = [{yaw[0]:.3f}, {yaw[1]:.3f}]")
            lines.append("")

    result = "\n".join(lines).rstrip() + "\n"
    print(result)
    return result


def _handle_osd(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    layout_defaults = ButtonLayout().clamp()
    layout = ButtonLayout(
        left=layout_defaults.left if namespace.layout_left is None else int(namespace.layout_left),
        top=layout_defaults.top if namespace.layout_top is None else int(namespace.layout_top),
        width=layout_defaults.width if namespace.layout_width is None else int(namespace.layout_width),
        height=layout_defaults.height if namespace.layout_height is None else int(namespace.layout_height),
        ucid=layout_defaults.ucid,
        inst=layout_defaults.inst,
        click_id=layout_defaults.click_id,
        style=layout_defaults.style,
        type_in=layout_defaults.type_in,
    )
    controller = OSDController(
        host=str(namespace.host),
        outsim_port=int(namespace.outsim_port),
        outgauge_port=int(namespace.outgauge_port),
        insim_port=int(namespace.insim_port),
        insim_keepalive=float(namespace.insim_keepalive),
        update_rate=float(namespace.update_rate),
        car_model=str(namespace.car_model or _default_car_model(config)),
        track_name=str(namespace.track or _default_track_name(config)),
        layout=layout,
    )
    return controller.run()


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
    destination = _resolve_baseline_destination(namespace, config)

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

    _persist_records(records, destination, namespace.format)
    message = (
        f"Baseline saved {len(records)} samples to {destination} "
        f"({namespace.format})."
    )
    print(message)
    return message


def _handle_analyze(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    records = _load_records(namespace.telemetry)
    car_model = _default_car_model(config)
    track_name = _default_track_name(config)
    bundles, microsectors, thresholds = _compute_insights(
        records,
        car_model=car_model,
        track_name=track_name,
    )
    lap_segments = _group_records_by_lap(records)
    metrics = orchestrate_delta_metrics(
        lap_segments,
        namespace.target_delta,
        namespace.target_si,
        coherence_window=namespace.coherence_window,
        recursion_decay=namespace.recursion_decay,
        microsectors=microsectors,
        phase_weights=thresholds.phase_weights,
    )
    phase_messages = _phase_deviation_messages(
        bundles,
        microsectors,
        config,
        car_model=car_model,
        track_name=track_name,
    )
    reports = _generate_out_reports(
        records,
        bundles,
        microsectors,
        _resolve_output_dir(config) / namespace.telemetry.stem,
        microsector_variability=metrics.get("microsector_variability"),
        metrics=metrics,
    )
    phase_templates = _phase_templates_from_config(config, "analyze")
    intermediate_metrics = {
        "epi_evolution": metrics.get("epi_evolution"),
        "nodal_metrics": metrics.get("nodal_metrics"),
        "sense_memory": metrics.get("sense_memory"),
    }
    summary_metrics = {
        "delta_nfr": metrics.get("delta_nfr", 0.0),
        "sense_index": metrics.get("sense_index", 0.0),
        "dissonance": metrics.get("dissonance", 0.0),
        "coupling": metrics.get("coupling", 0.0),
        "resonance": metrics.get("resonance", 0.0),
        "dissonance_breakdown": metrics.get("dissonance_breakdown"),
        "pairwise_coupling": metrics.get("pairwise_coupling"),
        "microsector_variability": metrics.get("microsector_variability", []),
    }
    payload: Dict[str, Any] = {
        "series": bundles,
        "microsectors": microsectors,
        "telemetry_samples": len(records),
        "metrics": summary_metrics,
        "smoothed_series": metrics.get("bundles", []),
        "intermediate_metrics": intermediate_metrics,
        "stages": metrics.get("stages"),
        "objectives": metrics.get("objectives", {}),
        "phase_messages": phase_messages,
        "reports": reports,
    }
    if phase_templates:
        payload["phase_templates"] = phase_templates
    return _render_payload(payload, namespace.export)


def _handle_suggest(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    records = _load_records(namespace.telemetry)
    bundles, microsectors, thresholds = _compute_insights(
        records,
        car_model=namespace.car_model,
        track_name=namespace.track,
    )
    lap_segments = _group_records_by_lap(records)
    suggest_cfg = dict(config.get("suggest", {}))
    metrics = orchestrate_delta_metrics(
        lap_segments,
        float(suggest_cfg.get("target_delta", 0.0)),
        float(suggest_cfg.get("target_si", 0.75)),
        coherence_window=int(suggest_cfg.get("coherence_window", 3)),
        recursion_decay=float(suggest_cfg.get("recursion_decay", 0.4)),
        microsectors=microsectors,
        phase_weights=thresholds.phase_weights,
    )
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
        metrics=metrics,
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
    if metrics:
        payload["metrics"] = {
            "delta_nfr": metrics.get("delta_nfr", 0.0),
            "sense_index": metrics.get("sense_index", 0.0),
            "dissonance": metrics.get("dissonance", 0.0),
            "coupling": metrics.get("coupling", 0.0),
            "resonance": metrics.get("resonance", 0.0),
            "pairwise_coupling": metrics.get("pairwise_coupling"),
            "dissonance_breakdown": metrics.get("dissonance_breakdown"),
        }
    return _render_payload(payload, namespace.export)


def _handle_report(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    records = _load_records(namespace.telemetry)
    car_model = _default_car_model(config)
    track_name = _default_track_name(config)
    bundles, microsectors, thresholds = _compute_insights(
        records,
        car_model=car_model,
        track_name=track_name,
    )
    lap_segments = _group_records_by_lap(records)
    metrics = orchestrate_delta_metrics(
        lap_segments,
        namespace.target_delta,
        namespace.target_si,
        coherence_window=namespace.coherence_window,
        recursion_decay=namespace.recursion_decay,
        microsectors=microsectors,
        phase_weights=thresholds.phase_weights,
    )
    reports = _generate_out_reports(
        records,
        bundles,
        microsectors,
        _resolve_output_dir(config) / namespace.telemetry.stem,
        microsector_variability=metrics.get("microsector_variability"),
        metrics=metrics,
    )
    payload: Dict[str, Any] = {
        "objectives": metrics.get("objectives", {}),
        "delta_nfr": metrics.get("delta_nfr", 0.0),
        "sense_index": metrics.get("sense_index", 0.0),
        "dissonance": metrics.get("dissonance", 0.0),
        "coupling": metrics.get("coupling", 0.0),
        "resonance": metrics.get("resonance", 0.0),
        "recursive_trace": metrics.get("recursive_trace", []),
        "pairwise_coupling": metrics.get("pairwise_coupling", {}),
        "dissonance_breakdown": metrics.get("dissonance_breakdown"),
        "microsector_variability": metrics.get("microsector_variability", []),
        "lap_sequence": metrics.get("lap_sequence", []),
        "series": bundles if bundles else metrics.get("bundles", []),
        "reports": reports,
        "intermediate_metrics": {
            "epi_evolution": metrics.get("epi_evolution"),
            "nodal_metrics": metrics.get("nodal_metrics"),
            "sense_memory": metrics.get("sense_memory"),
        },
        "stages": metrics.get("stages"),
    }
    phase_templates = _phase_templates_from_config(config, "report")
    if phase_templates:
        payload["phase_templates"] = phase_templates
    return _render_payload(payload, namespace.export)


def _handle_write_set(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    records = _load_records(namespace.telemetry)
    bundles, microsectors, _ = _compute_insights(
        records,
        car_model=namespace.car_model,
        track_name=_default_track_name(config),
    )
    planner = SetupPlanner()
    plan = planner.plan(bundles, microsectors, car_model=namespace.car_model)

    action_recommendations = [
        rec for rec in plan.recommendations if rec.parameter and rec.delta is not None
    ]

    aggregated_rationales = [rec.rationale for rec in plan.recommendations if rec.rationale]
    aggregated_effects = [rec.message for rec in plan.recommendations if rec.message]

    if action_recommendations:
        ordered_actions = sorted(
            action_recommendations,
            key=lambda rec: (rec.priority, -abs(rec.delta or 0.0)),
        )
        changes = [
            SetupChange(
                parameter=rec.parameter or "",
                delta=float(rec.delta or 0.0),
                rationale=rec.rationale or "",
                expected_effect=rec.message,
            )
            for rec in ordered_actions
        ]
        if not aggregated_rationales:
            aggregated_rationales = [rec.rationale for rec in ordered_actions if rec.rationale]
        if not aggregated_effects:
            aggregated_effects = [rec.message for rec in ordered_actions if rec.message]
    else:
        default_rationales = aggregated_rationales or ["Optimización de objetivo Si/ΔNFR"]
        default_effects = aggregated_effects or ["Mejora equilibrada del coche"]
        changes = [
            SetupChange(
                parameter=name,
                delta=value,
                rationale="; ".join(default_rationales),
                expected_effect="; ".join(default_effects),
            )
            for name, value in sorted(plan.decision_vector.items())
        ]
        aggregated_rationales = default_rationales
        aggregated_effects = default_effects

    setup_plan = SetupPlan(
        car_model=namespace.car_model,
        session=namespace.session,
        changes=tuple(changes),
        rationales=tuple(aggregated_rationales),
        expected_effects=tuple(aggregated_effects),
        sensitivities=plan.sensitivities,
    )

    payload = {
        "setup_plan": setup_plan,
        "objective_value": plan.objective_value,
        "recommendations": plan.recommendations,
        "series": plan.telemetry,
        "sensitivities": plan.sensitivities,
        "set_output": namespace.set_output,
    }
    return _render_payload(payload, namespace.export)


def _render_payload(payload: Mapping[str, Any], exporter_name: str) -> str:
    exporter = exporters_registry[exporter_name]
    rendered = exporter(dict(payload))
    print(rendered)
    return rendered


def _group_records_by_lap(records: Records) -> List[Records]:
    if not records:
        return []
    labels: List[Any] = []
    last_label: Any = None
    for record in records:
        lap_value = getattr(record, "lap", None)
        if lap_value is not None:
            last_label = lap_value
        labels.append(last_label)
    if labels and labels[0] is None:
        first_label = next((label for label in labels if label is not None), None)
        if first_label is not None:
            labels = [label if label is not None else first_label for label in labels]
    unique_labels = {label for label in labels if label is not None}
    if len(unique_labels) <= 1:
        return [records]
    groups: List[Records] = []
    current_group: List[TelemetryRecord] = []
    current_label: Any = labels[0]
    for record, label in zip(records, labels):
        if current_group and label != current_label:
            groups.append(current_group)
            current_group = []
        current_group.append(record)
        current_label = label
    if current_group:
        groups.append(current_group)
    return groups or [records]


def _compute_insights(
    records: Records,
    *,
    car_model: str,
    track_name: str,
) -> tuple[Bundles, Sequence[Microsector], ThresholdProfile]:
    engine = RecommendationEngine(car_model=car_model, track_name=track_name)
    profile = engine._resolve_context(car_model, track_name).thresholds  # type: ignore[attr-defined]
    if not records:
        return [], [], profile
    extractor = EPIExtractor()
    bundles = extractor.extract(records)
    if not bundles:
        return bundles, [], profile
    overrides = profile.phase_weights
    microsectors = segment_microsectors(
        records,
        bundles,
        phase_weight_overrides=overrides if overrides else None,
    )
    return bundles, microsectors, profile


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
        compress = destination.suffix in {".gz", ".gzip"}
        logs.write_run(records, destination, compress=compress)
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
    name = source.name.lower()
    if suffix == ".csv":
        return OutSimClient().ingest(source)
    if name.endswith(".jsonl") or name.endswith(".jsonl.gz") or name.endswith(".jsonl.gzip"):
        return list(logs.iter_run(source))
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
