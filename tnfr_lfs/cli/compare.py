"""Command helpers for the ``compare`` sub-command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping

from ..analysis import SUPPORTED_LAP_METRICS, ab_compare_by_lap
from ..core.operators import orchestrate_delta_metrics
from ..session import format_session_messages

SUPPORTED_AB_METRICS: tuple[str, ...] = tuple(sorted(SUPPORTED_LAP_METRICS))


def register_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    config: Mapping[str, Any],
) -> None:
    """Register the ``compare`` sub-command."""

    compare_cfg = dict(config.get("compare", {}))
    default_metric = str(compare_cfg.get("metric", "sense_index"))
    if default_metric not in SUPPORTED_AB_METRICS:
        default_metric = "sense_index"

    parser = subparsers.add_parser(
        "compare",
        help="Compara dos stints de telemetría agregando métricas por vuelta.",
    )
    parser.add_argument(
        "telemetry_a",
        type=Path,
        help="Ruta a la telemetría baseline o configuración A.",
    )
    parser.add_argument(
        "telemetry_b",
        type=Path,
        help="Ruta a la telemetría variante o configuración B.",
    )
    parser.add_argument(
        "--metric",
        choices=SUPPORTED_AB_METRICS,
        default=default_metric,
        help="Métrica por vuelta utilizada para la comparación (default: sense_index).",
    )

    from . import tnfr_lfs_cli as cli

    cli._add_export_argument(
        parser,
        default=cli._validated_export(compare_cfg.get("export"), fallback="markdown"),
        help_text="Exporter usado para renderizar la comparación A/B (default: markdown).",
    )
    parser.set_defaults(handler=handle)


def handle(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    """Execute the ``compare`` command returning the rendered payload."""

    from . import tnfr_lfs_cli as cli

    metric = str(namespace.metric)
    pack_root = cli._resolve_pack_root(namespace, config)
    track_selection = cli._resolve_track_argument(None, config, pack_root=pack_root)
    car_model = cli._default_car_model(config)
    track_name = track_selection.name or cli._default_track_name(config)
    report_cfg = dict(config.get("report", {}))
    target_delta = float(report_cfg.get("target_delta", 0.0))
    target_si = float(report_cfg.get("target_si", 0.75))
    coherence_window = int(report_cfg.get("coherence_window", 3))
    recursion_decay = float(report_cfg.get("recursion_decay", 0.4))

    def _compute_metrics(path: Path) -> Mapping[str, Any]:
        records = cli._load_records(path)
        lap_segments = cli._group_records_by_lap(records)
        return orchestrate_delta_metrics(
            lap_segments,
            target_delta,
            target_si,
            coherence_window=coherence_window,
            recursion_decay=recursion_decay,
        )

    try:
        baseline_metrics = _compute_metrics(namespace.telemetry_a)
        variant_metrics = _compute_metrics(namespace.telemetry_b)
        abtest_result = ab_compare_by_lap(
            baseline_metrics,
            variant_metrics,
            metric=metric,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    cars = cli._load_pack_cars(pack_root)
    track_profiles = cli._load_pack_track_profiles(pack_root)
    modifiers = cli._load_pack_modifiers(pack_root)
    session_payload = cli._assemble_session_payload(
        car_model,
        track_selection,
        cars=cars,
        track_profiles=track_profiles,
        modifiers=modifiers,
    )
    if isinstance(session_payload, Mapping):
        session_mapping: Dict[str, Any] = dict(session_payload)
    else:
        session_mapping = {
            "car_model": car_model,
            "track_profile": track_selection.track_profile or track_name,
        }
    session_mapping["abtest"] = abtest_result

    payload: Dict[str, Any] = {
        "metric": metric,
        "baseline": {
            "telemetry": str(namespace.telemetry_a),
            "mean": abtest_result.baseline_mean,
            "lap_means": list(abtest_result.baseline_laps),
            "lap_count": len(abtest_result.baseline_laps),
        },
        "variant": {
            "telemetry": str(namespace.telemetry_b),
            "mean": abtest_result.variant_mean,
            "lap_means": list(abtest_result.variant_laps),
            "lap_count": len(abtest_result.variant_laps),
        },
        "session": session_mapping,
    }
    session_messages = format_session_messages(session_mapping)
    if session_messages:
        payload["session_messages"] = session_messages
    return cli._render_payload(payload, cli._resolve_exports(namespace))


__all__ = ["register_subparser", "handle", "SUPPORTED_AB_METRICS"]
