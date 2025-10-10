"""Command helpers for the ``compare`` sub-command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping

from ..analysis import SUPPORTED_LAP_METRICS, ab_compare_by_lap
from ..core.operators import orchestrate_delta_metrics
from .session import format_session_messages

from .common import (
    CliError,
    add_export_argument,
    default_car_model,
    default_track_name,
    group_records_by_lap,
    load_pack_cars,
    load_pack_modifiers,
    load_pack_track_profiles,
    load_records,
    render_payload,
    resolve_exports,
    resolve_pack_root,
    resolve_track_argument,
    validated_export,
)
from .workflows import assemble_session_payload

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
        help="Compare two telemetry runs aggregating lap metrics.",
    )
    parser.add_argument(
        "telemetry_a",
        type=Path,
        help="Path to the baseline telemetry or configuration A.",
    )
    parser.add_argument(
        "telemetry_b",
        type=Path,
        help="Path to the variant telemetry or configuration B.",
    )
    parser.add_argument(
        "--metric",
        choices=SUPPORTED_AB_METRICS,
        default=default_metric,
        help="Lap metric used for the comparison (default: sense_index).",
    )

    add_export_argument(
        parser,
        default=validated_export(compare_cfg.get("export"), fallback="markdown"),
        help_text="Exporter used to render the A/B comparison (default: markdown).",
    )
    parser.set_defaults(handler=handle)


def handle(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    """Execute the ``compare`` command returning the rendered payload."""

    metric = str(namespace.metric)
    pack_root = resolve_pack_root(namespace, config)
    track_selection = resolve_track_argument(None, config, pack_root=pack_root)
    car_model = default_car_model(config)
    track_name = track_selection.name or default_track_name(config)
    compare_cfg = dict(config.get("compare", {}))
    target_delta = float(compare_cfg.get("target_delta", 0.0))
    target_si = float(compare_cfg.get("target_si", 0.75))
    coherence_window = int(compare_cfg.get("coherence_window", 3))
    recursion_decay = float(compare_cfg.get("recursion_decay", 0.4))

    def _compute_metrics(path: Path) -> Mapping[str, Any]:
        records = load_records(path)
        lap_segments = group_records_by_lap(records)
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
        raise CliError(str(exc)) from exc

    cars = load_pack_cars(pack_root)
    track_profiles = load_pack_track_profiles(pack_root)
    modifiers = load_pack_modifiers(pack_root)
    session_payload = assemble_session_payload(
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
    return render_payload(payload, resolve_exports(namespace))


__all__ = ["register_subparser", "handle", "SUPPORTED_AB_METRICS"]
