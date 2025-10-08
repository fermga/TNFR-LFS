"""Command helpers for the ``pareto`` sub-command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional

from ..recommender import pareto_front, sweep_candidates


def register_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    config: Mapping[str, Any],
) -> None:
    """Register the ``pareto`` sub-command."""

    from . import tnfr_lfs_cli as cli

    write_set_cfg = dict(config.get("write_set", {}))

    parser = subparsers.add_parser(
        "pareto",
        help="Evalúa candidatos de setup y exporta el frente de Pareto resultante.",
    )
    parser.add_argument(
        "telemetry",
        type=Path,
        help="Ruta al archivo (.raf, .csv, .jsonl, .json, .parquet) con la telemetría base.",
    )
    cli._add_export_argument(
        parser,
        default=cli._validated_export(write_set_cfg.get("export"), fallback="markdown"),
        help_text="Exporter usado para representar el frente Pareto (default: markdown).",
    )
    parser.add_argument(
        "--car-model",
        default=str(write_set_cfg.get("car_model", cli._default_car_model(config))),
        help="Modelo de coche utilizado para resolver el espacio de decisiones.",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Etiqueta opcional de sesión que acompañará a los resultados.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=int(write_set_cfg.get("pareto_radius", 1)),
        help=(
            "Número de pasos +/- por parámetro en el barrido de candidatos (default: 1)."
        ),
    )
    parser.set_defaults(handler=handle)


def handle(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    """Execute the ``pareto`` command returning the rendered payload."""

    from . import tnfr_lfs_cli as cli

    namespace.car_model = str(namespace.car_model or cli._default_car_model(config)).strip()
    if not namespace.car_model:
        raise SystemExit(
            "Debe proporcionar un --car-model válido para evaluar el frente Pareto."
        )
    context = cli._compute_setup_plan(namespace, config=config)
    payload = cli._build_setup_plan_payload(context, namespace)

    planner = context.planner
    space = planner._adapt_space(
        planner._space_for_car(namespace.car_model),
        namespace.car_model,
        context.track_name,
    )
    centre_vector: Mapping[str, float] = (
        context.plan.decision_vector or space.initial_guess()
    )

    session_weights: Optional[Mapping[str, Mapping[str, float]]] = None
    session_hints: Optional[Mapping[str, object]] = None
    if isinstance(context.session_payload, Mapping):
        weights_candidate = context.session_payload.get("weights")
        if isinstance(weights_candidate, Mapping):
            session_weights = weights_candidate  # type: ignore[assignment]
        hints_candidate = context.session_payload.get("hints")
        if isinstance(hints_candidate, Mapping):
            session_hints = hints_candidate  # type: ignore[assignment]

    radius = max(0, int(getattr(namespace, "radius", 1)))
    candidates = sweep_candidates(
        space,
        centre_vector,
        context.bundles,
        microsectors=context.microsectors,
        simulator=None,
        session_weights=session_weights,
        session_hints=session_hints,
        radius=radius,
        include_centre=True,
    )
    front = pareto_front(candidates)
    pareto_payload = [dict(point.as_dict()) for point in front]

    session_section: Optional[Mapping[str, Any]] = payload.get("session")  # type: ignore[assignment]
    if isinstance(session_section, Mapping):
        updated_session = dict(session_section)
    elif isinstance(context.session_payload, Mapping):
        updated_session = dict(context.session_payload)
    else:
        updated_session = {}
    updated_session["pareto"] = pareto_payload
    payload["session"] = updated_session
    payload["pareto_points"] = pareto_payload
    payload["pareto_radius"] = radius
    return cli._render_payload(payload, cli._resolve_exports(namespace))


__all__ = ["register_subparser", "handle"]
