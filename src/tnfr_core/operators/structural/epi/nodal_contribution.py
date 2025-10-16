"""Computation helpers for structural EPI nodal contributions."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Any, Dict, Mapping

from tnfr_core.equations.epi import _phase_weight

from .metadata import NON_NODAL_METADATA_KEYS, PhaseContext

__all__ = ["compute_nodal_contributions"]


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def compute_nodal_contributions(
    delta_map: Any,
    nu_f_by_node: Mapping[str, float],
    nu_targets: Mapping[str, float] | None,
    phase_context: PhaseContext,
    dt: float,
) -> tuple[Dict[str, tuple[float, float]], Dict[str, float], float]:
    """Return nodal EPI contributions and the aggregated derivative."""

    nodes = set(nu_f_by_node)
    if nu_targets:
        nodes.update(nu_targets)

    if isinstance(delta_map, MappingABC):
        nodes.update(
            str(node)
            for node in delta_map
            if isinstance(node, str)
            and not node.startswith("__")
            and node not in NON_NODAL_METADATA_KEYS
        )

    contributions: Dict[str, tuple[float, float]] = {}
    theta_effects: Dict[str, float] = {}
    derivative_total = 0.0

    for node in nodes:
        base_weight = nu_f_by_node.get(node, 0.0)
        weight = _coerce_float(base_weight)

        if nu_targets and node in nu_targets:
            weight = _coerce_float(nu_targets[node])

        phase_factor = 1.0
        if phase_context.identifier is not None or phase_context.weights is not None:
            try:
                phase_factor = float(
                    _phase_weight(node, phase_context.identifier, phase_context.weights)
                )
            except Exception:  # pragma: no cover - defensive guard
                phase_factor = 1.0
            phase_factor = max(0.5, min(3.0, phase_factor))
            weight *= phase_factor
            theta_effects[node] = phase_factor

        node_delta = 0.0
        if isinstance(delta_map, MappingABC):
            node_delta = _coerce_float(delta_map.get(node, 0.0))

        node_derivative = weight * node_delta
        node_integral = node_derivative * dt
        derivative_total += node_derivative
        contributions[node] = (node_integral, node_derivative)

    return contributions, theta_effects, derivative_total
