"""Shared helpers for structural EPI computations.

This module hosts utilities that are required by both the operator-facing
APIs (``tnfr_core.operators.structural``) and the equation integrators in
``tnfr_core.equations``.  Keeping them in this neutral location avoids
introducing circular imports when the higher level modules depend on each
other.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from tnfr_core.equations.phase_weights import _phase_weight

__all__ = [
    "PhaseContext",
    "extract_phase_context",
    "resolve_nu_targets",
    "compute_nodal_contributions",
    "NON_NODAL_METADATA_KEYS",
]


NON_NODAL_METADATA_KEYS = frozenset(
    {
        "__theta__",
        "__w_phase__",
        "nu_f_objectives",
        "__nu_f__",
        "nu_f_targets",
    }
)


@dataclass(frozen=True)
class PhaseContext:
    """Container describing the active EPI phase metadata."""

    identifier: Any | None
    weights: Mapping[str, Mapping[str, float] | float] | None


def _extract_metadata_value(delta_map: Any, keys: Sequence[str]) -> Any:
    """Return the first non-``None`` metadata value matched by ``keys``."""

    for key in keys:
        if hasattr(delta_map, key):
            value = getattr(delta_map, key)
            if value is not None:
                return value
    if isinstance(delta_map, MappingABC):
        for key in keys:
            if key in delta_map:
                value = delta_map[key]
                if value is not None:
                    return value
    return None


def extract_phase_context(delta_map: Any) -> PhaseContext:
    """Extract phase identifier and weights from ``delta_map`` metadata."""

    phase_identifier = _extract_metadata_value(delta_map, ("__theta__", "theta"))
    raw_phase_weights = _extract_metadata_value(
        delta_map, ("__w_phase__", "phase_weights")
    )
    phase_weights: Mapping[str, Mapping[str, float] | float] | None
    if isinstance(raw_phase_weights, MappingABC):
        phase_weights = raw_phase_weights  # type: ignore[assignment]
    else:
        phase_weights = None
    return PhaseContext(identifier=phase_identifier, weights=phase_weights)


def resolve_nu_targets(delta_map: Any) -> Dict[str, float] | None:
    """Resolve explicit ν_f objectives from ``delta_map`` metadata."""

    raw_nu_targets = _extract_metadata_value(
        delta_map, ("nu_f_objectives", "__nu_f__", "nu_f_targets")
    )
    if not isinstance(raw_nu_targets, MappingABC):
        return None

    targets: Dict[str, float] = {}
    for key, value in raw_nu_targets.items():
        try:
            targets[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return targets


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
