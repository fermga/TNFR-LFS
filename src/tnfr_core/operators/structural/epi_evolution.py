"""Structural evolution utilities for EPI computations."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Any, Dict, Mapping, Sequence

from tnfr_core.equations.epi import _phase_weight


class NodalEvolution(dict[str, tuple[float, float]]):
    """Dictionary mapping nodes to ``(integral, derivative)`` tuples with metadata."""

    metadata: Dict[str, Any]

    def __init__(self) -> None:  # noqa: D401 - short custom initialiser
        super().__init__()
        self.metadata = {}


def evolve_epi(
    prev_epi: float,
    delta_map: Mapping[str, float],
    dt: float,
    nu_f_by_node: Mapping[str, float],
) -> tuple[float, float, Dict[str, tuple[float, float]]]:
    """Integrate the Event Performance Index using explicit Euler steps.

    The integrator returns the global derivative/integral together with a per-node
    breakdown.  The nodal contribution dictionary maps the node name to a
    ``(integral, derivative)`` tuple representing the instantaneous change produced
    during ``dt``.  When contextual metadata is supplied in ``delta_map`` the
    returned mapping exposes it through the ``metadata`` attribute so that advanced
    consumers can inspect phase effects without affecting the tuple contract
    consumed elsewhere.
    """

    if dt < 0.0:
        raise ValueError("dt must be non-negative")

    nodal_evolution: NodalEvolution = NodalEvolution()
    derivative = 0.0

    def _extract_metadata_value(keys: Sequence[str]) -> Any:
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

    phase_identifier = _extract_metadata_value(("__theta__", "theta"))
    raw_phase_weights = _extract_metadata_value(("__w_phase__", "phase_weights"))
    phase_weights = (
        raw_phase_weights if isinstance(raw_phase_weights, MappingABC) else None
    )

    raw_nu_targets = _extract_metadata_value(
        ("nu_f_objectives", "__nu_f__", "nu_f_targets")
    )
    nu_targets: Dict[str, float] | None = None
    if isinstance(raw_nu_targets, MappingABC):
        nu_targets = {}
        for key, value in raw_nu_targets.items():
            try:
                nu_targets[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

    nodes = set(nu_f_by_node)
    if nu_targets:
        nodes.update(nu_targets)
    if isinstance(delta_map, MappingABC):
        metadata_keys = {
            "__theta__",
            "__w_phase__",
            "nu_f_objectives",
            "__nu_f__",
            "nu_f_targets",
        }
        nodes.update(
            str(node)
            for node in delta_map
            if isinstance(node, str)
            and not node.startswith("__")
            and node not in metadata_keys
        )

    theta_effects: Dict[str, float] = {}

    for node in nodes:
        base_weight = nu_f_by_node.get(node, 0.0)
        try:
            weight = float(base_weight)
        except (TypeError, ValueError):
            weight = 0.0

        if nu_targets and node in nu_targets:
            weight = nu_targets[node]

        phase_factor = 1.0
        if phase_identifier is not None or phase_weights is not None:
            try:
                phase_factor = float(
                    _phase_weight(node, phase_identifier, phase_weights)
                )
            except Exception:  # pragma: no cover - defensive guard
                phase_factor = 1.0
            phase_factor = max(0.5, min(3.0, phase_factor))
            weight *= phase_factor
            theta_effects[node] = phase_factor

        node_delta = 0.0
        if isinstance(delta_map, MappingABC):
            try:
                node_delta = float(delta_map.get(node, 0.0))
            except (TypeError, ValueError):
                node_delta = 0.0
        node_derivative = weight * node_delta
        node_integral = node_derivative * dt
        derivative += node_derivative
        nodal_evolution[node] = (node_integral, node_derivative)

    if theta_effects:
        nodal_evolution.metadata["theta_effect"] = theta_effects
    if phase_identifier is not None:
        nodal_evolution.metadata["theta"] = phase_identifier
    if phase_weights is not None:
        nodal_evolution.metadata["w_phase"] = dict(phase_weights)
    if nu_targets is not None:
        nodal_evolution.metadata["nu_f_objectives"] = dict(nu_targets)

    new_epi = prev_epi + (derivative * dt)
    return new_epi, derivative, nodal_evolution


__all__ = ["NodalEvolution", "evolve_epi"]
