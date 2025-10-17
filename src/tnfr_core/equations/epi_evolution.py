"""EPI evolution integrator utilities."""

from __future__ import annotations

from typing import Any, Dict, Mapping, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from tnfr_core.operators.structural.epi import (
        compute_nodal_contributions,
        extract_phase_context,
        resolve_nu_targets,
    )


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
    """Integrate the Event Performance Index using explicit Euler steps."""

    if dt < 0.0:
        raise ValueError("dt must be non-negative")

    nodal_evolution: NodalEvolution = NodalEvolution()

    from tnfr_core.operators.structural import epi as structural_epi

    phase_context = structural_epi.extract_phase_context(delta_map)
    nu_targets = structural_epi.resolve_nu_targets(delta_map)
    contributions, theta_effects, derivative = structural_epi.compute_nodal_contributions(
        delta_map,
        nu_f_by_node,
        nu_targets,
        phase_context,
        dt,
    )

    for node, values in contributions.items():
        nodal_evolution[node] = values

    if theta_effects:
        nodal_evolution.metadata["theta_effect"] = theta_effects
    if phase_context.identifier is not None:
        nodal_evolution.metadata["theta"] = phase_context.identifier
    if phase_context.weights is not None:
        nodal_evolution.metadata["w_phase"] = dict(phase_context.weights)
    if nu_targets is not None:
        nodal_evolution.metadata["nu_f_objectives"] = dict(nu_targets)

    new_epi = prev_epi + (derivative * dt)
    return new_epi, derivative, nodal_evolution


__all__ = ["NodalEvolution", "evolve_epi"]
