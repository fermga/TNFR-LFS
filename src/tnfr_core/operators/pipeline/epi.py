"""EPI evolution stage helpers for the ΔNFR×Si pipeline."""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Sequence

from tnfr_core.equations.baseline import delta_nfr_by_node
from tnfr_core.equations.epi import (
    NaturalFrequencyAnalyzer,
    TelemetryRecord,
    resolve_nu_f_by_node,
)
from tnfr_core.equations.epi_evolution import evolve_epi
from tnfr_core.runtime.shared import SupportsEPIBundle, SupportsTelemetrySample

EnsureBundle = Callable[[SupportsEPIBundle], SupportsEPIBundle]
NormaliseNodeEvolution = Callable[[Mapping[str, tuple[float, float]] | None], Dict[str, tuple[float, float]]]


def _stage_epi_evolution(
    records: Sequence[SupportsTelemetrySample],
    *,
    bundles: Sequence[SupportsEPIBundle] | None,
    phase_assignments: Mapping[int, str] | None,
    phase_weight_lookup: Mapping[int, Mapping[str, Mapping[str, float] | float]] | None,
    global_phase_weights: Mapping[str, Mapping[str, float] | float] | None,
    ensure_bundle: EnsureBundle,
    normalise_node_evolution: NormaliseNodeEvolution,
) -> Dict[str, object]:
    if not records:
        return {
            "integrated": [],
            "derivative": [],
            "per_node_integrated": {},
            "per_node_derivative": {},
        }

    reuse_bundle_evolution = all(
        hasattr(bundle, "integrated_epi") and hasattr(bundle, "node_evolution")
        for bundle in bundles or ()
    )
    has_phase_customisation = bool(phase_assignments) or bool(phase_weight_lookup) or bool(
        global_phase_weights
    )
    if has_phase_customisation:
        reuse_bundle_evolution = False

    integrated_series = []
    derivative_series = []
    per_node_integrated: Dict[str, list[float]] = {}
    per_node_derivative: Dict[str, list[float]] = {}
    cumulative_by_node: Dict[str, float] = {}
    assignment_map = phase_assignments or {}
    has_global_weights = bool(global_phase_weights)

    if reuse_bundle_evolution:
        concrete_bundles = [ensure_bundle(bundle) for bundle in bundles or ()]
        for bundle in concrete_bundles:
            integrated_series.append(float(bundle.integrated_epi))
            derivative_series.append(float(bundle.dEPI_dt))
            nodal = normalise_node_evolution(getattr(bundle, "node_evolution", None))
            nodes = set(per_node_integrated) | set(nodal)
            for node in nodes:
                node_integral, node_derivative = nodal.get(node, (0.0, 0.0))
                node_integral = float(node_integral)
                node_derivative = float(node_derivative)
                cumulative = cumulative_by_node.get(node, 0.0) + node_integral
                cumulative_by_node[node] = cumulative
                per_node_integrated.setdefault(node, []).append(cumulative)
                per_node_derivative.setdefault(node, []).append(node_derivative)
        return {
            "integrated": integrated_series,
            "derivative": derivative_series,
            "per_node_integrated": per_node_integrated,
            "per_node_derivative": per_node_derivative,
        }

    prev_epi = 0.0
    prev_timestamp = records[0].timestamp
    analyzer = NaturalFrequencyAnalyzer()

    for index, record in enumerate(records):
        delta_map = delta_nfr_by_node(record)
        phase = assignment_map.get(index) if assignment_map else None
        weights = None
        if phase_weight_lookup and index in phase_weight_lookup:
            weights = phase_weight_lookup[index]
        elif has_global_weights and global_phase_weights:
            weights = global_phase_weights
        nu_snapshot = resolve_nu_f_by_node(
            record,
            phase=phase,
            phase_weights=weights,
            analyzer=analyzer,
        )
        dt = 0.0 if index == 0 else max(0.0, record.timestamp - prev_timestamp)
        new_epi, derivative, nodal = evolve_epi(
            prev_epi, delta_map, dt, nu_snapshot.by_node
        )
        integrated_series.append(new_epi)
        derivative_series.append(derivative)
        nodes = set(per_node_integrated) | set(nodal)
        for node in nodes:
            node_integral, node_derivative = nodal.get(node, (0.0, 0.0))
            cumulative = cumulative_by_node.get(node, 0.0) + node_integral
            cumulative_by_node[node] = cumulative
            per_node_integrated.setdefault(node, []).append(cumulative)
            per_node_derivative.setdefault(node, []).append(node_derivative)
        prev_epi = new_epi
        prev_timestamp = record.timestamp

    return {
        "integrated": integrated_series,
        "derivative": derivative_series,
        "per_node_integrated": per_node_integrated,
        "per_node_derivative": per_node_derivative,
    }
