"""Nodal stage helpers for the ΔEPI/ΔNFR pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Sequence

from tnfr_core.equations.contextual_delta import ContextMatrix, resolve_series_context
from tnfr_core.operators.interfaces import (
    SupportsChassisNode,
    SupportsEPIBundle,
    SupportsSuspensionNode,
    SupportsTyresNode,
)

ContextLoader = Callable[[], ContextMatrix]
ArrayModule = Any
PairwiseCouplingOperator = Callable[
    [Mapping[str, Sequence[float]], Sequence[tuple[str, str]]],
    Dict[str, float],
]


@dataclass
class StructuralDeltaComponent:
    """Compute ΔNFR contributions for structural operators (AL/EN/IL)."""

    component_to_node: Mapping[str, str] = field(
        default_factory=lambda: {"AL": "tyres", "EN": "suspension", "IL": "chassis"}
    )

    def component_series(
        self, bundles: Sequence[SupportsEPIBundle]
    ) -> Dict[str, list[float]]:
        series: Dict[str, list[float]] = {
            component: [] for component in self.component_to_node
        }
        for bundle in bundles:
            for component, node_name in self.component_to_node.items():
                node = getattr(bundle, node_name, None)
                if node is None:
                    series[component].append(0.0)
                    continue
                delta_value = getattr(node, "delta_nfr", 0.0)
                series[component].append(float(delta_value))
        return series


def _stage_nodal_metrics(
    bundles: Sequence[SupportsEPIBundle],
    *,
    load_context_matrix: ContextLoader,
    xp: ArrayModule,
    structural_component: StructuralDeltaComponent | None = None,
    pairwise_coupling: PairwiseCouplingOperator | None = None,
) -> Dict[str, object]:
    if not bundles:
        empty: Dict[str, Sequence[float]] = {"tyres": [], "suspension": [], "chassis": []}
        return {
            "delta_by_node": empty,
            "sense_index_by_node": empty,
            "pairwise_coupling": {"delta_nfr": {}, "sense_index": {}},
            "structural_delta": {},
            "dEPI_dt_by_operator": {},
        }

    structural_component = structural_component or StructuralDeltaComponent()
    if pairwise_coupling is None:
        raise ValueError("pairwise_coupling operator must be provided")
    node_pairs = (
        ("tyres", "suspension"),
        ("tyres", "chassis"),
        ("suspension", "chassis"),
    )
    context_matrix = load_context_matrix()
    bundle_context = resolve_series_context(bundles, matrix=context_matrix)
    tyre_nodes: Sequence[SupportsTyresNode] = [bundle.tyres for bundle in bundles]
    suspension_nodes: Sequence[SupportsSuspensionNode] = [
        bundle.suspension for bundle in bundles
    ]
    chassis_nodes: Sequence[SupportsChassisNode] = [
        bundle.chassis for bundle in bundles
    ]
    sample_count = min(
        len(tyre_nodes),
        len(suspension_nodes),
        len(chassis_nodes),
        len(bundle_context),
    )
    multipliers = xp.asarray(
        [float(bundle_context[idx].multiplier) for idx in range(sample_count)],
        dtype=float,
    )
    multipliers = xp.clip(
        multipliers,
        float(context_matrix.min_multiplier),
        float(context_matrix.max_multiplier),
    )
    tyre_delta = xp.asarray(
        [float(tyre_nodes[idx].delta_nfr) for idx in range(sample_count)],
        dtype=float,
    )
    suspension_delta = xp.asarray(
        [float(suspension_nodes[idx].delta_nfr) for idx in range(sample_count)],
        dtype=float,
    )
    chassis_delta = xp.asarray(
        [float(chassis_nodes[idx].delta_nfr) for idx in range(sample_count)],
        dtype=float,
    )
    tyre_si = xp.asarray([float(node.sense_index) for node in tyre_nodes], dtype=float)
    suspension_si = xp.asarray(
        [float(node.sense_index) for node in suspension_nodes], dtype=float
    )
    chassis_si = xp.asarray(
        [float(node.sense_index) for node in chassis_nodes], dtype=float
    )

    delta_by_node = {
        "tyres": xp.multiply(tyre_delta, multipliers).tolist(),
        "suspension": xp.multiply(suspension_delta, multipliers).tolist(),
        "chassis": xp.multiply(chassis_delta, multipliers).tolist(),
    }
    si_by_node = {
        "tyres": tyre_si.tolist(),
        "suspension": suspension_si.tolist(),
        "chassis": chassis_si.tolist(),
    }

    structural_delta = structural_component.component_series(bundles)
    derivative_by_operator: Dict[str, list[float]] = {}
    for component, values in structural_delta.items():
        component_array = xp.asarray(values[:sample_count], dtype=float)
        derivative_by_operator[component] = xp.multiply(component_array, multipliers).tolist()

    pairwise_delta = pairwise_coupling(delta_by_node, node_pairs)
    pairwise_si = pairwise_coupling(si_by_node, node_pairs)
    return {
        "delta_by_node": delta_by_node,
        "sense_index_by_node": si_by_node,
        "pairwise_coupling": {
            "delta_nfr": pairwise_delta,
            "sense_index": pairwise_si,
        },
        "structural_delta": structural_delta,
        "dEPI_dt_by_operator": derivative_by_operator,
    }
