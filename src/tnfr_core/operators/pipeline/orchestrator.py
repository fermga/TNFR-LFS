"""Pipeline orchestration helpers for the ΔNFR×Si metrics pipeline."""

from __future__ import annotations

from statistics import mean
from typing import Callable, Dict, Mapping, Sequence

import numpy as np

try:  # pragma: no cover - exercised when JAX is installed
    import jax.numpy as jnp  # type: ignore[import-not-found]

    _HAS_JAX = True
except ModuleNotFoundError:  # pragma: no cover - exercised when JAX is unavailable
    jnp = None
    _HAS_JAX = False

from tnfr_core.equations.telemetry import TelemetryRecord
from tnfr_core.operators.entry.recursivity import extract_network_memory
from tnfr_core.runtime.shared import SupportsEPIBundle, SupportsMicrosector
import tnfr_core.operators.en_operator as en_operator_module
from tnfr_core.operators.pipeline.coherence import (
    _stage_coherence as pipeline_stage_coherence,
)
from tnfr_core.operators.pipeline.epi import _stage_epi_evolution as pipeline_stage_epi
from tnfr_core.operators.pipeline.events import (
    _aggregate_operator_events as pipeline_aggregate_operator_events,
)
from tnfr_core.operators.pipeline.nodal import (
    StructuralDeltaComponent,
    _stage_nodal_metrics as pipeline_stage_nodal,
)
from tnfr_core.operators.pipeline.reception import (
    _stage_reception as pipeline_stage_reception,
)
from tnfr_core.operators.pipeline.sense import _stage_sense as pipeline_stage_sense
from tnfr_core.operators.pipeline.variability import (
    _microsector_variability as pipeline_microsector_variability,
    _phase_context_from_microsectors,
    compute_window_metrics as pipeline_compute_window_metrics,
)
from tnfr_core.operators.al_operator import (
    _zero_dissonance_breakdown,
    coherence_operator,
    coupling_operator,
    dissonance_breakdown_operator,
    pairwise_coupling_operator,
    resonance_operator,
)
from tnfr_core.operators.en_operator import (
    _ensure_bundle,
    _normalise_node_evolution,
    _update_bundles,
    emission_operator,
)
from tnfr_core.operators.il_operator import (
    _delta_integral_series,
    _variance_payload,
    recursive_filter_operator,
)

from tnfr_core.operators.pipeline.dependencies import PipelineDependencies
from tnfr_core.operators.pipeline.delta_workflow import (
    build_delta_metrics_dependencies,
)



def _empty_pipeline_payload(
    *,
    objectives: Mapping[str, float],
    reception_stage: Mapping[str, object],
    recursion_decay: float,
    aggregate_events: Callable[[Sequence[SupportsMicrosector] | None], Mapping[str, object]],
    microsectors: Sequence[SupportsMicrosector] | None,
    zero_breakdown_factory: Callable[[], object],
    network_memory: Mapping[str, object],
) -> Dict[str, object]:
    empty_breakdown = zero_breakdown_factory()
    stages = {
        "reception": reception_stage,
        "coherence": {
            "raw_delta": [],
            "raw_sense_index": [],
            "smoothed_delta": [],
            "smoothed_sense_index": [],
            "bundles": [],
            "dissonance": 0.0,
            "dissonance_breakdown": empty_breakdown,
            "coupling": 0.0,
            "resonance": 0.0,
        },
        "nodal": {
            "delta_by_node": {},
            "sense_index_by_node": {},
            "pairwise_coupling": {"delta_nfr": {}, "sense_index": {}},
        },
        "epi": {
            "integrated": [],
            "derivative": [],
            "per_node_integrated": {},
            "per_node_derivative": {},
        },
        "sense": {
            "series": [],
            "memory": [],
            "average": 0.0,
            "decay": recursion_decay,
            "network": network_memory,
        },
    }
    return {
        "objectives": objectives,
        "bundles": [],
        "delta_nfr_series": [],
        "sense_index_series": [],
        "delta_nfr": 0.0,
        "sense_index": 0.0,
        "dissonance": 0.0,
        "dissonance_breakdown": empty_breakdown,
        "coupling": 0.0,
        "resonance": 0.0,
        "support_effective": 0.0,
        "load_support_ratio": 0.0,
        "structural_expansion_longitudinal": 0.0,
        "structural_contraction_longitudinal": 0.0,
        "structural_expansion_lateral": 0.0,
        "structural_contraction_lateral": 0.0,
        "recursive_trace": [],
        "lap_sequence": reception_stage.get("lap_sequence", []),
        "microsector_variability": [],
        "pairwise_coupling": {"delta_nfr": {}, "sense_index": {}},
        "nodal_metrics": stages["nodal"],
        "epi_evolution": stages["epi"],
        "sense_memory": stages["sense"],
        "operator_events": aggregate_events(microsectors),
        "stages": stages,
        "network_memory": network_memory,
    }


def orchestrate_delta_metrics(
    telemetry_segments: Sequence[Sequence[TelemetryRecord]],
    target_delta_nfr: float,
    target_sense_index: float,
    *,
    coherence_window: int = 3,
    recursion_decay: float = 0.4,
    microsectors: Sequence[SupportsMicrosector] | None = None,
    phase_weights: Mapping[str, Mapping[str, float] | float] | None = None,
    operator_state: Mapping[str, Dict[str, object]] | None = None,
    dependencies: PipelineDependencies | None = None,
) -> Mapping[str, object]:
    """Coordinate the execution of the ΔNFR×Si pipeline stages."""

    if dependencies is None:
        from tnfr_core.equations.contextual_delta import (
            apply_contextual_delta,
            load_context_matrix,
            resolve_context_from_bundle,
        )
        xp_module = jnp if _HAS_JAX else np
        structural_component = StructuralDeltaComponent()

        dependencies = build_delta_metrics_dependencies(
            coherence_window=coherence_window,
            recursion_decay=recursion_decay,
            microsectors=microsectors,
            xp_module=xp_module,
            has_jax=_HAS_JAX,
            structural_component=structural_component,
            emission_operator=emission_operator,
            reception_operator=en_operator_module.reception_operator,
            dissonance_breakdown_operator=dissonance_breakdown_operator,
            coherence_operator=coherence_operator,
            coupling_operator=coupling_operator,
            resonance_operator=resonance_operator,
            pairwise_coupling_operator=pairwise_coupling_operator,
            recursive_filter_operator=recursive_filter_operator,
            stage_reception=pipeline_stage_reception,
            stage_coherence=pipeline_stage_coherence,
            stage_nodal=pipeline_stage_nodal,
            stage_epi=pipeline_stage_epi,
            stage_sense=pipeline_stage_sense,
            stage_variability=pipeline_microsector_variability,
            aggregate_events=pipeline_aggregate_operator_events,
            compute_window_metrics=pipeline_compute_window_metrics,
            phase_context_resolver=_phase_context_from_microsectors,
            network_memory_extractor=extract_network_memory,
            zero_breakdown_factory=_zero_dissonance_breakdown,
            load_context_matrix=load_context_matrix,
            resolve_context_from_bundle=resolve_context_from_bundle,
            apply_contextual_delta=apply_contextual_delta,
            update_bundles=_update_bundles,
            ensure_bundle=_ensure_bundle,
            normalise_node_evolution=_normalise_node_evolution,
            delta_integral=_delta_integral_series,
            variance_payload=_variance_payload,
        )

    return _run_pipeline(
        telemetry_segments,
        target_delta_nfr,
        target_sense_index,
        recursion_decay=recursion_decay,
        microsectors=microsectors,
        phase_weights=phase_weights,
        operator_state=operator_state,
        dependencies=dependencies,
    )


def _run_pipeline(
    telemetry_segments: Sequence[Sequence[TelemetryRecord]],
    target_delta_nfr: float,
    target_sense_index: float,
    *,
    recursion_decay: float,
    microsectors: Sequence[SupportsMicrosector] | None,
    phase_weights: Mapping[str, Mapping[str, float] | float] | None,
    operator_state: Mapping[str, Dict[str, object]] | None,
    dependencies: PipelineDependencies,
) -> Mapping[str, object]:
    objectives = dependencies.emission_operator(target_delta_nfr, target_sense_index)
    reception_stage, flattened_records = dependencies.reception_stage(telemetry_segments)
    phase_assignments, weight_lookup = dependencies.phase_context_resolver(microsectors)
    network_memory = dependencies.network_memory_extractor(operator_state)

    bundles: Sequence[SupportsEPIBundle] = reception_stage.get("bundles", [])  # type: ignore[assignment]
    if not bundles:
        return _empty_pipeline_payload(
            objectives=objectives,
            reception_stage=reception_stage,
            recursion_decay=recursion_decay,
            aggregate_events=dependencies.aggregate_events,
            microsectors=microsectors,
            zero_breakdown_factory=dependencies.zero_breakdown_factory,
            network_memory=network_memory,
        )

    coherence_stage = dependencies.coherence_stage(
        bundles,
        objectives,
    )
    nodal_stage = dependencies.nodal_stage(coherence_stage["bundles"])
    epi_stage = dependencies.epi_stage(
        flattened_records,
        bundles=coherence_stage["bundles"],
        phase_assignments=phase_assignments,
        phase_weight_lookup=weight_lookup,
        global_phase_weights=phase_weights,
    )
    sense_stage = dependencies.sense_stage(
        coherence_stage["smoothed_sense_index"],
    )
    sense_stage["network"] = network_memory
    variability = dependencies.variability_stage(
        microsectors,
        coherence_stage["bundles"],
        reception_stage.get("lap_indices", []),
        reception_stage.get("lap_sequence", []),
    )

    stages = {
        "reception": reception_stage,
        "coherence": coherence_stage,
        "nodal": nodal_stage,
        "epi": epi_stage,
        "sense": sense_stage,
    }

    window_metrics = dependencies.window_metrics(
        flattened_records,
        bundles=coherence_stage["bundles"],
        fallback_to_chronological=True,
        objectives=objectives,
    )

    return {
        "objectives": objectives,
        "bundles": coherence_stage["bundles"],
        "delta_nfr_series": coherence_stage["smoothed_delta"],
        "sense_index_series": coherence_stage["smoothed_sense_index"],
        "delta_nfr": mean(coherence_stage["smoothed_delta"])
        if coherence_stage["smoothed_delta"]
        else 0.0,
        "sense_index": mean(coherence_stage["smoothed_sense_index"])
        if coherence_stage["smoothed_sense_index"]
        else 0.0,
        "dissonance": coherence_stage["dissonance"],
        "dissonance_breakdown": coherence_stage["dissonance_breakdown"],
        "coupling": coherence_stage["coupling"],
        "resonance": coherence_stage["resonance"],
        "coherence_index": coherence_stage["coherence_index"],
        "coherence_index_series": coherence_stage["coherence_index_series"],
        "raw_coherence_index": coherence_stage["raw_coherence_index"],
        "frequency_label": coherence_stage["frequency_label"],
        "frequency_classification": coherence_stage["frequency_classification"],
        "support_effective": window_metrics.support_effective,
        "load_support_ratio": window_metrics.load_support_ratio,
        "structural_expansion_longitudinal": window_metrics.structural_expansion_longitudinal,
        "structural_contraction_longitudinal": window_metrics.structural_contraction_longitudinal,
        "structural_expansion_lateral": window_metrics.structural_expansion_lateral,
        "structural_contraction_lateral": window_metrics.structural_contraction_lateral,
        "recursive_trace": sense_stage["memory"],
        "lap_sequence": reception_stage["lap_sequence"],
        "microsector_variability": variability,
        "pairwise_coupling": nodal_stage["pairwise_coupling"],
        "nodal_metrics": nodal_stage,
        "epi_evolution": epi_stage,
        "sense_memory": sense_stage,
        "operator_events": dependencies.aggregate_events(microsectors),
        "stages": stages,
        "network_memory": network_memory,
    }
