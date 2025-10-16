"""Pipeline orchestration helpers for the ΔNFR×Si metrics pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Callable, Dict, Mapping, Sequence

from tnfr_core.equations.epi import TelemetryRecord
from tnfr_core.operators.interfaces import (
    SupportsEPIBundle,
    SupportsMicrosector,
    SupportsTelemetrySample,
)


@dataclass(frozen=True)
class PipelineDependencies:
    """Bundle the stage callables and helpers required by the orchestrator."""

    emission_operator: Callable[[float, float], Mapping[str, float]]
    reception_stage: Callable[
        [Sequence[Sequence[TelemetryRecord]]],
        tuple[Dict[str, object], Sequence[TelemetryRecord]],
    ]
    coherence_stage: Callable[[Sequence[SupportsEPIBundle], Mapping[str, float]], Dict[str, object]]
    nodal_stage: Callable[[Sequence[SupportsEPIBundle]], Dict[str, object]]
    epi_stage: Callable[[Sequence[SupportsTelemetrySample]], Dict[str, object]]
    sense_stage: Callable[[Sequence[float]], Dict[str, object]]
    variability_stage: Callable[
        [
            Sequence[SupportsMicrosector] | None,
            Sequence[SupportsEPIBundle],
            Sequence[int],
            Sequence[Mapping[str, object]],
        ],
        Sequence[Mapping[str, object]],
    ]
    aggregate_events: Callable[[Sequence[SupportsMicrosector] | None], Mapping[str, object]]
    window_metrics: Callable[..., object]
    phase_context_resolver: Callable[
        [Sequence[SupportsMicrosector] | None],
        tuple[Dict[int, str], Dict[int, Mapping[str, Mapping[str, float] | float]]],
    ]
    network_memory_extractor: Callable[[Mapping[str, Dict[str, object]] | None], Mapping[str, object]]
    zero_breakdown_factory: Callable[[], object]


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
    coherence_window: int,
    recursion_decay: float,
    microsectors: Sequence[SupportsMicrosector] | None,
    phase_weights: Mapping[str, Mapping[str, float] | float] | None,
    operator_state: Mapping[str, Dict[str, object]] | None,
    dependencies: PipelineDependencies,
) -> Mapping[str, object]:
    """Coordinate the execution of the ΔNFR×Si pipeline stages."""

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
