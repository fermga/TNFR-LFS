"""Helpers for wiring the ΔNFR×Si pipeline dependencies."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

from tnfr_core.equations.epi import TelemetryRecord
from tnfr_core.operators.interfaces import (
    SupportsEPIBundle,
    SupportsMicrosector,
    SupportsTelemetrySample,
)
from tnfr_core.operators.pipeline.dependencies import PipelineDependencies


def build_delta_metrics_dependencies(
    *,
    coherence_window: int,
    recursion_decay: float,
    microsectors: Sequence[SupportsMicrosector] | None,
    xp_module: Any,
    has_jax: bool,
    structural_component: Any,
    emission_operator: Callable[[float, float], Mapping[str, float]],
    reception_operator: Callable[[Sequence[TelemetryRecord]], Mapping[str, object]],
    dissonance_breakdown_operator: Callable[..., object],
    coherence_operator: Callable[[Sequence[float], int], Sequence[float]],
    coupling_operator: Callable[[Sequence[float], Sequence[float]], float],
    resonance_operator: Callable[[Sequence[float]], float],
    pairwise_coupling_operator: Callable[
        [Mapping[str, Sequence[float]], Sequence[tuple[str, str]]],
        Dict[str, float],
    ],
    recursive_filter_operator: Callable[[Sequence[float], float, float], Sequence[float]],
    stage_reception: Callable[..., tuple[Dict[str, object], Sequence[TelemetryRecord]]],
    stage_coherence: Callable[..., Dict[str, object]],
    stage_nodal: Callable[..., Dict[str, object]],
    stage_epi: Callable[..., Dict[str, object]],
    stage_sense: Callable[..., Dict[str, object]],
    stage_variability: Callable[..., Sequence[Mapping[str, object]]],
    aggregate_events: Callable[[Sequence[SupportsMicrosector] | None], Mapping[str, object]],
    compute_window_metrics: Callable[..., object],
    phase_context_resolver: Callable[
        [Sequence[SupportsMicrosector] | None],
        tuple[
            Dict[int, str],
            Dict[int, Mapping[str, Mapping[str, float] | float]],
        ],
    ],
    network_memory_extractor: Callable[[Mapping[str, Dict[str, object]] | None], Mapping[str, object]],
    zero_breakdown_factory: Callable[[], object],
    load_context_matrix: Callable[..., object],
    resolve_context_from_bundle: Callable[..., Mapping[str, float]],
    apply_contextual_delta: Callable[..., float],
    update_bundles: Callable[
        [Sequence[SupportsEPIBundle], Sequence[float], Sequence[float]],
        Sequence[SupportsEPIBundle],
    ],
    ensure_bundle: Callable[..., SupportsEPIBundle],
    normalise_node_evolution: Callable[..., Mapping[str, object]],
    delta_integral: Callable[..., Sequence[float]],
    variance_payload: Callable[..., Mapping[str, float]],
) -> PipelineDependencies:
    """Return the pipeline dependencies for the ΔNFR×Si workflow."""

    def _reception_stage(
        segments: Sequence[Sequence[TelemetryRecord]],
    ) -> tuple[Dict[str, object], Sequence[TelemetryRecord]]:
        return stage_reception(segments, reception_fn=reception_operator)

    def _dissonance_wrapper(
        series: Sequence[float],
        target: float,
        *,
        microsectors: Sequence[SupportsMicrosector] | None = None,
        bundles: Sequence[SupportsEPIBundle] | None = None,
    ) -> object:
        return dissonance_breakdown_operator(
            series,
            target,
            microsectors=microsectors,
            bundles=bundles,
        )

    def _coherence_stage(
        bundles: Sequence[SupportsEPIBundle],
        objectives: Mapping[str, float],
    ) -> Dict[str, object]:
        return stage_coherence(
            bundles,
            objectives,
            coherence_window=coherence_window,
            microsectors=microsectors,
            load_context_matrix=load_context_matrix,
            resolve_context_from_bundle=resolve_context_from_bundle,
            apply_contextual_delta=apply_contextual_delta,
            update_bundles=update_bundles,
            coherence_operator=coherence_operator,
            dissonance_operator=_dissonance_wrapper,
            coupling_operator=coupling_operator,
            resonance_operator=resonance_operator,
            empty_breakdown_factory=zero_breakdown_factory,
        )

    def _pairwise(
        series_by_node: Mapping[str, Sequence[float]],
        pairs: Sequence[tuple[str, str]],
    ) -> Dict[str, float]:
        return pairwise_coupling_operator(series_by_node, pairs=pairs)

    def _nodal_stage(bundles_seq: Sequence[SupportsEPIBundle]) -> Dict[str, object]:
        return stage_nodal(
            bundles_seq,
            load_context_matrix=load_context_matrix,
            xp=xp_module,
            structural_component=structural_component,
            pairwise_coupling=_pairwise,
        )

    def _epi_stage(
        records: Sequence[SupportsTelemetrySample],
        *,
        bundles: Sequence[SupportsEPIBundle] | None = None,
        phase_assignments: Mapping[int, str] | None = None,
        phase_weight_lookup: Mapping[
            int, Mapping[str, Mapping[str, float] | float]
        ] | None = None,
        global_phase_weights: Mapping[str, Mapping[str, float] | float] | None = None,
    ) -> Dict[str, object]:
        return stage_epi(
            records,
            bundles=bundles,
            phase_assignments=phase_assignments,
            phase_weight_lookup=phase_weight_lookup,
            global_phase_weights=global_phase_weights,
            ensure_bundle=ensure_bundle,
            normalise_node_evolution=normalise_node_evolution,
        )

    def _recursive_filter(series: Sequence[float], seed: float, decay: float) -> Sequence[float]:
        return recursive_filter_operator(series, seed=seed, decay=decay)

    def _sense_stage(series: Sequence[float]) -> Dict[str, object]:
        return stage_sense(
            series,
            recursion_decay=recursion_decay,
            recursive_filter=_recursive_filter,
        )

    def _variability_stage(
        microsectors_arg: Sequence[SupportsMicrosector] | None,
        bundles_arg: Sequence[SupportsEPIBundle],
        lap_indices: Sequence[int],
        lap_metadata: Sequence[Mapping[str, object]],
    ) -> Sequence[Mapping[str, object]]:
        return stage_variability(
            microsectors_arg,
            bundles_arg,
            lap_indices,
            lap_metadata,
            xp=xp_module,
            has_jax=has_jax,
            delta_integral=delta_integral,
            variance_payload=variance_payload,
        )

    return PipelineDependencies(
        emission_operator=emission_operator,
        reception_stage=_reception_stage,
        coherence_stage=_coherence_stage,
        nodal_stage=_nodal_stage,
        epi_stage=_epi_stage,
        sense_stage=_sense_stage,
        variability_stage=_variability_stage,
        aggregate_events=aggregate_events,
        window_metrics=compute_window_metrics,
        phase_context_resolver=phase_context_resolver,
        network_memory_extractor=network_memory_extractor,
        zero_breakdown_factory=zero_breakdown_factory,
    )
