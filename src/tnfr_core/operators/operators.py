"""Compatibility layer re-exporting high-level telemetry operators."""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Dict, Mapping, Sequence

from tnfr_core._canonical import CANONICAL_REQUESTED, import_tnfr
from tnfr_core.equations.epi import (
    NaturalFrequencyAnalyzer,
    TelemetryRecord,
    delta_nfr_by_node,
    resolve_nu_f_by_node,
)
from tnfr_core.operators.entry.recursivity import (
    RecursivityMicroState,
    RecursivityMicroStateSnapshot,
    RecursivityNetworkHistoryEntry,
    RecursivityNetworkMemory,
    RecursivityNetworkSession,
    RecursivityOperatorResult,
    RecursivitySessionState,
    RecursivityStateRoot,
    recursivity_operator,
)
from tnfr_core.runtime.shared import (
    SupportsChassisNode,
    SupportsEPIBundle,
    SupportsMicrosector,
    SupportsSuspensionNode,
    SupportsTelemetrySample,
    SupportsTyresNode,
)
from tnfr_core.operators.operator_detection import (
    normalize_structural_operator_identifier,
    silence_event_payloads,
)
from tnfr_core.operators.structural.coherence_il import coherence_operator_il
from tnfr_core.equations.epi_evolution import NodalEvolution
from tnfr_core.equations import epi_evolution as _epi_evolution
from tnfr_core.operators.pipeline import (
    orchestrate_delta_metrics as pipeline_orchestrate_delta_metrics,
)

from .al_operator import (
    DissonanceBreakdown,
    _batch_coupling,
    _prepare_series_pair,
    _zero_dissonance_breakdown,
    acoplamiento_operator,
    coherence_operator,
    coupling_operator,
    dissonance_breakdown_operator,
    dissonance_operator,
    pairwise_coupling_operator,
    resonance_operator,
)
from .en_operator import (
    _clone_bundle,
    _coerce_node,
    _ensure_bundle,
    _normalise_delta_breakdown,
    _normalise_node_evolution,
    _update_bundles,
    emission_operator,
    reception_operator,
)
from .il_operator import (
    TyreBalanceControlOutput,
    _delta_integral_series,
    _variance_payload,
    mutation_operator,
    recursive_filter_operator,
    recursividad_operator,
    tyre_balance_controller,
)


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
) -> Mapping[str, object]:
    """Compatibility wrapper delegating to the pipeline orchestrator."""

    warnings.warn(
        "'tnfr_core.operators.operators.orchestrate_delta_metrics' is deprecated; "
        "import 'tnfr_core.operators.orchestrate_delta_metrics' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return pipeline_orchestrate_delta_metrics(
        telemetry_segments,
        target_delta_nfr,
        target_sense_index,
        coherence_window=coherence_window,
        recursion_decay=recursion_decay,
        microsectors=microsectors,
        phase_weights=phase_weights,
        operator_state=operator_state,
    )


EPI_EVOLUTION_DEPRECATION = (
    "'tnfr_core.operators.operators.evolve_epi' is deprecated; import "
    "'tnfr_core.operators.structural.epi_evolution.evolve_epi' instead."
)


def evolve_epi(
    prev_epi: float,
    delta_map: Mapping[str, float],
    dt: float,
    nu_f_by_node: Mapping[str, float],
):
    """Compatibility wrapper around :func:`structural.epi_evolution.evolve_epi`."""

    warnings.warn(EPI_EVOLUTION_DEPRECATION, DeprecationWarning, stacklevel=2)
    return _epi_evolution.evolve_epi(prev_epi, delta_map, dt, nu_f_by_node)


__all__ = [
    "DissonanceBreakdown",
    "TyreBalanceControlOutput",
    "_batch_coupling",
    "_clone_bundle",
    "_coerce_node",
    "_delta_integral_series",
    "_ensure_bundle",
    "_normalise_delta_breakdown",
    "_normalise_node_evolution",
    "_prepare_series_pair",
    "_update_bundles",
    "_variance_payload",
    "_zero_dissonance_breakdown",
    "acoplamiento_operator",
    "coherence_operator",
    "coherence_operator_il",
    "coupling_operator",
    "delta_nfr_by_node",
    "dissonance_breakdown_operator",
    "dissonance_operator",
    "emission_operator",
    "evolve_epi",
    "mutation_operator",
    "orchestrate_delta_metrics",
    "pairwise_coupling_operator",
    "recursividad_operator",
    "recursivity_operator",
    "reception_operator",
    "recursive_filter_operator",
    "resonance_operator",
    "resolve_nu_f_by_node",
    "silence_event_payloads",
    "tyre_balance_controller",
]


if CANONICAL_REQUESTED:  # pragma: no cover - depends on optional package
    tnfr = import_tnfr()
    canonical_ops = import_module(f"{tnfr.__name__}.operators.operators")

    canonical_exports = getattr(canonical_ops, "__all__", None)
    if canonical_exports is not None:
        __all__ = list(dict.fromkeys([*canonical_exports, *__all__]))

    for name in dir(canonical_ops):
        if name.startswith("__"):
            continue
        globals()[name] = getattr(canonical_ops, name)
