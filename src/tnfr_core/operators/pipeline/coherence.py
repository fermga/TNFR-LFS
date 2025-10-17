"""Coherence stage helpers for the ΔNFR×Si pipeline."""

from __future__ import annotations

from statistics import mean
from typing import Callable, Dict, Mapping, Sequence

from tnfr_core.equations.contextual_delta import ContextMatrix
from tnfr_core.runtime.shared import SupportsEPIBundle, SupportsMicrosector
from tnfr_core.operators._types import DissonanceBreakdown

BundleUpdater = Callable[
    [Sequence[SupportsEPIBundle], Sequence[float], Sequence[float]],
    Sequence[SupportsEPIBundle],
]
CoherenceOperator = Callable[[Sequence[float], int], Sequence[float]]
       
DissonanceOperator = Callable[[Sequence[float], float], DissonanceBreakdown]
CouplingOperator = Callable[[Sequence[float], Sequence[float]], float]
ResonanceOperator = Callable[[Sequence[float]], float]
ContextResolver = Callable[[ContextMatrix, SupportsEPIBundle], Mapping[str, float]]
ContextLoader = Callable[[], ContextMatrix]
DeltaApplier = Callable[[float, Mapping[str, float], ContextMatrix], float]


def _stage_coherence(
    bundles: Sequence[SupportsEPIBundle],
    objectives: Mapping[str, float],
    *,
    coherence_window: int,
    microsectors: Sequence[SupportsMicrosector] | None,
    load_context_matrix: ContextLoader,
    resolve_context_from_bundle: ContextResolver,
    apply_contextual_delta: DeltaApplier,
    update_bundles: BundleUpdater,
    coherence_operator: CoherenceOperator,
    dissonance_operator: DissonanceOperator,
    coupling_operator: CouplingOperator,
    resonance_operator: ResonanceOperator,
    empty_breakdown_factory: Callable[[], DissonanceBreakdown],
) -> Dict[str, object]:
    """Return coherence metrics for ``bundles`` and ``objectives``."""

    if not bundles:
        empty_breakdown = empty_breakdown_factory()
        return {
            "raw_delta": [],
            "raw_sense_index": [],
            "smoothed_delta": [],
            "smoothed_sense_index": [],
            "bundles": [],
            "dissonance": 0.0,
            "dissonance_breakdown": empty_breakdown,
            "coupling": 0.0,
            "resonance": 0.0,
            "coherence_index_series": [],
            "coherence_index": 0.0,
            "raw_coherence_index": 0.0,
            "frequency_label": "",
            "frequency_classification": "no data",
        }

    context_matrix = load_context_matrix()
    bundle_context = [
        resolve_context_from_bundle(context_matrix, bundle) for bundle in bundles
    ]
    delta_series = [
        apply_contextual_delta(
            bundle.delta_nfr,
            factors,
            context_matrix=context_matrix,
        )
        for bundle, factors in zip(bundles, bundle_context)
    ]
    si_series = [bundle.sense_index for bundle in bundles]
    smoothed_delta = coherence_operator(delta_series, coherence_window)
    smoothed_si = coherence_operator(si_series, coherence_window)
    clamped_si = [max(0.0, min(1.0, value)) for value in smoothed_si]
    updated_bundles = update_bundles(bundles, smoothed_delta, clamped_si)
    breakdown = dissonance_operator(
        smoothed_delta,
        objectives.get("delta_nfr", 0.0),
        microsectors=microsectors,
        bundles=updated_bundles,
    )
    dissonance = breakdown.value
    coupling = coupling_operator(smoothed_delta, clamped_si)
    resonance = resonance_operator(clamped_si)
    ct_series = [bundle.coherence_index for bundle in updated_bundles]
    average_ct = mean(ct_series) if ct_series else 0.0
    mean_si = mean(clamped_si) if clamped_si else 0.0
    target_si = max(1e-6, min(1.0, float(objectives.get("sense_index", 0.75))))
    normalised_ct = max(0.0, min(1.0, average_ct * (mean_si / target_si)))
    frequency_label = ""
    frequency_classification = "no data"
    if updated_bundles:
        last_bundle = updated_bundles[-1]
        frequency_label = getattr(last_bundle, "nu_f_label", "")
        frequency_classification = getattr(
            last_bundle, "nu_f_classification", "no data"
        )

    return {
        "raw_delta": delta_series,
        "raw_sense_index": si_series,
        "smoothed_delta": smoothed_delta,
        "smoothed_sense_index": clamped_si,
        "bundles": updated_bundles,
        "dissonance": dissonance,
        "dissonance_breakdown": breakdown,
        "coupling": coupling,
        "resonance": resonance,
        "coherence_index_series": ct_series,
        "coherence_index": normalised_ct,
        "raw_coherence_index": average_ct,
        "frequency_label": frequency_label,
        "frequency_classification": frequency_classification,
    }
