"""Core computation utilities for TNFR × LFS."""

from .coherence import compute_node_delta_nfr, sense_index
from .epi import DeltaCalculator, EPIExtractor, TelemetryRecord, delta_nfr_by_node
from .epi_models import EPIBundle
from .metrics import AeroCoherence, WindowMetrics, compute_aero_coherence, compute_window_metrics
from .operators import (
    DissonanceBreakdown,
    acoplamiento_operator,
    coherence_operator,
    dissonance_breakdown_operator,
    dissonance_operator,
    evolve_epi,
    emission_operator,
    mutation_operator,
    orchestrate_delta_metrics,
    recepcion_operator,
    recursivity_operator,
    recursividad_operator,
    resonance_operator,
    TyreBalanceControlOutput,
    tyre_balance_controller,
)
from .resonance import ModalAnalysis, ModalPeak, analyse_modal_resonance
from .segmentation import Goal, Microsector, segment_microsectors
from .structural_time import compute_structural_timestamps, resolve_time_axis

__all__ = [
    "TelemetryRecord",
    "EPIExtractor",
    "DeltaCalculator",
    "EPIBundle",
    "delta_nfr_by_node",
    "compute_node_delta_nfr",
    "sense_index",
    "compute_window_metrics",
    "compute_aero_coherence",
    "compute_structural_timestamps",
    "resolve_time_axis",
    "AeroCoherence",
    "WindowMetrics",
    "emission_operator",
    "recepcion_operator",
    "coherence_operator",
    "dissonance_operator",
    "dissonance_breakdown_operator",
    "DissonanceBreakdown",
    "acoplamiento_operator",
    "resonance_operator",
    "recursivity_operator",
    "mutation_operator",
    "recursividad_operator",
    "orchestrate_delta_metrics",
    "evolve_epi",
    "tyre_balance_controller",
    "TyreBalanceControlOutput",
    "Goal",
    "Microsector",
    "segment_microsectors",
    "analyse_modal_resonance",
    "ModalPeak",
    "ModalAnalysis",
]
