"""Shared data structures for the pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Sequence

from tnfr_core.equations.telemetry import TelemetryRecord
from tnfr_core.runtime.shared import (
    SupportsEPIBundle,
    SupportsMicrosector,
    SupportsTelemetrySample,
)

__all__ = ["PipelineDependencies"]


@dataclass(frozen=True)
class PipelineDependencies:
    """Bundle the stage callables and helpers required by the orchestrator."""

    emission_operator: Callable[[float, float], Mapping[str, float]]
    reception_stage: Callable[
        [Sequence[Sequence[TelemetryRecord]]],
        tuple[Dict[str, object], Sequence[TelemetryRecord]],
    ]
    coherence_stage: Callable[
        [Sequence[SupportsEPIBundle], Mapping[str, float]],
        Dict[str, object],
    ]
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
    network_memory_extractor: Callable[
        [Mapping[str, Dict[str, object]] | None],
        Mapping[str, object],
    ]
    zero_breakdown_factory: Callable[[], object]

