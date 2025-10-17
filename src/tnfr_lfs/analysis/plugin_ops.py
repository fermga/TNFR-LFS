"""Helpers bridging :mod:`tnfr_core` primitives with :class:`TNFRPlugin`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from tnfr_core.equations.epi import (
    NaturalFrequencyAnalyzer,
    NaturalFrequencySnapshot,
    NaturalFrequencySettings,
    resolve_nu_f_by_node,
)
from tnfr_core.operators import coherence_operator
from tnfr_core.runtime.shared import SupportsTelemetrySample
from tnfr_lfs.plugins import TNFRPlugin

if TYPE_CHECKING:  # pragma: no cover - optional cache dependency
    from tnfr_core.runtime.shared import CacheOptions


def plugin_coherence_operator(
    plugin: TNFRPlugin, series: Sequence[float], window: int = 3
) -> list[float]:
    """Run :func:`coherence_operator` and push the result into ``plugin``."""

    smoothed = coherence_operator(series, window=window)
    plugin.apply_coherence_series(smoothed)
    return smoothed


def apply_plugin_nu_f_snapshot(
    plugin: TNFRPlugin, snapshot: NaturalFrequencySnapshot
) -> NaturalFrequencySnapshot:
    """Copy the ν_f snapshot into the plugin's internal state."""

    plugin.apply_natural_frequency_snapshot(snapshot)
    return snapshot


def resolve_plugin_nu_f(
    plugin: TNFRPlugin,
    record: SupportsTelemetrySample,
    *,
    phase: str | None = None,
    phase_weights: Mapping[str, Mapping[str, float] | float] | None = None,
    history: Sequence[SupportsTelemetrySample] | None = None,
    car_model: str | None = None,
    analyzer: NaturalFrequencyAnalyzer | None = None,
    settings: NaturalFrequencySettings | None = None,
    cache_options: "CacheOptions" | None = None,
) -> NaturalFrequencySnapshot:
    """Resolve ν_f for ``record`` and synchronise the plugin state."""

    snapshot = resolve_nu_f_by_node(
        record,
        phase=phase,
        phase_weights=phase_weights,
        history=history,
        car_model=car_model,
        analyzer=analyzer,
        settings=settings,
        cache_options=cache_options,
    )
    plugin.apply_natural_frequency_snapshot(snapshot)
    return snapshot


__all__ = [
    "plugin_coherence_operator",
    "apply_plugin_nu_f_snapshot",
    "resolve_plugin_nu_f",
]
