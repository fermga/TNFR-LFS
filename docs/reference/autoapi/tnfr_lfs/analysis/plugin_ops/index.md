# `tnfr_lfs.analysis.plugin_ops` module
Helpers bridging :mod:`tnfr_core` primitives with :class:`TNFRPlugin`.

## Functions
- `plugin_coherence_operator(plugin: TNFRPlugin, series: Sequence[float], window: int = 3) -> list[float]`
  - Run :func:`coherence_operator` and push the result into ``plugin``.
- `apply_plugin_nu_f_snapshot(plugin: TNFRPlugin, snapshot: NaturalFrequencySnapshot) -> NaturalFrequencySnapshot`
  - Copy the ν_f snapshot into the plugin's internal state.
- `resolve_plugin_nu_f(plugin: TNFRPlugin, record: SupportsTelemetrySample, *, phase: str | None = None, phase_weights: Mapping[str, Mapping[str, float] | float] | None = None, history: Sequence[SupportsTelemetrySample] | None = None, car_model: str | None = None, analyzer: NaturalFrequencyAnalyzer | None = None, settings: NaturalFrequencySettings | None = None, cache_options: 'CacheOptions' | None = None) -> NaturalFrequencySnapshot`
  - Resolve ν_f for ``record`` and synchronise the plugin state.

