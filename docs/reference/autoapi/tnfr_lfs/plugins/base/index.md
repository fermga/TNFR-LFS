# `tnfr_lfs.plugins.base` module
Base classes for TNFR × LFS plugin implementations.

## Classes
### `TNFRPlugin` (ABC)
Abstract plugin contract for TNFR × LFS integrations.

Implementations provide identifying metadata together with dynamic state
derived from natural-frequency and coherence computations.  The base class
offers lifecycle hooks so concrete plugins can react whenever the spectral
inputs change.

#### Methods
- `nu_f(self) -> Mapping[str, float]`
  - Current natural frequency mapping keyed by subsystem node.
- `coherence_index(self) -> float`
  - Latest coherence index associated with the plugin.
- `reset_state(self) -> None`
  - Clear the dynamic plugin state and trigger :meth:`on_reset`.
- `on_reset(self) -> None`
  - Hook executed after :meth:`reset_state` clears plugin state.
- `apply_natural_frequency_snapshot(self, snapshot: 'NaturalFrequencySnapshot') -> None`
  - Update the ν_f mapping and invoke :meth:`on_nu_f_updated`.

Parameters
----------
snapshot:
    The spectral snapshot returned by
    :class:`~tnfr_lfs.core.epi.NaturalFrequencyAnalyzer`.
- `on_nu_f_updated(self, snapshot: 'NaturalFrequencySnapshot') -> None`
  - Hook executed after a new natural-frequency snapshot is applied.
- `apply_coherence_index(self, coherence_index: float, *, series: Sequence[float] | None = None) -> None`
  - Persist the coherence index and invoke :meth:`on_coherence_updated`.
- `apply_coherence_series(self, series: Sequence[float]) -> None`
  - Store coherence information derived from an operator output.
- `on_coherence_updated(self, coherence_index: float, series: Sequence[float] | None = None) -> None`
  - Hook executed whenever the coherence state changes.

