# `tnfr_core.metrics.resonance` module
Modal resonance analysis helpers.

## Classes
### `ModalPeak`
Single resonant peak extracted from the spectral analysis.

### `ModalAnalysis`
Aggregated spectral information for a single rotational axis.

## Functions
- `estimate_excitation_frequency(records: Sequence[SupportsTelemetrySample], sample_rate: float | None = None) -> float`
  - Return the dominant excitation frequency from :class:`SupportsTelemetrySample` data.
- `analyse_modal_resonance(records: Sequence[SupportsTelemetrySample], *, max_peaks: int = 3) -> Dict[str, ModalAnalysis]`
  - Compute modal energy for yaw/roll/pitch axes from telemetry-like sequences.

## Attributes
- `AxisSeries = Dict[str, List[float]]`

