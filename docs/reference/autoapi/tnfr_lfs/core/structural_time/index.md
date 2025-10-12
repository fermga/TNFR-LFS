# `tnfr_lfs.core.structural_time` module
Structural time helpers based on event density heuristics.

## Functions
- `compute_structural_timestamps(records: Sequence[SupportsTelemetrySample], *, window_size: int = 5, weights: Mapping[str, float] | None = None, base_timestamp: float | None = None) -> list[float]`
  - Return a structural time axis derived from event density.

The helper inspects a subset of high-activation telemetry channels and
computes a moving average of their normalised derivatives.  The resulting
event density inflates the elapsed time for dense segments while keeping
quiet stretches close to the chronological axis.  The first timestamp is
anchored to ``base_timestamp`` when provided, otherwise it reuses the
earliest available reference in ``records``.
- `resolve_time_axis(sequence: Sequence[object], *, fallback_to_chronological: bool = True, structural_attr: str = 'structural_timestamp', chronological_attr: str = 'timestamp') -> list[float] | None`
  - Return a monotonically increasing time axis for ``sequence``.

When the structural axis is incomplete or not strictly ordered and
``fallback_to_chronological`` is true the chronological attribute is used
instead.  ``None`` is returned when no valid axis can be extracted.

## Attributes
- `DEFAULT_STRUCTURAL_WEIGHTS = {'brake_pressure': 0.35, 'throttle': 0.25, 'yaw_rate': 0.25, 'steer': 0.15}`

