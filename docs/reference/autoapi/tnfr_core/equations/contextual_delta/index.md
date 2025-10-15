# `tnfr_core.equations.contextual_delta` module
Context-aware ΔNFR adjustments.

This module centralises the heuristics required to contextualise ΔNFR values
based on curvature, track surface grip and traffic density cues.  The
calibration payload is supplied by the host application via
:func:`configure_context_matrix_loader`, allowing :mod:`tnfr_core` to remain
agnostic of where the resources live.

## Classes
### `ContextFactors`
Triplet of multiplicative factors applied to ΔNFR.

#### Methods
- `as_mapping(self) -> Mapping[str, float]`
- `multiplier(self) -> float`

### `ContextMatrix`
Calibration payload loaded from ``context_factors.toml``.

#### Methods
- `curve_factor(self, value: float) -> float`
- `curve_band(self, value: float) -> tuple[float | None, float, str | None]`
- `surface_factor(self, ratio: float) -> float`
- `surface_band(self, ratio: float) -> tuple[float | None, float | None, float, str | None]`
- `traffic_factor(self, load: float) -> float`
- `traffic_band(self, load: float) -> tuple[float | None, float, str | None]`

## Functions
- `configure_context_matrix_loader(loader: ContextPayloadLoader) -> None`
  - Register ``loader`` as the source of contextual calibration payloads.
- `load_context_matrix(track: str | None = None, *, payload: ContextPayload | None = None) -> ContextMatrix`
  - Return the context matrix for ``track`` (defaults to the generic profile).

When ``payload`` is omitted the configured loader registered via
:func:`configure_context_matrix_loader` is invoked.  Passing ``payload``
bypasses the global cache and allows callers to supply bespoke calibration
data directly.
- `apply_contextual_delta(delta_value: float, factors: ContextFactors | Mapping[str, float], *, context_matrix: ContextMatrix | None = None) -> float`
  - Return ``delta_value`` scaled by the contextual factor matrix.
- `resolve_context_from_record(matrix: ContextMatrix, record: SupportsContextRecord | Mapping[str, object], *, baseline_vertical_load: float | None = None) -> ContextFactors`
  - Derive factors from a :class:`~tnfr_core.equations.epi.TelemetryRecord`-like payload.
- `resolve_context_from_bundle(matrix: ContextMatrix, bundle: SupportsContextBundle | Mapping[str, object]) -> ContextFactors`
  - Resolve factors using the information stored inside an :class:`EPIBundle`.
- `resolve_microsector_context(matrix: ContextMatrix, *, curvature: float, grip_rel: float, speed_drop: float, direction_changes: float) -> ContextFactors`
  - Resolve aggregate factors for a microsector from its summary metrics.
- `resolve_series_context(series: Iterable[SupportsContextBundle | SupportsContextRecord | Mapping[str, object]], *, matrix: ContextMatrix | None = None, baseline_vertical_load: float | None = None) -> list[ContextFactors]`
  - Return context factors for every element in ``series``.

## Attributes
- `ContextPayload = Mapping[str, object]`
- `ContextPayloadLoader = Callable[[], ContextPayload]`

