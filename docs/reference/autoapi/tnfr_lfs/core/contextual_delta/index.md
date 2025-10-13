# `tnfr_core.contextual_delta` module
Context-aware ΔNFR adjustments.

This module centralises the heuristics required to contextualise ΔNFR values
based on curvature, track surface grip and traffic density cues.  Host
applications must provide the calibration payload via
`configure_context_matrix_loader`, keeping :mod:`tnfr_core` agnostic of where
the data lives.  The default implementation ships with
`tnfr_lfs.analysis.contextual_delta` which wires the loader to the packaged
TOML resource.

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
  - Register a callable returning the contextual calibration payload.
- `load_context_matrix(track: str | None = None, *, payload: ContextPayload | None = None) -> ContextMatrix`
  - Return the context matrix for ``track`` (defaults to the generic profile).
- `apply_contextual_delta(delta_value: float, factors: ContextFactors | Mapping[str, float], *, context_matrix: ContextMatrix | None = None) -> float`
  - Return ``delta_value`` scaled by the contextual factor matrix.
- `resolve_context_from_record(matrix: ContextMatrix, record: SupportsContextRecord | Mapping[str, object], *, baseline_vertical_load: float | None = None) -> ContextFactors`
  - Derive factors from a :class:`~tnfr_core.epi.TelemetryRecord`-like payload.
- `resolve_context_from_bundle(matrix: ContextMatrix, bundle: SupportsContextBundle | Mapping[str, object]) -> ContextFactors`
  - Resolve factors using the information stored inside an :class:`EPIBundle`.
- `resolve_microsector_context(matrix: ContextMatrix, *, curvature: float, grip_rel: float, speed_drop: float, direction_changes: float) -> ContextFactors`
  - Resolve aggregate factors for a microsector from its summary metrics.
- `resolve_series_context(series: Sequence[SupportsContextBundle | SupportsContextRecord | Mapping[str, object]], *, matrix: ContextMatrix | None = None, baseline_vertical_load: float | None = None) -> list[ContextFactors]`
  - Return context factors for every element in ``series``.

