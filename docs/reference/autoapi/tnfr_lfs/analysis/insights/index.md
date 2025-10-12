# `tnfr_lfs.analysis.insights` module
High-level processing helpers for telemetry insights.

## Classes
### `InsightsResult`
Structured output produced by :func:`compute_insights`.

#### Methods
- `with_robustness(self, *, bundles: Sequence[EPIBundle] | None = None, lap_indices: Sequence[int] | None = None, lap_metadata: Sequence[Mapping[str, Any]] | None = None, microsectors: Sequence[Microsector] | None = None, thresholds: Mapping[str, Mapping[str, float]] | None = None) -> 'InsightsResult'`
  - Return a new instance with robustness metrics recomputed.

## Functions
- `compute_insights(records: Sequence[TelemetryRecord], *, car_model: str, track_name: str, engine: RecommendationEngine | None = None, profile_manager: ProfileManager | None = None, robustness_bundles: Sequence[EPIBundle] | None = None, robustness_lap_indices: Sequence[int] | None = None, robustness_lap_metadata: Sequence[Mapping[str, Any]] | None = None, robustness_thresholds: Mapping[str, Mapping[str, float]] | None = None) -> InsightsResult`
  - Compute EPI bundles, microsectors and robustness metrics for a stint.

## Attributes
- `logger = logging.getLogger(__name__)`

