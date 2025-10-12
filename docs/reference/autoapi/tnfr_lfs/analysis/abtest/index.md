# `tnfr_lfs.analysis.abtest` module
Lap-level A/B comparison utilities.

## Classes
### `ABResult`
Summary of an A/B comparison between two telemetry stints.

## Functions
- `ab_compare_by_lap(baseline_metrics: Mapping[str, object], variant_metrics: Mapping[str, object], *, metric: str, iterations: int = 2000, alpha: float = 0.05, rng: random.Random | None = None) -> ABResult`
  - Compare two telemetry stints by aggregating lap-level metrics.

## Attributes
- `SUPPORTED_LAP_METRICS = ('sense_index', 'delta_nfr', 'coherence_index', 'delta_nfr_proj_longitudinal', 'delta_nfr_proj_lateral')`

