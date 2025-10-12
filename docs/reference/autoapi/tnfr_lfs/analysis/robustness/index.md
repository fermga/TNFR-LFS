# `tnfr_lfs.analysis.robustness` module
Session robustness metrics helpers.

## Functions
- `compute_session_robustness(bundles: Sequence[EPIBundle] | None, *, lap_indices: Sequence[int] | None = None, lap_metadata: Sequence[Mapping[str, Any]] | None = None, microsectors: Sequence[Any] | None = None, thresholds: Mapping[str, Mapping[str, float]] | None = None) -> Mapping[str, Any]`
  - Return robustness statistics grouped by lap and driving phase.

