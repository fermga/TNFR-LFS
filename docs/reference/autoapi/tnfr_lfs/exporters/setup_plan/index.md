# `tnfr_lfs.exporters.setup_plan` module
Utilities to serialise setup plans into exportable payloads.

## Classes
### `SetupChange`
Represents an actionable change in the car setup.

### `SetupPlan`
Structured plan combining optimisation insights with explainability.

## Functions
- `compute_phase_axis_summary(targets: Mapping[str, Mapping[str, float]] | None, weights: Mapping[str, Mapping[str, float]] | None) -> Tuple[Dict[str, Dict[str, str]], Tuple[str, ...]]`
- `phase_axis_summary_lines(summary: Mapping[str, Mapping[str, str]] | None) -> Tuple[str, ...]`
- `serialise_setup_plan(plan: SetupPlan) -> Dict[str, Any]`
  - Convert a :class:`SetupPlan` into a serialisable mapping.

