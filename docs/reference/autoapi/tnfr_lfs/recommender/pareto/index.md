# `tnfr_lfs.recommender.pareto` module
Pareto utilities to analyse multi-objective optimisation sweeps.

This module considers any non-finite metric value (``NaN``/``±inf``) as a
dominated outcome. Values are normalised during comparisons so that
invalid points are always outranked by finite counterparts.

## Classes
### `ParetoPoint`
Container describing the outcome of a candidate evaluation.

#### Methods
- `as_dict(self) -> Mapping[str, object]`
  - Return a serialisable representation of the point.

## Functions
- `pareto_front(points: Iterable[ParetoPoint]) -> list[ParetoPoint]`
  - Filter ``points`` keeping only Pareto optimal entries.

Candidates containing non-finite metrics are discarded before the
dominance check to ensure the resulting front only contains valid
evaluations.

