# `tnfr_lfs.recommender.search` module
Search utilities to build optimisation-aware setup plans.

This module introduces a light-weight coordinate descent optimiser that works
on top of a decision vector whose bounds are defined per car model.  The
optimiser evaluates candidate vectors through a domain specific objective that
favours a higher Sense Index (Si) while penalising the integral of the absolute
ΔNFR within each microsector.  The resulting plan can be combined with the
rule-based recommendation engine to surface explainable guidance together with
the optimised setup deltas.

## Classes
### `DecisionVariable`
Represents an adjustable setup parameter bounded by the car model.

#### Methods
- `clamp(self, value: float) -> float`

### `DecisionSpace`
Collection of decision variables valid for a given car model.

#### Methods
- `initial_guess(self) -> Dict[str, float]`
- `clamp(self, vector: Mapping[str, float]) -> Dict[str, float]`

### `SearchResult`
Outcome of the optimisation stage.

### `Plan`
Aggregated plan mixing optimisation deltas with explainable rules.

### `CoordinateDescentOptimizer`
Simple coordinate descent routine with bound-aware steps.

#### Methods
- `optimise(self, objective: Callable[[Mapping[str, float]], float], space: DecisionSpace, initial_vector: Mapping[str, float] | None = None) -> tuple[Dict[str, float], float, int, int]`
  - Return the best decision vector according to ``objective``.

### `SetupPlanner`
High level API combining optimisation with explainable rules.

#### Methods
- `cache_size(self) -> int`
  - Return the maximum entries stored in the planner cache.
- `plan(self, baseline: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, *, car_model: str = 'XFG', track_name: str | None = None, simulator: Callable[[Mapping[str, float], Sequence[EPIBundle]], Sequence[EPIBundle]] | None = None) -> Plan`
  - Generate the final plan that blends search and rule-based guidance.

## Functions
- `objective_score(results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, *, session_weights: Mapping[str, Mapping[str, float]] | None = None, session_hints: Mapping[str, object] | None = None, breakdown: MutableMapping[str, float] | None = None) -> float`
  - Compute the Integrated Control Score combining Si and stability penalties.
- `evaluate_candidate(space: DecisionSpace, vector: Mapping[str, float], baseline: Sequence[EPIBundle], *, microsectors: Sequence[Microsector] | None = None, simulator: Callable[[Mapping[str, float], Sequence[EPIBundle]], Sequence[EPIBundle]] | None = None, session_weights: Mapping[str, Mapping[str, float]] | None = None, session_hints: Mapping[str, object] | None = None, cache: LRUCache[tuple[tuple[str, float], ...], tuple[float, tuple[EPIBundle, ...], Mapping[str, float], Mapping[str, float]]] | None = None) -> ParetoPoint`
  - Evaluate ``vector`` returning a :class:`ParetoPoint`.
- `axis_sweep_vectors(space: DecisionSpace, centre: Mapping[str, float], *, radius: int = 1, include_centre: bool = True) -> list[Dict[str, float]]`
  - Generate axis-aligned candidates around ``centre``.
- `sweep_candidates(space: DecisionSpace, centre: Mapping[str, float], baseline: Sequence[EPIBundle], *, microsectors: Sequence[Microsector] | None = None, simulator: Callable[[Mapping[str, float], Sequence[EPIBundle]], Sequence[EPIBundle]] | None = None, session_weights: Mapping[str, Mapping[str, float]] | None = None, session_hints: Mapping[str, object] | None = None, radius: int = 1, include_centre: bool = True, candidates: Iterable[Mapping[str, float]] | None = None, cache_size: int | None = None, cache_options: CacheOptions | None = None) -> list[ParetoPoint]`
  - Evaluate a sweep of candidates returning :class:`ParetoPoint` entries.

## Attributes
- `DecisionVector = Mapping[str, float]`
- `DEFAULT_DECISION_LIBRARY = {**_LFS_DECISION_LIBRARY, **{alias: _LFS_DECISION_LIBRARY[target] for alias, target in _ALIAS_DECISION_KEYS.items() if target in _LFS_DECISION_LIBRARY}}`

