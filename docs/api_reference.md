# API Reference

## Acquisition

### `tnfr_lfs.acquisition.outsim_client.OutSimClient`

Reads OutSim-style telemetry from CSV sources and returns a list of
:class:`tnfr_lfs.core.epi.TelemetryRecord` objects.  By default the
client validates the column order and converts values to floats.

```
from tnfr_lfs.acquisition import OutSimClient

client = OutSimClient()
records = client.ingest("stint.csv")
```

## Core Analytics

### `tnfr_lfs.core.epi.EPIExtractor`

Computes :class:`tnfr_lfs.core.epi_models.EPIBundle` objects including EPI,
ΔNFR, entropy-penalised sense index values, y el reparto nodal calculado por
:func:`tnfr_lfs.core.epi.delta_nfr_by_node`.  Para desgloses adicionales por
subcaracterística utilice :func:`tnfr_lfs.core.coherence.compute_node_delta_nfr`
combinado con :func:`tnfr_lfs.core.coherence.sense_index` dentro de pipelines
personalizados.  El nuevo Sense Index acepta el mapa de frecuencias naturales,
la fase activa y los pesos ``w_phase`` para evaluar ``1 / (1 + Σ w · |ΔNFR| ·
g(ν_f)) - λ·H`` con penalización entrópica configurable.

Each :class:`EPIBundle` records the per-node natural frequency ``nu_f`` (Hz),
allowing downstream tooling to weigh ΔNFR contributions using documented
subsystem dynamics.  Bundles now expose the derivative ``dEPI_dt`` and the
cumulatively integrated ``integrated_epi`` value obtained through explicit
Euler integration.

### Operator utilities

```
tnfr_lfs.core.operators.emission_operator(target_delta_nfr: float, target_sense_index: float) -> Dict[str, float]
tnfr_lfs.core.operators.recepcion_operator(records: Sequence[TelemetryRecord], extractor: Optional[EPIExtractor] = None) -> List[EPIBundle]
tnfr_lfs.core.operators.coherence_operator(series: Sequence[float], window: int = 3) -> List[float]
tnfr_lfs.core.operators.dissonance_operator(series: Sequence[float], target: float) -> float
tnfr_lfs.core.operators.acoplamiento_operator(series_a: Sequence[float], series_b: Sequence[float]) -> float
tnfr_lfs.core.operators.resonance_operator(series: Sequence[float]) -> float
tnfr_lfs.core.operators.recursividad_operator(series: Sequence[float], *, seed: float = 0.0, decay: float = 0.5) -> List[float]
tnfr_lfs.core.operators.orchestrate_delta_metrics(telemetry_segments: Sequence[Sequence[TelemetryRecord]], target_delta_nfr: float, target_sense_index: float, *, coherence_window: int = 3, recursion_decay: float = 0.4) -> Mapping[str, object]
tnfr_lfs.core.operators.evolve_epi(prev_epi: float, delta_map: Mapping[str, float], dt: float, nu_f_by_node: Mapping[str, float]) -> Tuple[float, float]
```

These functions compose an end-to-end ΔNFR/Sense Index pipeline covering
objective setting (Emisión), telemetry reception, smoothing (Coherencia),
contrast measurement (Disonancia), coupling analysis, resonance estimation,
recursive filtering, and orchestration over segmented telemetry streams.

### `tnfr_lfs.core.segmentation`

```
tnfr_lfs.core.segmentation.segment_microsectors(records: Sequence[TelemetryRecord], bundles: Sequence[EPIBundle]) -> List[Microsector]
tnfr_lfs.core.segmentation.Microsector
tnfr_lfs.core.segmentation.Goal
```

The segmentation module aligns telemetry samples with their EPI bundles to
derive microsectors that follow curvature, support events, and ΔNFR
signatures.  Each :class:`Microsector` exposes phase boundaries for entry,
apex, and exit together with an ``active_phase`` selector and the
``dominant_nodes`` mapping that lists the subsystems driving each phase.
The per-phase :class:`Goal` instances define not only the target ΔNFR and
Sense Index averages but also:

* ``nu_f_target`` – the weighted natural frequency of the fastest nodes in the
  phase.
* ``slip_lat_window`` / ``slip_long_window`` – allowable slip bands derived
  from the telemetry means under lateral/longitudinal load.
* ``yaw_rate_window`` – the expected yaw-rate envelope for the phase.
* ``dominant_nodes`` – the subsystems whose ΔNFR signature anchors the goal.

## Recommendation Engine

### `tnfr_lfs.recommender.rules.RecommendationEngine`

Applies load balance, stability index, and coherence rules to produce a
list of :class:`tnfr_lfs.recommender.rules.Recommendation` objects.
Custom rules can be added by implementing the
:class:`tnfr_lfs.recommender.rules.RecommendationRule` protocol.

### `tnfr_lfs.recommender.search`

```
tnfr_lfs.recommender.search.DecisionVariable
tnfr_lfs.recommender.search.DecisionSpace
tnfr_lfs.recommender.search.CoordinateDescentOptimizer
tnfr_lfs.recommender.search.objective_score(results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None) -> float
tnfr_lfs.recommender.search.SetupPlanner
tnfr_lfs.recommender.search.Plan
tnfr_lfs.recommender.search.SearchResult
```

The search utilities expose a bound-aware coordinate descent optimiser that
explores decision vectors tied to each car model.  The
:func:`objective_score` helper favours higher Sense Index values while
penalising ΔNFR integrals, and :class:`SetupPlanner` combines the optimiser
with the rule-based engine to emit an explainable :class:`Plan` object that
includes the telemetry trace, recommendations, and the resulting decision
vector.

## Exporters

Two exporters are provided out of the box:

* ``json`` – produces a JSON document ready for storage or pipelines.
* ``csv`` – returns a CSV representation of the EPI series.
* ``markdown`` – renders setup plans as Markdown tables with aggregated
  rationales and expected effects.

```
tnfr_lfs.exporters.setup_plan.SetupChange
tnfr_lfs.exporters.setup_plan.SetupPlan
tnfr_lfs.exporters.setup_plan.serialise_setup_plan(plan: SetupPlan) -> Dict[str, Any]
```

The setup plan dataclasses model the actionable advice produced by the
CLI's ``write-set`` command and by :class:`SetupPlanner`.  Use
:func:`serialise_setup_plan` to convert the plan into a JSON-compatible
payload with deduplicated rationales and expected effects.
