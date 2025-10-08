# API Reference

## Configuration utilities

### `tnfr_lfs.config_loader`

The configuration loader resolves car manifests and profile bundles shipped
with the project.  Use :func:`tnfr_lfs.config_loader.load_cars` and
:func:`tnfr_lfs.config_loader.load_profiles` to populate immutable mappings of
cars and tuning profiles, then :func:`tnfr_lfs.config_loader.resolve_targets`
to expose the ``targets``/``policy``/``recommender`` sections for the selected
vehicle. 【F:tnfr_lfs/config_loader.py†L38-L148】

### `tnfr_lfs.track_loader`

The track loader complements the car/profile helpers by exposing the bundled
track manifests, phase-weight profiles, and per-combination modifiers.  The
module exposes:

```python
from tnfr_lfs.track_loader import (
    load_track,
    load_track_profiles,
    load_modifiers,
    assemble_session_weights,
)

track = load_track("SO")  # -> tnfr_lfs.track_loader.Track
profiles = load_track_profiles()
modifiers = load_modifiers()
session = assemble_session_weights(
    "gtr_mid",  # car profile or group
    track.configs["SO1"].track_profile,
    track_profiles=profiles,
    modifiers=modifiers,
)
```

* :func:`load_track` parses the ``[config.*]`` sections of the requested TOML
  manifest, normalising optional lengths and surface labels into
  :class:`tnfr_lfs.track_loader.TrackConfig` instances accessible from the
  returned :class:`tnfr_lfs.track_loader.Track`. 【F:tnfr_lfs/track_loader.py†L89-L147】
* :func:`load_track_profiles` discovers ``data/track_profiles/*.toml`` and
  returns a mapping keyed by ``meta.id`` (or filename) with normalised phase
  weights and hints ready for consumption. 【F:tnfr_lfs/track_loader.py†L149-L175】
* :func:`load_modifiers` scans ``modifiers/combos/*.toml`` and maps each
  ``(car_profile, track_profile)`` pair to its scale factors and optional
  hints. 【F:tnfr_lfs/track_loader.py†L178-L210】
* :func:`assemble_session_weights` merges a car profile with a track profile,
  applying any matching modifier scales to produce a ``{"weights", "hints"}``
  payload suitable for the CLI and HUD pipelines. 【F:tnfr_lfs/track_loader.py†L213-L247】

## Acquisition

### `tnfr_lfs.acquisition.outsim_client.OutSimClient`

Reads OutSim-style telemetry from CSV sources and returns a list of
:class:`tnfr_lfs.core.epi.TelemetryRecord` objects.  By default the
client validates the column order, converts values to floats, and keeps
any optional column that is missing in the source as ``math.nan`` instead
of fabricating estimates.【F:tnfr_lfs/acquisition/outsim_client.py†L87-L155】
This allows downstream exporters to surface "sin datos" ("no data") when Live for
Speed does not broadcast the extra wheel block.

```
from math import isnan

from tnfr_lfs.acquisition import OutSimClient

client = OutSimClient()
records = client.ingest("stint.csv")
# Optional columns remain math.nan when the CSV omits them, so
# downstream metrics can flag "sin datos" (“no data”).
isnan(records[0].tyre_temp_fl)
```

## Core Analytics

### `tnfr_lfs.core.epi.EPIExtractor`

Computes :class:`tnfr_lfs.core.epi_models.EPIBundle` objects including EPI,
ΔNFR, entropy-penalised sense index values, and the nodal distribution
calculated by :func:`tnfr_lfs.core.epi.delta_nfr_by_node`.  For additional
subcomponent breakdowns use
:func:`tnfr_lfs.core.coherence.compute_node_delta_nfr` together with
:func:`tnfr_lfs.core.coherence.sense_index` inside custom pipelines.  The
updated Sense Index consumes the natural-frequency map, the active phase, and
the ``w_phase`` weights to evaluate ``1 / (1 + Σ w · |ΔNFR| · g(ν_f)) - λ·H``
with a configurable entropy penalty.

Each :class:`EPIBundle` records the per-node natural frequency ``nu_f`` (Hz),
allowing downstream tooling to weigh ΔNFR contributions using documented
subsystem dynamics.  Bundles now expose the derivative ``dEPI_dt`` and the
cumulatively integrated ``integrated_epi`` value obtained through explicit
Euler integration.  The ΔNFR signal is also decomposed into longitudinal and
lateral components via the ``delta_nfr_proj_longitudinal``/``delta_nfr_proj_lateral``
fields which project the nodal gradient onto each axis, providing immediate
visibility of brake/traction versus balance imbalances while keeping the raw
`Fz`/`ΔFz` channels available for force-based analysis.

### Operator utilities

```
tnfr_lfs.core.operators.emission_operator(target_delta_nfr: float, target_sense_index: float) -> Dict[str, float]
tnfr_lfs.core.operators.recepcion_operator(records: Sequence[TelemetryRecord], extractor: Optional[EPIExtractor] = None) -> List[EPIBundle]
tnfr_lfs.core.operators.coherence_operator(series: Sequence[float], window: int = 3) -> List[float]
tnfr_lfs.core.operators.dissonance_operator(series: Sequence[float], target: float) -> float
tnfr_lfs.core.operators.dissonance_breakdown_operator(series: Sequence[float], target: float, *, microsectors: Optional[Sequence[Microsector]] = None, bundles: Optional[Sequence[EPIBundle]] = None) -> DissonanceBreakdown
tnfr_lfs.core.operators.acoplamiento_operator(series_a: Sequence[float], series_b: Sequence[float]) -> float
tnfr_lfs.core.operators.resonance_operator(series: Sequence[float]) -> float
tnfr_lfs.core.operators.recursividad_operator(series: Sequence[float], *, seed: float = 0.0, decay: float = 0.5) -> List[float]
tnfr_lfs.core.operators.orchestrate_delta_metrics(telemetry_segments: Sequence[Sequence[TelemetryRecord]], target_delta_nfr: float, target_sense_index: float, *, coherence_window: int = 3, recursion_decay: float = 0.4, microsectors: Optional[Sequence[Microsector]] = None, phase_weights: Optional[Mapping[str, Mapping[str, float] | float]] = None, operator_state: Optional[Mapping[str, Dict[str, object]]] = None) -> Mapping[str, object]
tnfr_lfs.core.operators.evolve_epi(prev_epi: float, delta_map: Mapping[str, float], dt: float, nu_f_by_node: Mapping[str, float]) -> Tuple[float, float]
```

These functions compose an end-to-end ΔNFR/Sense Index pipeline covering
objective setting (Emission), telemetry reception, smoothing (Coherence),
contrast measurement (Dissonance) with a breakdown of useful vs. parasitic
support events, coupling analysis, resonance estimation, recursive filtering,
and orchestration over segmented telemetry streams. The orchestrator exposes
``microsector_variability`` entries summarising ΔNFR↓ and Sense Index variance
plus population-standard deviation for each microsector, alongside a
``lap_sequence`` describing the lap labels detected in the incoming segments.
The :class:`DissonanceBreakdown` payload now includes the Useful Dissonance
Ratio (UDR) through the ``useful_dissonance_ratio``/``useful_dissonance_samples``
fields, quantifying the fraction of high yaw-acceleration samples where
ΔNFR is already decaying.

``orchestrate_delta_metrics`` also surfaces support metrics derived from
``WindowMetrics``: ``support_effective`` (ΔNFR sustained by tyres and suspension
with structural weighting), ``load_support_ratio`` (normalised by the average
Fz load), and the pairs
``structural_expansion_longitudinal``/``structural_contraction_longitudinal`` and
``structural_expansion_lateral``/``structural_contraction_lateral`` which
describe how the longitudinal/lateral ΔNFR components expand or compress the
structural axis of the analysed window.  The steering budgets
``ackermann_parallel_index`` and ``slide_catch_budget`` are derived exclusively
from the ``slip_angle_*`` channels and the ``yaw_rate`` emitted by OutSim; when
that Live for Speed telemetry is missing the output reports the literal
``"sin datos"`` (“no data”) marker.  The ``aero_balance_drift`` entry groups the average
rake (pitch plus per-axle travel) and the ``μ_front - μ_rear`` delta for low,
medium, and high speed bands.  Rake relies solely on the ``pitch`` and
suspension-travel channels provided by OutSim, ensuring the aerodynamic drift
matches the native data even if ``AeroCoherence`` appears neutral.

``WindowMetrics`` also publishes the ΔNFR entropy. ``delta_nfr_entropy``
summarises the structural distribution per phase (0 ≙ energy concentrated in a
single phase, 1 ≙ balanced window across all observed phases) while
``node_entropy`` captures the overall nodal diversity using the same normalised
range [0, 1]. The ``phase_delta_nfr_entropy`` map contains the normalised phase
probabilities (summing to 1.0) used to compute entropy, and
``phase_node_entropy`` details the Shannon entropy per phase derived from the
nodal contributions.  Both maps stay within the [0, 1] interval and preserve the
legacy phase aliases so HUDs and the CLI can consume them directly.

``WindowMetrics.cphi`` now yields a :class:`~tnfr_lfs.core.metrics.CPHIReport`
with per-wheel :class:`~tnfr_lfs.core.metrics.CPHIWheel` data and the shared
:class:`~tnfr_lfs.core.metrics.CPHIThresholds`. The thresholds follow a
red/amber/green traffic-light scheme so HUD pages and CLI reports colour tyre
health consistently. Consumers that require the historical flat keys can rely
on :meth:`~tnfr_lfs.core.metrics.CPHIReport.as_legacy_mapping`.

When the ``operator_state`` shared by ``segment_microsectors`` is supplied, the
orchestrator adds the ``network_memory`` field and a mirror in
``sense_memory["network"]`` with the per-session memory
(``car_model``/``track_name``/``tyre_compound``), including stint histories and
each microsector’s active state.  If OutGauge does not transmit the extended
per-wheel temperature/pressure block, HUD surfaces and exporters show the
literal ``"sin datos"`` (“no data”) token in those fields to make it clear that LFS telemetry
was unavailable.

### `tnfr_lfs.core.segmentation`

```
tnfr_lfs.core.segmentation.segment_microsectors(records: Sequence[TelemetryRecord], bundles: Sequence[EPIBundle], *, phase_weight_overrides: Optional[Mapping[str, Mapping[str, float] | float]] = None) -> List[Microsector]
tnfr_lfs.core.segmentation.Microsector
tnfr_lfs.core.segmentation.Goal
```

The segmentation module aligns telemetry samples with their EPI bundles to
derive microsectors that follow curvature, support events, and ΔNFR
signatures.  Each :class:`Microsector` exposes phase boundaries for entry,
apex, and exit together with an ``active_phase`` selector and the
``dominant_nodes`` mapping that lists the subsystems driving each phase.
The per-phase :class:`Goal` instances define not only the target ΔNFR and
Sense Index averages but also longitudinal/lateral ΔNFR objectives and the
relative weighting between them.  These values are propagated to
:class:`Microsector` objects (``phase_axis_targets``/``phase_axis_weights``)
so downstream recommenders and exporters can highlight whether the
microsector demands longitudinal support (brake bias, differential locking)
or lateral balance (anti-roll bars, toe, alignments).

The ``filtered_measures`` mapping now exposes a structured ``"cphi"`` block
that mirrors :class:`~tnfr_lfs.core.metrics.CPHIReport`, including the
shared traffic-light thresholds so JSON exports and dashboards can reuse
the same red/amber/green semantics without recomputing the bands.

* ``nu_f_target`` – the weighted natural frequency of the fastest nodes in the
  phase.
* ``nu_exc_target`` – the dominant excitation frequency derived from steer and
  suspension inputs inside the phase window.
* ``rho_target`` – the detune ratio ``ν_exc/ν_f`` observed for the archetype,
  used to flag chassis setups that fall out of tune under load.
* ``slip_lat_window`` / ``slip_long_window`` – allowable slip bands derived
  from the telemetry means under lateral/longitudinal load.
* ``yaw_rate_window`` – the expected yaw-rate envelope for the phase.
* ``dominant_nodes`` – the subsystems whose ΔNFR signature anchors the goal.
* ``window_occupancy`` – percentage of telemetry samples that remain within
  each window for entry, apex, and exit.
* ``operator_events`` – grouped by ``AL``/``OZ``/``IL``/``SILENCIO`` and enriched
  with the surface type derived from ``context_factors`` plus the contextual
  ΔNFR threshold (``delta_nfr_threshold``) for the microsector.  Each event
  lists the peak/mean ΔNFR observed in the window and its ratio to the
  threshold (``delta_nfr_ratio``), while the aggregate ``SILENCIO`` entry adds
  average coverage and structural density to flag low-activation latent states.

## Recommendation Engine

### `tnfr_lfs.recommender.rules.RecommendationEngine`

Applies load balance, stability index, and coherence rules to produce a
list of :class:`tnfr_lfs.recommender.rules.Recommendation` objects.
Custom rules can be added by implementing the
:class:`tnfr_lfs.recommender.rules.RecommendationRule` protocol. The default
rule-set now evaluates the Useful Dissonance Ratio (UDR) to suggest stiffening
rear support/LSD when yaw impulses fail to tame ΔNFR, or to soften the
offending axle when UDR collapses.

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
includes the telemetry trace, recommendations, the resulting decision vector,
and the Sense Control Index (SCI) with its component breakdown.

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
payload with deduplicated rationales and expected effects.  The
``sensitivities`` mapping now captures both ΔSi/Δp and
Δ∫|ΔNFR|/Δp entries, while ``phase_sensitivities`` stores the
per-phase Δ∫|ΔNFR|/Δp gradients used by the optimiser and the
profile persistence layer.  The exported mapping also includes the aggregate
SCI value and a fixed-order SCI breakdown so downstream dashboards can report
the optimisation balance directly.
