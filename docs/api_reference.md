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
This allows downstream exporters to surface "sin datos" when Live for
Speed does not broadcast the extra wheel block.

```
from math import isnan

from tnfr_lfs.acquisition import OutSimClient

client = OutSimClient()
records = client.ingest("stint.csv")
# Optional columns remain math.nan when the CSV omits them, so
# downstream metrics can flag "sin datos".
isnan(records[0].tyre_temp_fl)
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
Euler integration.  The ΔNFR signal is also decomposed into longitudinal and
lateral components via the ``delta_nfr_longitudinal``/``delta_nfr_lateral``
fields which are derived from the underlying feature contributions and
provide immediate visibility of brake/traction versus balance imbalances.

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
objective setting (Emisión), telemetry reception, smoothing (Coherencia),
contrast measurement (Disonancia) with a breakdown of useful vs. parasitic
support events, coupling analysis, resonance estimation, recursive filtering,
and orchestration over segmented telemetry streams. The orchestrator exposes
``microsector_variability`` entries summarising ΔNFR↓ and Sense Index variance
plus population-standard deviation for each microsector, alongside a
``lap_sequence`` describing the lap labels detected in the incoming segments.
The :class:`DissonanceBreakdown` payload now includes the Useful Dissonance
Ratio (UDR) through the ``useful_dissonance_ratio``/``useful_dissonance_samples``
fields, quantifying the fraction of high yaw-acceleration samples where
ΔNFR is already decaying.

``orchestrate_delta_metrics`` también expone métricas de apoyo derivadas de
``WindowMetrics``: ``support_effective`` (ΔNFR sostenido por neumáticos y
suspensión ponderado estructuralmente), ``load_support_ratio`` (normalizado por
la carga vertical media) y los pares
``structural_expansion_longitudinal``/``structural_contraction_longitudinal`` y
``structural_expansion_lateral``/``structural_contraction_lateral`` que
describen cómo las componentes longitudinal/lateral de ΔNFR expanden o
comprimen el eje estructural de la ventana analizada.  Los presupuestos de
dirección ``ackermann_parallel_index`` y ``slide_catch_budget`` se calculan
exclusivamente a partir de los ``slip_angle_*`` y del ``yaw_rate`` que OutSim
emite; cuando esa telemetría de Live for Speed no está presente la salida
reporta ``"sin datos"``.  Además se publica ``aero_balance_drift``, que agrupa
el rake medio (pitch + viajes por eje) y la diferencia ``μ_front - μ_rear`` por
bandas de velocidad baja/media/alta.  El rake se evalúa únicamente con el
``pitch`` y los viajes de suspensión proporcionados por OutSim, garantizando
que la deriva aerodinámica refleja directamente los datos nativos incluso si
``AeroCoherence`` todavía parece neutra.

Cuando se comparte el ``operator_state`` utilizado por
``segment_microsectors``, la orquestación añade el campo ``network_memory`` y
un espejo en ``sense_memory["network"]`` con la memoria de red por sesión
(``car_model``/``track_name``/``tyre_compound``), incluyendo historiales por
stint y el estado activo de cada microsector.  Si OutGauge no transmite el
bloque extendido con temperaturas/presiones por rueda, las superficies de HUD y
los exportadores muestran ``"sin datos"`` en esos campos para dejar claro que la
telemetría de LFS no estaba disponible.

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
microsector demands longitudinal support (bias de frenos, bloqueo del
diferencial) or lateral balance (barras, toe, alineaciones).

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
* ``operator_events`` – agrupados por ``AL``/``OZ``/``IL``/``SILENCIO`` e
  incrementados con el tipo de superficie derivado de ``context_factors`` y
  el umbral ΔNFR contextual (``delta_nfr_threshold``) asociado al microsector.
  Cada evento incluye el pico/medio ΔNFR observado durante la ventana y la
  relación frente al umbral (``delta_nfr_ratio``), mientras que el agregado
  ``SILENCIO`` añade cobertura y densidad estructural medias para identificar
  estados latentes de baja activación.

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
