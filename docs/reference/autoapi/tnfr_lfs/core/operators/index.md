# `tnfr_lfs.core.operators` module
High-level TNFR × LFS operators for telemetry analytics pipelines.

## Classes
### `DissonanceBreakdown`
Breakdown of dissonance events by usefulness.

### `RecursivityMicroState` (TypedDict)
In-memory representation of a microsector recursivity state.

### `RecursivityMicroStateSnapshot` (TypedDict)
Serialisable snapshot of a microsector recursivity state.

### `RecursivityHistoryEntry` (TypedDict)
Historical record for a completed stint.

### `RecursivitySessionState` (TypedDict)
State container for an entire recursivity session.

### `RecursivityStateRoot` (TypedDict)
Root object storing sessions keyed by identifier.

### `RecursivityOperatorResult` (TypedDict)
Return payload from :func:`recursivity_operator`.

### `RecursivityNetworkHistoryEntry` (TypedDict)
Snapshot of a historical stint for network payloads.

### `RecursivityNetworkSession` (TypedDict)
Session snapshot serialised for network transmission.

### `RecursivityNetworkMemory` (TypedDict)
Root payload returned by :func:`_extract_network_memory`.

### `TyreBalanceControlOutput`
Aggregated ΔP/Δcamber recommendations for a stint.

## Functions
- `evolve_epi(prev_epi: float, delta_map: Mapping[str, float], dt: float, nu_f_by_node: Mapping[str, float]) -> tuple[float, float, Dict[str, tuple[float, float]]]`
  - Integrate the Event Performance Index using explicit Euler steps.

The integrator now returns the global derivative/integral together with a
per-node breakdown.  The nodal contribution dictionary maps the node name
to a ``(integral, derivative)`` tuple representing the instantaneous
change produced during ``dt``.
- `emission_operator(target_delta_nfr: float, target_sense_index: float) -> Dict[str, float]`
  - Return normalised objectives for ΔNFR and sense index targets.
- `reception_operator(records: Sequence[TelemetryRecord], extractor: EPIExtractor | None = None) -> List[EPIBundle]`
  - Convert raw telemetry records into EPI bundles.
- `coherence_operator(series: Sequence[float], window: int = 3) -> List[float]`
  - Smooth a numeric series while preserving its average value.
- `plugin_coherence_operator(plugin: TNFRPlugin, series: Sequence[float], window: int = 3) -> List[float]`
  - Run :func:`coherence_operator` and push the result into ``plugin``.
- `dissonance_operator(series: Sequence[float], target: float) -> float`
  - Compute the mean absolute deviation relative to a target value.
- `dissonance_breakdown_operator(series: Sequence[float], target: float, *, microsectors: Sequence[SupportsMicrosector] | None = None, bundles: Sequence[SupportsEPIBundle] | None = None) -> DissonanceBreakdown`
  - Classify support events into useful (positive) and parasitic dissonance.
- `coupling_operator(series_a: Sequence[float], series_b: Sequence[float], *, strict_length: bool = True) -> float`
  - Return the normalised coupling (correlation) between two series.
- `acoplamiento_operator(series_a: Sequence[float], series_b: Sequence[float], *, strict_length: bool = True) -> float`
  - Compatibility wrapper for :func:`coupling_operator`.
- `pairwise_coupling_operator(series_by_node: Mapping[str, Sequence[float]], *, pairs: Sequence[tuple[str, str]] | None = None) -> Dict[str, float]`
  - Compute coupling metrics for each node pair using :func:`coupling_operator`.
- `resonance_operator(series: Sequence[float]) -> float`
  - Compute the root-mean-square (RMS) resonance of a series.
- `recursivity_operator(state: RecursivityStateRoot, session_id: Mapping[str, object] | Sequence[object] | str | None, microsector_id: str, measures: Mapping[str, float | str], *, decay: float = 0.4, history: int = 20, max_samples: int = 600, max_time_gap: float = 60.0, convergence_window: int = 5, convergence_threshold: float = 0.02) -> RecursivityOperatorResult`
  - Maintain recursive thermal/style state per session and microsector.
- `tyre_balance_controller(filtered_metrics: Mapping[str, float], *, delta_nfr_flat: float | None = None, target_front: float = 0.82, target_rear: float = 0.8, pressure_gain: float = 0.25, nfr_gain: float = 0.2, pressure_max_step: float = 0.16, camber_gain: float = 0.18, camber_max_step: float = 0.25, bias_gain: float = 0.04, offsets: Mapping[str, float] | None = None) -> TyreBalanceControlOutput`
  - Compute ΔP and camber tweaks from CPHI-derived tyre metrics.
- `mutation_operator(state: MutableMapping[str, Dict[str, object]], triggers: Mapping[str, object], *, entropy_threshold: float = 0.65, entropy_increase: float = 0.08, style_threshold: float = 0.12) -> Dict[str, object]`
  - Update the target archetype when entropy or style shifts are detected.

The operator keeps per-microsector memory of the previous entropy, style
index and active phase to detect meaningful regime changes.  When the
entropy rises sharply or the driving style drifts outside the configured
window the archetype mutates to the provided candidate or fallback.

Parameters
----------
state:
    Mutable mapping storing the mutation state per microsector.
triggers:
    Mapping providing the measurements required to evaluate the mutation
    rules.  Expected keys include ``"microsector_id"``,
    ``"current_archetype"``, ``"candidate_archetype"``,
    ``"fallback_archetype"``, ``"entropy"``, ``"style_index"``,
    ``"style_reference"`` and ``"phase"``.
entropy_threshold:
    Absolute entropy level that must be reached to trigger a fallback
    archetype.
entropy_increase:
    Minimum entropy delta compared to the stored baseline to trigger the
    fallback archetype.
style_threshold:
    Allowed absolute deviation between the filtered style index and the
    reference target before mutating towards the candidate archetype.

Returns
-------
dict
    A dictionary with the selected archetype, whether a mutation happened
    and diagnostic information.
- `recursive_filter_operator(series: Sequence[float], *, seed: float = 0.0, decay: float = 0.5) -> List[float]`
  - Apply a recursive filter to a series to capture hysteresis effects.
- `recursividad_operator(series: Sequence[float], *, seed: float = 0.0, decay: float = 0.5) -> List[float]`
  - Compatibility wrapper for :func:`recursive_filter_operator`.
- `orchestrate_delta_metrics(telemetry_segments: Sequence[Sequence[TelemetryRecord]], target_delta_nfr: float, target_sense_index: float, *, coherence_window: int = 3, recursion_decay: float = 0.4, microsectors: Sequence[SupportsMicrosector] | None = None, phase_weights: Mapping[str, Mapping[str, float] | float] | None = None, operator_state: Mapping[str, Dict[str, object]] | None = None) -> Mapping[str, object]`
  - Pipeline orchestration producing aggregated ΔNFR and Si metrics.

## Attributes
- `NodeType = TypeVar('NodeType')`
- `WHEEL_TEMPERATURE_KEYS = ('tyre_temp_fl', 'tyre_temp_fr', 'tyre_temp_rl', 'tyre_temp_rr')`

