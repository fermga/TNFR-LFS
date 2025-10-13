# `tnfr_core.metrics.metrics` module
Windowed telemetry metrics used by the HUD and setup planner.

## Classes
### `SlideCatchBudget`
Composite steering margin metric derived from yaw and steer activity.

### `LockingWindowScore`
Aggregated stability score around throttle locking transitions.

### `AeroAxisCoherence`
Front/rear contribution pair for a given aerodynamic axis.

#### Methods
- `imbalance(self) -> float`
  - Signed imbalance favouring the front when negative.

### `AeroBandCoherence`
Per-speed-band aerodynamic coherence split by axes.

### `AeroCoherence`
Summarises aero balance deltas split by speed bins and axes.

#### Methods
- `dominant_axis(self, tolerance: float = 0.05) -> str | None`
  - Return the dominant axle when the imbalance exceeds ``tolerance``.
- `low_speed_front(self) -> float`
- `low_speed_rear(self) -> float`
- `low_speed_imbalance(self) -> float`
- `low_speed_samples(self) -> int`
- `medium_speed_front(self) -> float`
- `medium_speed_rear(self) -> float`
- `medium_speed_imbalance(self) -> float`
- `medium_speed_samples(self) -> int`
- `high_speed_front(self) -> float`
- `high_speed_rear(self) -> float`
- `high_speed_imbalance(self) -> float`
- `high_speed_samples(self) -> int`

### `AeroBalanceDriftBin`
Aggregate rake/μ metrics for a given aerodynamic speed band.

#### Methods
- `rake_deg(self) -> float`
  - Return the mean rake expressed in degrees.

### `AeroBalanceDrift`
Aerodynamic balance drift derived from pitch, travel and μ usage.

#### Methods
- `dominant_bin(self, tolerance: float | None = None) -> tuple[str, str, AeroBalanceDriftBin] | None`
  - Return the dominant drift bin exceeding ``tolerance`` in μ Δ.

### `BrakeHeadroom`
Aggregated braking capacity metrics for the current window.

### `BumpstopHistogram`
Density and energy accumulated in the bump stop zone by axle.

### `CPHIWheelComponents`
Normalised contributions to the Contact Patch Health Index.

### `CPHIThresholds`
Traffic-light thresholds for the Contact Patch Health Index.

Values below ``red`` demand immediate intervention, measurements between
``red`` and ``amber`` indicate marginal tyre health, and readings equal
or above ``green`` describe an optimal contact patch ready for push laps.

#### Methods
- `classify(self, value: float) -> str`
- `is_optimal(self, value: float) -> bool`

### `CPHIWheel`
Contact Patch Health Index components for a single wheel.

#### Methods
- `temperature_component(self) -> float`
- `gradient_component(self) -> float`
- `mu_component(self) -> float`
- `as_dict(self, *, thresholds: CPHIThresholds | None = None) -> dict[str, object]`

### `CPHIReport` (MappingABC[str, CPHIWheel])
Aggregate CPHI values and expose shared thresholds.

The mapping behaves like ``{suffix: CPHIWheel}`` while ``as_legacy_mapping``
preserves the historical flat keys consumed by exporters and rule engines.

#### Methods
- `classification(self, value: float) -> str`
- `classification_for(self, suffix: str) -> str`
- `is_optimal(self, value: float) -> bool`
- `is_optimal_for(self, suffix: str) -> bool`
- `as_dict(self, *, include_thresholds: bool = True, include_status: bool = True) -> dict[str, object]`
- `as_legacy_mapping(self) -> dict[str, float]`

### `SuspensionVelocityBands`
Distribution of suspension velocities split by direction and band.

#### Methods
- `compression_high_speed_percentage(self) -> float`
- `rebound_high_speed_percentage(self) -> float`

### `WindowMetrics`
Aggregated metrics derived from a telemetry window.

The payload captures the usual ΔNFR gradients and phase alignment values
while also exposing support efficiency information derived from the average
vertical load and nodal ΔNFR contributions.  ``support_effective`` reflects
the structurally-weighted ΔNFR absorbed by tyres and suspension, whereas
``load_support_ratio`` normalises that magnitude against the window's mean
vertical load.  The structural expansion/contraction fields quantify how
longitudinal and lateral ΔNFR components expand (positive) or contract
(negative) the structural timeline when weighted by structural occupancy
windows. The ``bumpstop_histogram`` field captures the occupancy density and
ΔNFR energy accumulated when the suspension operates within the bump stop
envelope for each axle.  ``phase_synchrony_index`` blends the normalised
phase lag with the phase alignment cosine to produce a stable, unitless
indicator for desynchronisation events.

## Functions
- `coherence_total(samples: Sequence[tuple[float, float]] | Mapping[float, float], *, initial: float = 0.0) -> list[tuple[float, float]]`
  - Return the cumulative TNFR coherence ``C(t)`` profile.

The HUD treats coherence as accumulated evidence that driver, chassis and
aero signals are converging.  Under this interpretation the curve must be
monotonic and any destabilising impulse is clipped rather than subtracted.

Parameters
----------
samples:
    Sequence or mapping of ``(time, delta)`` contributions.  ``time`` is
    converted to ``float`` and used for sorting.
initial:
    Starting value of the cumulative signal.  Negative values are clipped to
    zero to preserve monotonicity.

Returns
-------
list[tuple[float, float]]
    Sorted ``(time, cumulative)`` pairs representing ``C(t)``.

Examples
--------
>>> coherence_total([(0.0, 0.1), (0.5, -0.2), (0.75, 0.4)])
[(0.0, 0.1), (0.5, 0.1), (0.75, 0.5)]
>>> coherence_total({0.5: 0.2, 0.0: 0.1})
[(0.0, 0.1), (0.5, 0.30000000000000004)]
- `psi_norm(values: Sequence[float], *, ord: float = 2.0) -> float`
  - Return the ℓₚ norm of a PSI vector.

In TNFR the phase synchrony index (PSI) vector captures modal coupling
between driver, chassis and tyre responses.  The Euclidean (``p=2``)
norm therefore provides a reproducible aggregate synchrony score.

Parameters
----------
values:
    Iterable containing PSI components.
ord:
    Order of the ℓₚ norm.  ``math.inf`` returns the maximum
    absolute component.

Examples
--------
>>> psi_norm([0.4, 0.3])
0.5
>>> psi_norm([0.1, -0.8], ord=math.inf)
0.8
- `psi_support(values: Sequence[float], *, threshold: float = 0.001) -> int`
  - Return the TNFR PSI support count.

Support counts how many PSI components exceed a reproducible activation
threshold (strictly greater than ``threshold``) and therefore contribute to
the synchrony budget.

Parameters
----------
values:
    Iterable containing PSI components.
threshold:
    Absolute magnitude below which PSI modes are ignored.

Examples
--------
>>> psi_support([0.0, 0.002, 0.2])
2
>>> psi_support([0.0, 1e-5, -3e-4], threshold=1e-4)
1
- `bifurcation_threshold(samples: Sequence[float], *, quantile: float = 0.9) -> float`
  - Return the TNFR bifurcation threshold τ.

The threshold highlights coherence excursions large enough to trigger a
behavioural branch (driver correction or chassis response).  Using a high
quantile keeps the estimate reproducible while still reacting to genuine
peaks.

Examples
--------
>>> bifurcation_threshold([0.2, 0.35, 0.4, 0.5])
0.47
- `mutation_threshold(samples: Sequence[float], *, quantile: float = 0.75) -> float`
  - Return the TNFR mutation threshold ξ.

Mutations represent smaller disturbances that still warrant attention.  We
capture them via a mid-to-high quantile so the estimator is sensitive to
frequently repeating perturbations.

Examples
--------
>>> mutation_threshold([0.2, 0.35, 0.4, 0.5])
0.425
- `phase_synchrony_index(lag: float, alignment: float) -> float`
  - Return a composite synchrony index combining phase lag and alignment.

The index normalises the absolute phase lag to a [0, 1] range where ``1``
represents perfect synchronisation (zero lag) and ``0`` corresponds to a
π radian mismatch.  The alignment cosine, naturally bounded in [-1, 1], is
translated to the same [0, 1] interval.  The final score favours alignment
while preserving the sensitivity to lag, providing a stable indicator for
desynchronisation alerts.
- `compute_window_metrics(records: Sequence[SupportsTelemetrySample], *, phase_indices: Sequence[int] | Mapping[str, Sequence[int]] | None = None, bundles: Sequence[SupportsEPIBundle] | None = None, fallback_to_chronological: bool = True, objectives: object | None = None) -> WindowMetrics`
  - Return averaged plan metrics for a telemetry window.

Parameters
----------
records:
    Ordered window of telemetry samples implementing
    :class:`~tnfr_core.operators.interfaces.SupportsTelemetrySample`. Entries
    must also satisfy :class:`~tnfr_core.operators.interfaces.SupportsContextRecord`
    when contextual weighting is applied.
bundles:
    Optional precomputed insight series implementing
    :class:`~tnfr_core.operators.interfaces.SupportsEPIBundle` and matching
    ``records``. Each bundle must adhere to
    :class:`~tnfr_core.operators.interfaces.SupportsContextBundle` so the node
    metrics remain accessible to the contextual helpers.
fallback_to_chronological:
    When ``True`` the metric computation gracefully falls back to the
    chronological timestamps if the structural axis is missing or
    non-monotonic.  Disabling the fallback raises a :class:`ValueError`
    whenever the structural axis cannot be resolved.
- `compute_aero_coherence(records: Sequence[SupportsTelemetrySample], bundles: Sequence[SupportsEPIBundle] | None = None, *, low_speed_threshold: float = 35.0, high_speed_threshold: float = 50.0, imbalance_tolerance: float = 0.08) -> AeroCoherence`
  - Compute aero balance deltas at low and high speed.

The helper inspects ΔNFR contributions attributed to μ_eff front/rear terms
in the :attr:`~tnfr_core.operators.interfaces.SupportsEPIBundle.delta_breakdown`
payload.
When the optional ``bundles`` sequence is not provided or lacks breakdown
data the function gracefully returns a neutral :class:`AeroCoherence`
instance with zero samples.
- `resolve_aero_mechanical_coherence(coherence_index: float, aero: AeroCoherence, *, suspension_deltas: Sequence[float] | None = None, tyre_deltas: Sequence[float] | None = None, target_delta_nfr: float = 0.0, target_mechanical_ratio: float = 0.55, target_aero_imbalance: float = 0.12, rake_velocity_profile: Sequence[tuple[float, int]] | None = None, ackermann_parallel_index: float | None = None, ackermann_parallel_samples: int | None = None) -> float`
  - Return a blended aero-mechanical coherence indicator in ``[0, 1]``.

## Attributes
- `delta_nfr_by_node = _epi.delta_nfr_by_node`

