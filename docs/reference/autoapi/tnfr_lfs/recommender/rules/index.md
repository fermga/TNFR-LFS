# `tnfr_lfs.recommender.rules` module
Rule-based recommendation engine.

## Classes
### `PhaseActionTemplate`
Template that translates gradients into actionable setup deltas.

### `Recommendation`
Represents an actionable recommendation.

### `PhaseTargetWindow`
Slip and yaw windows associated to a ΔNFR target.

### `ThresholdProfile`
Thresholds tuned per car model and circuit.

#### Methods
- `tolerance_for_phase(self, phase: str) -> float`
- `target_for_phase(self, phase: str) -> PhaseTargetWindow | None`
- `weights_for_phase(self, phase: str) -> Mapping[str, float] | float`
- `archetype_targets_for(self, archetype: str) -> Mapping[str, PhaseArchetypeTargets]`
- `hud_threshold(self, key: str, default: float | None = None) -> float | None`
  - Return a HUD threshold override or fall back to the supplied default.

Callers are expected to provide the baseline HUD defaults (see
:func:`tnfr_lfs.cli.osd._hud_threshold_value`), so leaving a value
undefined guarantees we inherit the generic thresholds instead of
repeating them for every track.

### `RuleProfileObjectives`
Minimal snapshot of profile objectives for rule evaluation.

### `RuleContext`
Context shared with the rules to build rationales.

#### Methods
- `profile_label(self) -> str`

### `RecommendationRule` (Protocol)
Interface implemented by recommendation rules.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `LoadBalanceRule`
Suggests changes when ΔNFR deviates from the baseline.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `BottomingPriorityRule`
Bias ride height vs bump adjustments using bottoming ratios.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `SuspensionVelocityRule`
Detect damper packing and axle asymmetries using velocity histograms.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `BrakeHeadroomRule`
Adjust the maximum brake force when significant headroom mismatches arise.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `StabilityIndexRule`
Issue recommendations when the sense index degrades.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `AeroCoherenceRule`
React to aero imbalance at high speed when low speed remains stable.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `FrontWingBalanceRule`
Recommend front wing adjustments when high speed balance is front-limited.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `CoherenceRule`
High-level rule that considers the average sense index across a stint.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `PhaseDeltaDeviationRule`
Detects ΔNFR mismatches for a given phase of the corner.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `PhaseNodeOperatorRule`
Reinforce operator actions using dominant nodes and ν_f objectives.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `ParallelSteerRule`
React to Ackermann steering deviations using the aggregated index.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `TyreBalanceRule`
Recommend ΔP and camber tweaks from CPHI telemetry.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `FootprintEfficiencyRule`
Reduce ΔNFR guidance when the tyre footprint is saturated.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `CurbComplianceRule`
Analyses support events (pianos) against the ΔNFR target.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `DetuneRatioRule`
Escalate bar/damper guidance when detune ratio collapses under load.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `ShiftStabilityRule`
Escalate gearing guidance when apex→exit shifts destabilise the exit.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `LockingWindowRule`
Tighten differential guidance based on locking transition stability.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `UsefulDissonanceRule`
Adjust axle balance when the Useful Dissonance Ratio drifts.

#### Methods
- `evaluate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, context: RuleContext | None = None) -> Iterable[Recommendation]`

### `RecommendationEngine`
Aggregate a list of rules and produce recommendations.

#### Methods
- `register_plan(self, recommendations: Sequence[Recommendation], *, car_model: str | None = None, track_name: str | None = None, baseline_sense_index: float | None = None, baseline_delta_nfr: float | None = None, jacobian: Mapping[str, Mapping[str, float]] | None = None, phase_jacobian: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None) -> None`
- `register_stint_result(self, *, sense_index: float, delta_nfr: float, car_model: str | None = None, track_name: str | None = None) -> None`
- `generate(self, results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None, *, car_model: str | None = None, track_name: str | None = None) -> List[Recommendation]`

## Functions
- `lookup_threshold_profile(car_model: str, track_name: str, library: Mapping[str, Mapping[str, ThresholdProfile]] | None = None) -> ThresholdProfile`
  - Resolve the closest matching threshold profile for a car/track.

## Attributes
- `MANUAL_REFERENCES = {'braking': 'Basic Setup Guide · Optimal Braking [BAS-FRE]', 'antiroll': 'Advanced Setup Guide · Anti-Roll Bars [ADV-ARB]', 'differential': 'Advanced Setup Guide · Differential Configuration [ADV-DIF]', 'curbs': 'Basic Setup Guide · Kerb Usage [BAS-CUR]', 'ride_height': 'Advanced Setup Guide · Ride Heights & Load Distribution [ADV-RDH]', 'aero': 'Basic Setup Guide · Aero Balance [BAS-AER]', 'driver': 'Basic Setup Guide · Consistent Driving [BAS-DRV]', 'tyre_balance': 'Advanced Setup Guide · Pressures & Camber [ADV-TYR]', 'dampers': 'Advanced Setup Guide · Dampers [ADV-DMP]', 'springs': 'Advanced Setup Guide · Spring Stiffness [ADV-SPR]'}`
- `NODE_LABELS = {'tyres': 'Tyres', 'suspension': 'Suspension', 'chassis': 'Chassis', 'brakes': 'Brakes', 'transmission': 'Transmission', 'track': 'Track', 'driver': 'Driver'}`
- `DEFAULT_THRESHOLD_PROFILE = ThresholdProfile(entry_delta_tolerance=1.5, apex_delta_tolerance=1.0, exit_delta_tolerance=2.0, piano_delta_tolerance=2.5, rho_detune_threshold=0.7, phase_targets=_BASELINE_PHASE_TARGETS, phase_weights=_BASELINE_PHASE_WEIGHTS, archetype_phase_targets=_BASELINE_ARCHETYPE_TARGETS, robustness=_BASELINE_ROBUSTNESS_THRESHOLDS)`

