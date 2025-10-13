# `tnfr_lfs.telemetry.offline.profiles` module
Persistence helpers for car/track recommendation profiles.

## Classes
### `ProfileObjectives`
Target objectives tracked for a car/track profile.

### `ProfileTolerances`
Phase tolerances persisted alongside a profile.

### `StintMetrics`
Aggregated ΔNFR/SI metrics registered for a stint.

### `AeroProfile`
Persistent aero balance targets for specific strategies.

### `ProfileRecord`
Mutable representation of a persistent profile entry.

#### Methods
- `from_base(cls, car_model: str, track_name: str, base_profile: ThresholdProfile, *, objectives: ProfileObjectives | None = None, tolerances: ProfileTolerances | None = None, weights: Mapping[str, Mapping[str, float]] | None = None, aero_profiles: Mapping[str, AeroProfile] | Mapping[str, Mapping[str, float]] | None = None) -> 'ProfileRecord'`
- `to_threshold_profile(self, base_profile: ThresholdProfile) -> ThresholdProfile`
- `apply_mutation(self, phases: Mapping[str, float], reference: StintMetrics, outcome: StintMetrics) -> None`
- `update_jacobian_history(self, jacobian: Mapping[str, Mapping[str, float]] | None, phase_jacobian: Mapping[str, Mapping[str, Mapping[str, float]]] | None, *, smoothing: float = GRADIENT_SMOOTHING) -> None`

### `ProfileSnapshot`
Resolved profile state exposed to consumers.

### `ProfileManager`
Handle persistence of car/track recommendation profiles.

#### Methods
- `resolve(self, car_model: str, track_name: str, base_profile: ThresholdProfile | None = None, *, session: Mapping[str, Any] | None = None) -> ProfileSnapshot`
- `register_plan(self, car_model: str, track_name: str, phases: Mapping[str, float], baseline_metrics: Tuple[float, float] | None = None, *, jacobian: Mapping[str, Mapping[str, float]] | None = None, phase_jacobian: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None) -> None`
- `register_result(self, car_model: str, track_name: str, sense_index: float, delta_nfr: float) -> None`
- `update_tyre_offsets(self, car_model: str, track_name: str, offsets: Mapping[str, float]) -> None`
- `update_aero_profile(self, car_model: str, track_name: str, mode: str, *, low_speed_target: float | None = None, high_speed_target: float | None = None) -> None`
- `gradient_history(self, car_model: str, track_name: str) -> tuple[Mapping[str, Mapping[str, float]], Mapping[str, Mapping[str, Mapping[str, float]]]]`
- `update_objectives(self, car_model: str, track_name: str, target_delta_nfr: float, target_sense_index: float) -> None`
- `save(self) -> None`

## Attributes
- `GRADIENT_SMOOTHING = 0.35`

