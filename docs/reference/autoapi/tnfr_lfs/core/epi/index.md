# `tnfr_core.epi` module
EPI extraction and ΔNFR/ΔSi computations.

## Classes
### `NaturalFrequencySettings`
Configuration options for the natural frequency estimator.

#### Methods
- `resolve_vehicle_frequency(self, car_model: str | None) -> float`
- `resolve_category(self, car_model: str | None) -> str`
- `resolve_frequency_band(self, car_model: str | None) -> Tuple[float, float]`

### `NaturalFrequencySnapshot`
Result of a natural frequency analysis step.

#### Methods
- `frequency_label(self) -> str`

### `NaturalFrequencyAnalyzer`
Track dominant frequencies over a sliding telemetry window.

#### Methods
- `reset(self) -> None`
- `update(self, record: SupportsTelemetrySample, base_map: Mapping[str, float], *, car_model: str | None = None) -> NaturalFrequencySnapshot`
- `compute_from_history(self, history: Sequence[SupportsTelemetrySample], base_map: Mapping[str, float], *, record: SupportsTelemetrySample | None = None, car_model: str | None = None) -> NaturalFrequencySnapshot`

### `TelemetryRecord`
Single telemetry sample emitted by the acquisition backend.

### `EPIExtractor`
Compute EPI bundles for a stream of telemetry records.

#### Methods
- `reset(self) -> None`
  - Clear the incremental state.
- `update(self, record: SupportsTelemetrySample, *, calibration: 'CoherenceCalibrationStore' | None = None, player_name: str | None = None, car_model: str | None = None, track_name: str | None = None, tyre_compound: str | None = None) -> EPIBundle`
  - Process a single telemetry record and return the resulting bundle.
- `extract(self, records: Sequence[SupportsTelemetrySample], *, calibration: 'CoherenceCalibrationStore' | None = None, player_name: str | None = None, car_model: str | None = None) -> List[EPIBundle]`

### `DeltaCalculator`
Compute delta metrics relative to a baseline.

#### Methods
- `derive_baseline(records: Sequence[SupportsTelemetrySample], *, sample_factory: SampleFactory | None = None) -> SupportsTelemetrySample`
  - Return a synthetic baseline record representing the average state.
- `resolve_baseline(records: Sequence[SupportsTelemetrySample], *, calibration: 'CoherenceCalibrationStore' | None = None, player_name: str | None = None, car_model: str | None = None, sample_factory: SampleFactory | None = None) -> SupportsTelemetrySample`
- `compute_bundle(record: SupportsTelemetrySample, baseline: SupportsTelemetrySample, epi_value: float, *, prev_integrated_epi: Optional[float] = None, dt: float = 0.0, nu_f_by_node: Optional[Mapping[str, float]] = None, nu_f_snapshot: NaturalFrequencySnapshot | None = None, phase: str = 'entry', phase_weights: Optional[Mapping[str, Mapping[str, float] | float]] = None, phase_target_nu_f: Mapping[str, Mapping[str, float] | float] | Mapping[str, float] | float | None = None) -> EPIBundle`

## Functions
- `delta_nfr_by_node(record: SupportsTelemetrySample, *, cache_options: 'CacheOptions' | None = None) -> Mapping[str, float]`
  - Compute ΔNFR contributions for each subsystem.

The function expects ``record`` to optionally provide a ``reference``
sample, typically the baseline derived from the telemetry stint.  When a
reference is available the signal strength for every subsystem is
measured relative to it, otherwise the calculation degenerates into a
uniform distribution.
- `resolve_nu_f_by_node(record: SupportsTelemetrySample, *, phase: str | None = None, phase_weights: Mapping[str, Mapping[str, float] | float] | None = None, history: Sequence[SupportsTelemetrySample] | None = None, car_model: str | None = None, analyzer: NaturalFrequencyAnalyzer | None = None, settings: NaturalFrequencySettings | None = None, cache_options: 'CacheOptions' | None = None) -> NaturalFrequencySnapshot`
  - Return the natural frequency snapshot for a telemetry sample.

When the optional ``analyzer`` or ``history`` arguments are provided the
result incorporates sliding-window spectral analysis to align the
instantaneous ν_f values with the dominant excitation frequencies observed
in the steering, pedal and suspension signals.

## Attributes
- `NU_F_NODE_DEFAULTS = {'tyres': 0.18, 'suspension': 0.14, 'chassis': 0.12, 'brakes': 0.16, 'transmission': 0.11, 'track': 0.08, 'driver': 0.05}`
- `DEFAULT_PHASE_WEIGHTS = {'__default__': 1.0}`
- `SampleFactory = Callable[..., SupportsTelemetrySample]`
- `AXIS_FEATURE_MAP = {'tyres': {'slip_ratio': 'longitudinal', 'locking': 'longitudinal', 'slip_angle': 'lateral', 'mu_eff_front': 'both', 'mu_eff_rear': 'both'}, 'suspension': {'travel_front': 'both', 'travel_rear': 'both', 'velocity_front': 'longitudinal', 'velocity_rear': 'longitudinal', 'load_front': 'longitudinal', 'load_rear': 'longitudinal'}, 'chassis': {'yaw_rate': 'lateral', 'lateral_accel': 'lateral', 'roll': 'lateral', 'pitch': 'longitudinal'}, 'brakes': {'pressure': 'longitudinal', 'locking': 'longitudinal', 'longitudinal_decel': 'longitudinal', 'load_front': 'longitudinal'}, 'transmission': {'throttle': 'longitudinal', 'longitudinal_accel': 'longitudinal', 'slip_ratio': 'longitudinal', 'gear': 'longitudinal', 'speed': 'longitudinal'}, 'track': {'mu_eff_front': 'lateral', 'mu_eff_rear': 'lateral', 'axle_load_balance': 'lateral', 'axle_velocity_balance': 'lateral', 'yaw': 'lateral', 'vertical_load': 'longitudinal'}, 'driver': {'style_index': 'lateral', 'steer': 'lateral', 'throttle': 'longitudinal', 'yaw_rate': 'lateral'}}`

