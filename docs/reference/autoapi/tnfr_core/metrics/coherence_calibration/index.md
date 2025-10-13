# `tnfr_core.metrics.coherence_calibration` module
Coherence calibration utilities for persistent ΔNFR baselines.

## Classes
### `CalibrationMetric`
Exponentially smoothed statistic for a telemetry attribute.

#### Methods
- `update(self, value: float, decay: float) -> None`
- `norm_range(self) -> Tuple[float, float]`

### `CalibrationEntry`
Calibration data accumulated for a player/car combination.

#### Methods
- `update(self, record: TelemetryRecord, decay: float) -> None`
- `build_baseline(self) -> TelemetryRecord`
- `ranges(self) -> dict[str, Tuple[float, float]]`

### `CalibrationSnapshot`
Read-only view of a calibration entry.

### `CoherenceCalibrationStore`
Manage ΔNFR baseline calibrations for player/car combinations.

#### Methods
- `register_lap(self, player_name: str, car_model: str, records: Sequence[TelemetryRecord]) -> None`
- `observe_baseline(self, player_name: str, car_model: str, baseline: TelemetryRecord) -> None`
- `baseline_for(self, player_name: str, car_model: str, fallback: TelemetryRecord) -> TelemetryRecord`
- `snapshot(self, player_name: str, car_model: str) -> CalibrationSnapshot | None`
- `save(self) -> None`

## Attributes
- `TelemetryBaselineValue = float | int | str | None`

