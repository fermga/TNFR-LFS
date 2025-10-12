# `tnfr_lfs.analysis.brake_thermal` module
Brake temperature proxy integrating heating and cooling dynamics.

Live for Speed (LFS) does not transmit brake temperatures through OutGauge
telemetry by default, so most sessions rely on the TNFR proxy to estimate the
thermal state. Whenever the simulator provides a valid temperature reading, the
estimator simply adopts the observation; otherwise it integrates the configured
model to provide a physically plausible approximation.

## Classes
### `BrakeThermalConfig`
Configuration parameters for the brake thermal proxy.

The defaults are tuned to reproduce plausible brake temperatures when the
simulator does not expose native telemetry, which is the common Live for
Speed scenario.

Parameters
----------
ambient:
    Ambient temperature in Celsius used as the lower bound for the model and
    the fallback value when no measurement is available (the typical
    OutGauge situation). It represents the thermal sink that the estimator
    converges to when brakes are idle for long periods.
heat_capacity:
    Effective heat capacity of the assembly per wheel (J/K). Larger values
    slow down the rate of temperature change, capturing the tendency of
    heavier brake packages to retain heat between braking zones.
heating_efficiency:
    Fraction of the mechanical work converted into brake heat. Values in
    ``[0, 1]`` scale the ``m·a·v`` input and are calibrated so that the
    resulting temperature curve matches typical LFS braking events when no
    telemetry override is available.
convective_coefficient:
    Gain applied to the convective term. Higher values increase the
    influence of airflow, which scales with ``sqrt(speed)``. This mimics the
    cooling observed on straights as the car gains speed.
still_air_coefficient:
    Cooling gain applied when the vehicle is stationary. This prevents the
    system from staying indefinitely above the ambient temperature when no
    airflow is present and approximates the residual dissipation through the
    hub and surrounding structure.
minimum_brake_input:
    Threshold below which pedal input is ignored for heating purposes to
    reduce spurious spikes caused by brake bias adjustments.

#### Methods
- `clamp(self, value: float) -> float`
  - Clamp ``value`` within the valid temperature range.

### `BrakeThermalEstimator`
Stateful proxy that integrates brake temperatures per wheel.

#### Methods
- `reset(self) -> None`
  - Reset the estimator state to the ambient temperature.
- `temperatures(self) -> Tuple[float, float, float, float]`
  - Return the current temperature state.
- `observe(self, readings: Sequence[float]) -> None`
  - Override the state using direct measurements.
- `seed(self, readings: Sequence[float]) -> None`
  - Alias for :meth:`observe` for backwards compatibility.
- `step(self, dt: float, speed: float, deceleration: float, brake_input: float, wheel_loads: Sequence[float]) -> Tuple[float, float, float, float]`
  - Advance the estimator state returning the updated temperatures.

## Functions
- `merge_brake_config(base: BrakeThermalConfig, overrides: Mapping[str, object]) -> BrakeThermalConfig`
  - Return ``base`` updated with numeric overrides.

