"""Brake temperature proxy integrating heating and cooling dynamics.

Live for Speed (LFS) does not transmit brake temperatures through OutGauge
telemetry by default, so most sessions rely on the TNFR proxy to estimate the
thermal state. Whenever the simulator provides a valid temperature reading, the
estimator simply adopts the observation; otherwise it integrates the configured
model to provide a physically plausible approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import math
from typing import Mapping, Sequence, Tuple

_G = 9.80665
_MAX_TEMP = 1200.0


@dataclass(frozen=True)
class BrakeThermalConfig:
    """Configuration parameters for the brake thermal proxy.

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
    """

    ambient: float = 25.0
    heat_capacity: float = 18_000.0
    heating_efficiency: float = 0.35
    convective_coefficient: float = 0.06
    still_air_coefficient: float = 0.012
    minimum_brake_input: float = 0.02

    def clamp(self, value: float) -> float:
        """Clamp ``value`` within the valid temperature range."""

        minimum = float(self.ambient)
        return max(minimum, min(_MAX_TEMP, value))


@dataclass
class BrakeThermalEstimator:
    """Stateful proxy that integrates brake temperatures per wheel."""

    config: BrakeThermalConfig = field(default_factory=BrakeThermalConfig)
    _temperatures: list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:  # pragma: no cover - exercised indirectly
        self.reset()

    def reset(self) -> None:
        """Reset the estimator state to the ambient temperature."""

        self._temperatures = [self.config.ambient] * 4

    @property
    def temperatures(self) -> Tuple[float, float, float, float]:
        """Return the current temperature state."""

        return tuple(self._temperatures)

    def observe(self, readings: Sequence[float]) -> None:
        """Override the state using direct measurements."""

        values: list[float] = []
        for value in readings:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = math.nan
            if not math.isfinite(numeric):
                numeric = self.config.ambient
            values.append(self.config.clamp(numeric))
        if len(values) < 4:
            values.extend([self.config.ambient] * (4 - len(values)))
        self._temperatures = values[:4]

    def seed(self, readings: Sequence[float]) -> None:
        """Alias for :meth:`observe` for backwards compatibility."""

        self.observe(readings)

    def step(
        self,
        dt: float,
        speed: float,
        deceleration: float,
        brake_input: float,
        wheel_loads: Sequence[float],
    ) -> Tuple[float, float, float, float]:
        """Advance the estimator state returning the updated temperatures."""

        if not math.isfinite(dt) or dt <= 0.0:
            return self.temperatures

        clamped_dt = float(dt)
        speed_value = max(0.0, float(speed))
        braking = max(0.0, float(deceleration))
        brake_level = max(0.0, float(brake_input))

        finite_loads = [
            max(0.0, float(load))
            for load in wheel_loads[:4]
            if math.isfinite(load) and float(load) > 0.0
        ]
        fallback_load = sum(finite_loads) / len(finite_loads) if finite_loads else 0.0

        updated: list[float] = []
        convective_base = self.config.convective_coefficient
        still_coeff = self.config.still_air_coefficient

        for index in range(4):
            current = float(self._temperatures[index])
            load = 0.0
            if index < len(wheel_loads):
                candidate = wheel_loads[index]
                if math.isfinite(candidate) and float(candidate) > 0.0:
                    load = float(candidate)
            if load <= 0.0:
                load = fallback_load

            mass = load / _G if load > 0.0 else 0.0
            temperature = current

            if (
                brake_level >= self.config.minimum_brake_input
                and braking > 0.0
                and mass > 0.0
                and speed_value > 0.0
            ):
                power = mass * braking * speed_value
                energy = (
                    power * clamped_dt * max(0.0, self.config.heating_efficiency)
                )
                if energy > 0.0 and self.config.heat_capacity > 1e-6:
                    delta = energy / self.config.heat_capacity
                    temperature += delta

            excess = temperature - self.config.ambient
            if excess > 0.0:
                cooling_rate = still_coeff + convective_base * math.sqrt(speed_value)
                cooling_rate = max(0.0, cooling_rate)
                cooled = min(excess, excess * cooling_rate * clamped_dt)
                temperature -= cooled

            updated.append(self.config.clamp(temperature))

        self._temperatures = updated
        return self.temperatures


def merge_brake_config(
    base: BrakeThermalConfig, overrides: Mapping[str, object]
) -> BrakeThermalConfig:
    """Return ``base`` updated with numeric overrides."""

    updates: dict[str, float] = {}
    for field in fields(BrakeThermalConfig):
        key = field.name
        if key not in overrides:
            continue
        try:
            numeric = float(overrides[key])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        updates[key] = float(numeric)
    if updates:
        return replace(base, **updates)
    return base
