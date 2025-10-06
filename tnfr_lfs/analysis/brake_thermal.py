"""Brake thermal stress proxy estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import math

__all__ = ["BrakeThermalConfig", "BrakeThermalEstimator"]

_G = 9.80665


@dataclass
class BrakeThermalConfig:
    """Configuration values for :class:`BrakeThermalEstimator`."""

    ambient_C: float = 20.0
    eta_in: float = 0.85
    thermal_mass_J_per_K: float = 22000.0
    k_cool: float = 0.035
    k_speed: float = 0.015
    min_speed_cooling: float = 1.0
    smoothing_alpha: float = 0.10


class BrakeThermalEstimator:
    """Estimate brake temperature proxy via a simple energy balance model."""

    def __init__(self, cfg: BrakeThermalConfig | None = None) -> None:
        self.cfg = cfg or BrakeThermalConfig()
        self._temperatures: list[float] = [self.cfg.ambient_C] * 4

    def reset(self) -> None:
        """Reset the estimator to ambient temperature."""

        self._temperatures = [self.cfg.ambient_C] * 4

    def update(
        self,
        dt: float,
        speed_mps: float,
        decel_pos: float,
        brake_input: float,
        wheel_loads_N: Iterable[float],
    ) -> Tuple[float, float, float, float]:
        """Advance the estimator by ``dt`` seconds.

        Parameters
        ----------
        dt:
            Time step in seconds.
        speed_mps:
            Vehicle speed in metres per second.
        decel_pos:
            Positive longitudinal deceleration in metres per second squared.
        brake_input:
            Normalised brake pedal input ``[0, 1]``.
        wheel_loads_N:
            Sequence of four wheel loads in Newtons (front-left to rear-right).
        """

        if dt <= 0.0:
            return tuple(self._temperatures)

        cfg = self.cfg
        # Ensure we have exactly four loads and clamp negatives.
        loads = [max(0.0, float(load)) for load in wheel_loads_N][:4]
        if len(loads) < 4:
            loads.extend([0.0] * (4 - len(loads)))

        load_sum = sum(loads)
        mass = load_sum / _G if load_sum > 1.0 else 0.0

        v = max(0.0, float(speed_mps))
        a = max(0.0, float(decel_pos))
        brake = max(0.0, min(1.0, float(brake_input)))

        power_total = mass * a * v * brake

        if load_sum > 1.0:
            shares = [load / load_sum for load in loads]
        else:
            shares = [0.25, 0.25, 0.25, 0.25]

        for idx in range(4):
            q_in = cfg.eta_in * power_total * shares[idx]

            speed_term = max(v, cfg.min_speed_cooling)
            cool_gain = cfg.k_cool * (1.0 + cfg.k_speed * speed_term)
            q_out = cool_gain * max(0.0, self._temperatures[idx] - cfg.ambient_C)

            dT = (q_in - q_out) * dt / max(cfg.thermal_mass_J_per_K, 1e-6)
            T_next = self._temperatures[idx] + dT
            alpha = max(0.0, min(1.0, cfg.smoothing_alpha))
            T_next = (1.0 - alpha) * T_next + alpha * self._temperatures[idx]
            T_next = min(1200.0, max(cfg.ambient_C, T_next))
            self._temperatures[idx] = T_next

        return tuple(float(value) for value in self._temperatures)

