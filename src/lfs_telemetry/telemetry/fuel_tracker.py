"""Live fuel-consumption tracker and remaining-laps estimator.

Pure logic helper. Feed fuel% at every IS_LAP completion via
:meth:`observe_lap`. The tracker keeps a small ring of the last few
*per-lap* fuel deltas and exposes:

* :attr:`avg_burn_pct_per_lap` — mean fuel burned per lap over the
  last ``window`` laps (None until at least one lap has been seen);
* :meth:`laps_remaining` — given the current fuel%, the integer number
  of *whole* laps that should still be possible at the current burn
  rate (returns ``None`` if no estimate yet).

Used by the Live overlay's "fuel laps" module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class FuelTracker:
    window: int = 3
    _last_fuel_pct: float | None = None
    _per_lap_burn_pct: deque[float] = field(default_factory=deque)

    def observe_fuel(self, fuel_pct: float | None) -> None:
        """Record the most recent fuel% reading (any tick, not only laps)."""
        if fuel_pct is None:
            return
        if self._last_fuel_pct is None:
            self._last_fuel_pct = float(fuel_pct)

    def observe_lap(self, fuel_pct: float | None) -> None:
        """Close a lap: compute burn since previous lap and store it."""
        if fuel_pct is None:
            return
        if self._last_fuel_pct is not None:
            burn = self._last_fuel_pct - float(fuel_pct)
            # Refuel or sensor jitter → ignore (positive only).
            if burn > 0:
                self._per_lap_burn_pct.append(burn)
                while len(self._per_lap_burn_pct) > self.window:
                    self._per_lap_burn_pct.popleft()
        self._last_fuel_pct = float(fuel_pct)

    @property
    def avg_burn_pct_per_lap(self) -> float | None:
        if not self._per_lap_burn_pct:
            return None
        return sum(self._per_lap_burn_pct) / len(self._per_lap_burn_pct)

    def laps_remaining(self, fuel_pct: float | None) -> float | None:
        """Float estimate of laps left at the current burn rate."""
        burn = self.avg_burn_pct_per_lap
        if burn is None or burn <= 0 or fuel_pct is None:
            return None
        return float(fuel_pct) / burn


__all__ = ["FuelTracker"]
