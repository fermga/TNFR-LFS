"""Auto-split from live_modules.py — MH1."""
from __future__ import annotations

from PySide6.QtGui import (
    QColor,
)

from ._base import (
    _LabeledValueWindow,
)


class PositionWindow(_LabeledValueWindow):
    LABEL = "POS"
    DEFAULT_SIZE = (110, 90)

    def _value_text(self) -> str:
        pos = self._snap.get("view_position")
        return f"P{pos}" if pos else "--"


class FuelPctWindow(_LabeledValueWindow):
    MODULE_ID = "fuel_pct"
    LABEL = "FUEL"
    DEFAULT_SIZE = (140, 80)

    def _value_text(self) -> str:
        f = self._snap.get("view_fuel_pct")
        return f"{f:5.1f}%" if f is not None else "--"

    def _value_color(self) -> QColor:
        f = self._snap.get("view_fuel_pct")
        if f is None:
            return QColor(220, 220, 230)
        if f < 5:
            return QColor(255, 80, 80)
        if f < 15:
            return QColor(255, 220, 60)
        return QColor(220, 220, 230)


class FuelLapsRemainingWindow(_LabeledValueWindow):
    MODULE_ID = "fuel_laps"
    LABEL = "LAPS LEFT"
    DEFAULT_SIZE = (170, 80)

    def _value_text(self) -> str:
        n = self._snap.get("fuel_laps_remaining")
        return f"{n:4.1f}" if n is not None else "--"

    def _value_color(self) -> QColor:
        n = self._snap.get("fuel_laps_remaining")
        if n is None:
            return QColor(220, 220, 230)
        if n < 1.0:
            return QColor(255, 80, 80)
        if n < 2.0:
            return QColor(255, 220, 60)
        return QColor(220, 220, 230)


class SpeedWindow(_LabeledValueWindow):
    MODULE_ID = "speed"
    LABEL = "SPEED"
    DEFAULT_SIZE = (160, 80)

    def _value_text(self) -> str:
        v = self._snap.get("view_speed_kmh")
        return f"{v:5.1f}" if v is not None else "--"


