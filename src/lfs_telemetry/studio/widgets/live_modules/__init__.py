"""Independent overlay modules driven by :class:`LiveDataSource`.

Every datum the live snapshot carries is exposed as its own toggleable,
draggable, **resizable** frameless window. This package re-exports every
public window class so legacy imports ``from
lfs_telemetry.studio.widgets.live_modules import XxxWindow`` keep
working after the MH1 split.
"""
from __future__ import annotations

from ._base import (
    MIN_H,
    MIN_W,
    RESIZE_GRIP_PX,
    _fmt_clock,
    _fmt_delta,
    _fmt_gap,
    _LabeledValueWindow,
    _LiveModuleWindow,
    proximity_color,
)
from .delta_bar import DeltaBarWindow, SpeedDeltaBarWindow
from .diagnostics import (
    FlagsWindow,
    GMeterWindow,
    PitLimiterWindow,
)
from .gaps import GapAheadWindow, GapBehindWindow, _GapWindow
from .inputs import (
    BrakeWindow,
    ClutchWindow,
    GearWindow,
    RpmWindow,
    ThrottleWindow,
    _PedalWindow,
)
from .radar import RadarWindow
from .session import SessionInfoWindow
from .simple import (
    FuelLapsRemainingWindow,
    FuelPctWindow,
    PositionWindow,
    SpeedWindow,
)
from .tyre_risk import TyreRiskWindow

__all__ = [
    "MIN_H",
    "MIN_W",
    "RESIZE_GRIP_PX",
    "BrakeWindow",
    "ClutchWindow",
    "DeltaBarWindow",
    "FlagsWindow",
    "FuelLapsRemainingWindow",
    "FuelPctWindow",
    "GMeterWindow",
    "GapAheadWindow",
    "GapBehindWindow",
    "GearWindow",
    "PitLimiterWindow",
    "PositionWindow",
    "RadarWindow",
    "RpmWindow",
    "SessionInfoWindow",
    "SpeedDeltaBarWindow",
    "SpeedWindow",
    "ThrottleWindow",
    "TyreRiskWindow",
    "_GapWindow",
    "_LabeledValueWindow",
    "_LiveModuleWindow",
    "_PedalWindow",
    "_fmt_clock",
    "_fmt_delta",
    "_fmt_gap",
    "proximity_color",
]
