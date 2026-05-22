"""Process-wide signal bus.

A single :class:`SignalBus` instance is shared by every dock so panels
stay decoupled: the captures dock emits ``laps_selected``, the channels
dock emits ``channels_changed``, the chart dock listens to both. The
crosshair cursor on any chart broadcasts ``cursor_moved`` so all sibling
charts (and a future track-map dock) follow it in lockstep.

Why a singleton-via-attribute instead of qApp dependency injection?
Subclassing :class:`QApplication` works but breaks isolated widget tests;
a plain :class:`QObject` you instantiate once in :func:`run` is trivial
to mock in unit tests.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal


class SignalBus(QObject):
    """Cross-dock signal hub. Created once per :class:`QApplication`."""

    # ----- Workspace / capture catalog -----------------------------
    workspace_changed = Signal(Path)
    """Workspace directory was rescanned. Argument: new root path."""

    captures_refreshed = Signal()
    """The captures table was repopulated (no payload)."""

    capture_lap_streamed = Signal()
    """A live capture just wrote a new per-lap CSV to disk (streaming
    per-lap mode). Listeners (the captures dock) should rescan the
    workspace so the new file appears without manual F5. No payload."""

    # ----- Lap selection -------------------------------------------
    laps_selected = Signal(list)
    """User selected one or more laps. Argument: list[Path]."""

    # ----- Channel selection ---------------------------------------
    channels_changed = Signal(list)
    """User toggled channel checkboxes. Argument: list[str] (column names)."""

    channels_requested = Signal(list)
    """A sibling dock requests a specific channel selection (e.g. the
    charts dock applying a canonical overlay preset). The channels dock
    listens and ticks the requested columns, which in turn re-emits
    ``channels_changed`` through the normal path. Argument:
    list[str] (column names)."""

    available_columns_changed = Signal(list)
    """Columns present on the currently-loaded reference lap. Argument: list[str]."""

    # ----- X-axis kind ---------------------------------------------
    x_axis_changed = Signal(str)
    """X-axis radio toggled. Argument: 'distance' or 'time'."""

    # ----- Cursor sync ---------------------------------------------
    cursor_moved = Signal(float)
    """A chart's crosshair was moved. Argument: x in current axis units."""

    cursor_left = Signal()
    """The pointer left every chart; siblings should hide their crosshairs."""

    # ----- Setup garage editor ------------------------------------
    setup_overrides_changed = Signal(str, object)
    """User edited the in-app garage form for a car.

    Arguments:
        car_key: 4-letter car short-name the overrides apply to
            (e.g. ``"FBM"``). Empty string means "no car selected".
        bin: A :class:`~telemetry.car_info_bin.CarInfoBin` with the
            user's overrides merged on top of the loaded baseline.
            ``None`` means "fall back to the on-disk baseline".
    """

    # ----- Status -------------------------------------------------
    status_message = Signal(str, int)
    """Show ``msg`` in the status bar for ``timeout_ms`` (0 = persistent)."""


__all__ = ["SignalBus"]
