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

    # ----- Lap selection -------------------------------------------
    laps_selected = Signal(list)
    """User selected one or more laps. Argument: list[Path]."""

    # ----- Channel selection ---------------------------------------
    channels_changed = Signal(list)
    """User toggled channel checkboxes. Argument: list[str] (column names)."""

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

    # ----- Status -------------------------------------------------
    status_message = Signal(str, int)
    """Show ``msg`` in the status bar for ``timeout_ms`` (0 = persistent)."""


__all__ = ["SignalBus"]
