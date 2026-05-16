"""Native Qt-based telemetry studio (MoTeC i2 / AIM RaceStudio class).

This package replaces the Dash + Plotly viewer with a desktop GUI built on
PySide6 + pyqtgraph. Backend reuse is total: every loader, model and
analysis lives in :mod:`lfs_telemetry.telemetry` and is consumed unchanged.

Public entry point::

    python -m lfs_telemetry.studio C:\\path\\to\\captures

or, after install, ``lfs-telemetry-studio C:\\path\\to\\captures``.

Architecture
------------

* :mod:`lfs_telemetry.studio.app`          — :class:`QApplication` factory + theme.
* :mod:`lfs_telemetry.studio.main_window`  — :class:`QMainWindow` with dock layout.
* :mod:`lfs_telemetry.studio.signals`      — process-wide :class:`SignalBus` (Qt signals).
* :mod:`lfs_telemetry.studio.models`       — Qt models wrapping the telemetry layer.
* :mod:`lfs_telemetry.studio.widgets`      — dockable panels (captures, channels, charts).
* :mod:`lfs_telemetry.studio.charts`       — pyqtgraph chart widgets + LTTB decimation.

Why a new package and not edits in :mod:`lfs_telemetry.app`?
The Dash/Plotly app stays untouched while the Qt slice matures. Once the
Studio reaches feature parity the Dash app is removed in a single commit.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.2.0"
