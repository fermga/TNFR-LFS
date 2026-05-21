"""T2 — Smoke tests for :class:`TrackMapDock` covering construction,
overlay toggling, and opacity slot wiring.

The deeper replay-transport tests live in ``test_track_replay.py``;
this file complements them with breadth across the dock's
non-playback controls so a regression in overlay/layer wiring is
caught even when no lap is loaded.
"""

from __future__ import annotations

import os
import sys

import pytest

PySide6 = pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from lfs_telemetry.studio.app import create_app  # noqa: E402
from lfs_telemetry.studio.signals import SignalBus  # noqa: E402
from lfs_telemetry.studio.widgets.track_map_dock import (  # noqa: E402
    TrackMapDock,
)


@pytest.fixture(scope="module")
def qapp():
    return create_app([sys.argv[0]])


class _StubLoader:
    def __init__(self) -> None:
        from PySide6.QtCore import QObject, Signal

        class _Bus(QObject):
            lap_loaded = Signal(object, object)
            laps_unloaded = Signal(object)

        self._bus = _Bus()
        self.lap_loaded = self._bus.lap_loaded
        self.laps_unloaded = self._bus.laps_unloaded


def _make_dock(qapp) -> TrackMapDock:
    return TrackMapDock(_StubLoader(), SignalBus())


def test_dock_constructs_empty(qapp):
    dock = _make_dock(qapp)
    try:
        assert dock._axis_kind == "distance"
        assert dock._loaded_laps == {}
        assert dock._anchor_map is None
        assert dock._overlay_visible is True
        assert 0.0 <= dock._overlay_opacity <= 1.0
    finally:
        dock.deleteLater()


def test_overlay_toggle_updates_state(qapp):
    dock = _make_dock(qapp)
    try:
        dock._overlay_check.setChecked(False)
        assert dock._overlay_visible is False
        dock._overlay_check.setChecked(True)
        assert dock._overlay_visible is True
    finally:
        dock.deleteLater()


def test_overlay_opacity_slot_clamps_and_stores(qapp):
    dock = _make_dock(qapp)
    try:
        dock._on_overlay_opacity(0)
        assert dock._overlay_opacity == pytest.approx(0.0)
        dock._on_overlay_opacity(100)
        assert dock._overlay_opacity == pytest.approx(1.0)
        dock._on_overlay_opacity(35)
        assert dock._overlay_opacity == pytest.approx(0.35)
        # Out-of-range values must still clamp to [0, 1].
        dock._on_overlay_opacity(-50)
        assert dock._overlay_opacity == pytest.approx(0.0)
        dock._on_overlay_opacity(250)
        assert dock._overlay_opacity == pytest.approx(1.0)
    finally:
        dock.deleteLater()


def test_dock_show_does_not_crash(qapp):
    dock = _make_dock(qapp)
    try:
        dock.resize(640, 480)
        dock.show()
        qapp.processEvents()
        dock.hide()
    finally:
        dock.deleteLater()
