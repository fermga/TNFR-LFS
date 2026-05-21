"""Tests for the track-map replay transport (play/pause/stop/scrub).

Covers the slot wiring on :class:`TrackMapDock` without a running event
loop: we drive the playback timer's tick manually so the test is
deterministic and offscreen-safe.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PySide6 = pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from lfs_telemetry.studio.app import create_app  # noqa: E402
from lfs_telemetry.studio.signals import SignalBus  # noqa: E402
from lfs_telemetry.studio.widgets.track_map_dock import (  # noqa: E402
    TrackMapDock,
)
from lfs_telemetry.telemetry.lap import LapTelemetry  # noqa: E402

ASSETS = Path(__file__).resolve().parents[2] / "assets"
LAP1 = ASSETS / "synthetic_BL1_FBM_lap01.csv"
LAP2 = ASSETS / "synthetic_BL1_FBM_lap02.csv"


@pytest.fixture(scope="module")
def qapp():
    return create_app([sys.argv[0]])


class _StubLoader:
    """Bare-bones stand-in for :class:`LapLoader`.

    The dock only needs ``lap_loaded``/``laps_unloaded`` signals to wire
    its slots; the test drives the slots directly via the bus, so a
    namespace exposing the two Qt signals is enough.
    """

    def __init__(self) -> None:
        from PySide6.QtCore import QObject, Signal

        class _Bus(QObject):
            lap_loaded = Signal(object, object)
            laps_unloaded = Signal(object)

        self._bus = _Bus()
        self.lap_loaded = self._bus.lap_loaded
        self.laps_unloaded = self._bus.laps_unloaded


def _make_dock(qapp):
    bus = SignalBus()
    loader = _StubLoader()
    dock = TrackMapDock(loader, bus)
    return dock, bus


@pytest.mark.skipif(not LAP1.exists(), reason="synthetic asset missing")
def test_replay_buttons_disabled_until_lap_loaded(qapp):
    dock, _bus = _make_dock(qapp)
    # No lap selected → transport is disabled.
    assert not dock._btn_play.isEnabled()
    assert not dock._scrub_slider.isEnabled()
    dock.deleteLater()


@pytest.mark.skipif(not LAP1.exists(), reason="synthetic asset missing")
def test_play_emits_cursor_and_advances(qapp):
    dock, bus = _make_dock(qapp)
    cursor_events: list[float] = []
    bus.cursor_moved.connect(lambda d: cursor_events.append(float(d)))

    lap = LapTelemetry.from_csv(LAP1)
    p = Path(LAP1)
    dock._on_laps_selected([p])
    dock._on_lap_loaded(p, lap)

    # Lap loaded → transport enabled with a non-zero duration.
    assert dock._btn_play.isEnabled()
    assert dock._anchor_duration_s() > 0.0
    assert dock._anchor_path == p

    # Press play, then drive a single timer tick deterministically.
    dock._on_play_pause()
    assert dock._is_playing()
    n_before = len(cursor_events)
    dock._on_playback_tick()
    assert len(cursor_events) > n_before
    assert dock._playback_t_s > 0.0

    # Pause → toggle stops the timer but keeps the time.
    t_paused = dock._playback_t_s
    dock._on_play_pause()
    assert not dock._is_playing()
    assert dock._playback_t_s == pytest.approx(t_paused)

    # Stop → resets to 0 and emits a cursor_left.
    left_events: list[None] = []
    bus.cursor_left.connect(lambda: left_events.append(None))
    dock._on_stop()
    assert dock._playback_t_s == 0.0
    assert left_events  # at least one cursor_left fired

    dock.deleteLater()


@pytest.mark.skipif(not LAP1.exists(), reason="synthetic asset missing")
def test_skip_buttons_clamp_to_lap_bounds(qapp):
    dock, _bus = _make_dock(qapp)
    lap = LapTelemetry.from_csv(LAP1)
    p = Path(LAP1)
    dock._on_laps_selected([p])
    dock._on_lap_loaded(p, lap)

    dur = dock._anchor_duration_s()
    assert dur > 0

    dock._on_skip_forward()
    assert dock._playback_t_s == pytest.approx(dur)
    assert not dock._is_playing()  # parks paused

    dock._on_skip_back()
    assert dock._playback_t_s == 0.0
    assert not dock._is_playing()

    dock.deleteLater()


@pytest.mark.skipif(not LAP1.exists(), reason="synthetic asset missing")
def test_speed_buttons_cycle_through_steps(qapp):
    dock, _bus = _make_dock(qapp)

    # Default 1.0× at index 2.
    assert dock._playback_speeds[dock._playback_speed_idx] == 1.0

    dock._on_faster()
    assert dock._playback_speeds[dock._playback_speed_idx] == 2.0
    dock._on_slower()
    dock._on_slower()
    assert dock._playback_speeds[dock._playback_speed_idx] == 0.5

    # Clamp at the slow end — repeated slow presses don't underflow.
    for _ in range(10):
        dock._on_slower()
    assert dock._playback_speed_idx == 0
    assert dock._playback_speeds[0] == 0.25

    # Clamp at the fast end too.
    for _ in range(20):
        dock._on_faster()
    assert dock._playback_speed_idx == len(dock._playback_speeds) - 1

    dock.deleteLater()


@pytest.mark.skipif(not LAP1.exists(), reason="synthetic asset missing")
def test_loop_wraps_at_end_of_lap(qapp):
    dock, _bus = _make_dock(qapp)
    lap = LapTelemetry.from_csv(LAP1)
    p = Path(LAP1)
    dock._on_laps_selected([p])
    dock._on_lap_loaded(p, lap)

    dur = dock._anchor_duration_s()
    dock._loop_check.setChecked(True)
    assert dock._playback_loop is True

    # Park near the end then start: a single tick must wrap, not pause.
    dock._playback_t_s = dur - 1e-4
    dock._on_play_pause()
    assert dock._is_playing()
    dock._on_playback_tick()
    assert dock._is_playing()
    assert dock._playback_t_s < dur

    dock._on_stop()
    dock.deleteLater()


@pytest.mark.skipif(
    not (LAP1.exists() and LAP2.exists()),
    reason="synthetic assets missing",
)
def test_multi_lap_creates_ghost_dots(qapp):
    dock, _bus = _make_dock(qapp)
    lap_a = LapTelemetry.from_csv(LAP1)
    lap_b = LapTelemetry.from_csv(LAP2)
    pa, pb = Path(LAP1), Path(LAP2)

    dock._on_laps_selected([pa, pb])
    dock._on_lap_loaded(pa, lap_a)
    dock._on_lap_loaded(pb, lap_b)

    # Anchor is the first selected; the second lap gets a ghost dot.
    assert dock._anchor_path == pa
    dock._playback_t_s = dock._anchor_duration_s() * 0.25
    dock._update_ghost_dots()
    assert pb in dock._ghost_dots
    assert pa not in dock._ghost_dots

    # Stop hides every ghost.
    dock._on_stop()
    assert not dock._ghost_dots

    dock.deleteLater()


@pytest.mark.skipif(not LAP1.exists(), reason="synthetic asset missing")
def test_scrub_seeks_and_emits_cursor(qapp):
    dock, bus = _make_dock(qapp)
    cursor_events: list[float] = []
    bus.cursor_moved.connect(lambda d: cursor_events.append(float(d)))

    lap = LapTelemetry.from_csv(LAP1)
    p = Path(LAP1)
    dock._on_laps_selected([p])
    dock._on_lap_loaded(p, lap)

    dur = dock._anchor_duration_s()
    midpoint = dock._scrub_slider.maximum() // 2
    dock._scrub_slider.setValue(midpoint)
    assert dock._playback_t_s == pytest.approx(dur * 0.5, rel=0.05)
    assert cursor_events  # seek pushed the cursor

    dock.deleteLater()
