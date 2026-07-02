"""T1 — Broad smoke tests for live overlay window classes.

Instantiates every public overlay window in
``lfs_telemetry.studio.widgets.live_modules`` against a populated
``LiveDataSource`` snapshot, calls ``render_to_image()`` once, and
verifies no exception escapes and the resulting image is non-null.

The deeper paint-path test for the three core windows
(Speed/Gear/Rpm) lives in ``test_live_modules_offscreen.py``; this
file complements it with breadth across all overlays so any class
that fails to construct or paint is caught early.
"""

from __future__ import annotations

import os
import sys

import pytest

PySide6 = pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage  # noqa: E402

from lfs_telemetry.studio.app import create_app  # noqa: E402
from lfs_telemetry.studio.widgets import live_modules as lm  # noqa: E402
from lfs_telemetry.studio.widgets.live_data_source import (  # noqa: E402
    LiveDataSource,
)


@pytest.fixture(scope="module")
def qapp():
    return create_app([sys.argv[0]])


@pytest.fixture()
def source(qapp):
    src = LiveDataSource()
    src.snapshot.update({
        "speed_kmh": 142.7,
        "gear": 4,
        "rpm": 7250,
        "rpm_max": 9500,
        "throttle": 0.8,
        "brake": 0.1,
        "clutch": 0.0,
        "fuel_pct": 0.72,
        "fuel_laps_remaining": 12.4,
        "position": 3,
        "num_players": 16,
        "gap_ahead_s": 1.234,
        "gap_behind_s": 0.567,
        "flags": 0,
        "pit_limiter": False,
        "tc_active": False,
        "abs_active": False,
        "g_lat": 1.2,
        "g_lon": -0.5,
        "g_vert": 1.0,
        "lap_num": 7,
        "lap_total": 20,
        "lap_time_ms": 92345,
        "last_lap_ms": 91876,
        "best_lap_ms": 91500,
        "delta_pb_ms": -469,
        "delta_speed_kmh": 2.3,
        "tyre_temp_fl_c": 92.0,
        "tyre_temp_fr_c": 95.0,
        "tyre_temp_rl_c": 88.0,
        "tyre_temp_rr_c": 89.0,
        "tyre_wear_fl": 0.18,
        "tyre_wear_fr": 0.19,
        "tyre_wear_rl": 0.22,
        "tyre_wear_rr": 0.21,
        "session_clock_ms": 1234567,
        "session_remaining_ms": 600000,
    })
    return src


# Every public overlay window class shipped in the live_modules package.
WINDOW_CLASSES = [
    lm.PositionWindow,
    lm.FuelPctWindow,
    lm.FuelLapsRemainingWindow,
    lm.SpeedWindow,
    lm.GearWindow,
    lm.RpmWindow,
    lm.ThrottleWindow,
    lm.BrakeWindow,
    lm.ClutchWindow,
    lm.GapAheadWindow,
    lm.GapBehindWindow,
    lm.TyreRiskWindow,
    lm.SessionInfoWindow,
    lm.FlagsWindow,
    lm.PitLimiterWindow,
    lm.GMeterWindow,
    lm.RadarWindow,
    lm.DeltaBarWindow,
    lm.SpeedDeltaBarWindow,
]


@pytest.mark.parametrize("cls", WINDOW_CLASSES, ids=lambda c: c.__name__)
def test_window_constructs_and_renders(source, cls):
    win = cls(source)
    try:
        img = win.render_to_image()
        assert isinstance(img, QImage)
        assert not img.isNull()
        assert img.width() > 0
        assert img.height() > 0
    finally:
        win.deleteLater()


def test_all_public_window_classes_listed():
    """Guard: every public *Window symbol re-exported is covered above."""
    exported = {
        name for name in lm.__all__
        if name.endswith("Window") and not name.startswith("_")
    }
    covered = {cls.__name__ for cls in WINDOW_CLASSES}
    missing = exported - covered
    assert not missing, f"uncovered window classes: {sorted(missing)}"
