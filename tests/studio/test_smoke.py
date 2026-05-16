"""Studio smoke + structural tests.

Skipped automatically if PySide6 isn't installed (e.g. on a CI runner
that only validates the Dash viewer slice).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PySide6 = pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

# Force the Qt offscreen platform plugin so the test runs headless on
# Windows agents without a display server.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from lfs_telemetry.studio.app import create_app  # noqa: E402
from lfs_telemetry.studio.main_window import MainWindow  # noqa: E402
from lfs_telemetry.studio.models import (  # noqa: E402
    CapturesTableModel,
    ChannelTreeModel,
)
from lfs_telemetry.studio.signals import SignalBus  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = create_app([sys.argv[0]])
    yield app
    # Don't quit() — pytest may have other Qt fixtures.


def test_signal_bus_signals_exist():
    bus = SignalBus()
    # All cross-dock signals must be present so we don't break the
    # studio at refactor time without a test catching it.
    for name in (
        "workspace_changed", "captures_refreshed", "laps_selected",
        "channels_changed", "available_columns_changed", "x_axis_changed",
        "cursor_moved", "cursor_left", "status_message",
    ):
        assert hasattr(bus, name), name


def test_captures_model_empty():
    model = CapturesTableModel()
    assert model.rowCount() == 0
    assert model.columnCount() == 7
    assert model.path_at(0) is None


def test_channel_tree_model_starts_disabled(qapp):
    model = ChannelTreeModel()
    # Before set_available_columns(), every channel item is disabled.
    assert model.rowCount() > 0
    assert model.checked_columns() == []


def test_main_window_constructs(qapp, tmp_path: Path):
    # Empty workspace folder: MainWindow must build without crashing
    # even when no captures exist.
    ws = tmp_path / "captures"
    ws.mkdir()
    win = MainWindow(ws)
    try:
        assert win.windowTitle().startswith("LFS Telemetry Studio")
        assert win.centralWidget() is not None
    finally:
        win.close()


def test_main_window_with_synthetic_capture(qapp, tmp_path: Path):
    """If a synthetic capture exists, the captures dock loads it."""
    src = Path("captures/synthetic.csv")
    if not src.exists():
        pytest.skip("no synthetic capture available")
    ws = tmp_path / "captures"
    ws.mkdir()
    (ws / "synthetic.csv").write_bytes(src.read_bytes())
    win = MainWindow(ws)
    try:
        # The captures dock auto-refreshes during construction.
        from lfs_telemetry.studio.widgets import CapturesDock
        dock = win.findChild(CapturesDock)
        assert dock is not None
        assert len(dock.selected_paths()) == 0
        # And the model has at least one row.
        model = dock._model  # internal but stable
        assert model.rowCount() >= 1
    finally:
        win.close()
