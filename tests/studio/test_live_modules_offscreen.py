"""Off-screen render tests for live overlay modules.

Validates the ``render_to_image()`` hook used by alternate output sinks
(OpenVR/OpenXR overlay, screenshots, future capture pipelines). The
test runs headless via the Qt offscreen platform plugin.
"""

from __future__ import annotations

import os
import sys

import pytest

PySide6 = pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage  # noqa: E402

from lfs_telemetry.studio.app import create_app  # noqa: E402
from lfs_telemetry.studio.widgets.live_data_source import (  # noqa: E402
    LiveDataSource,
)
from lfs_telemetry.studio.widgets.live_modules import (  # noqa: E402
    GearWindow,
    RpmWindow,
    SpeedWindow,
)


@pytest.fixture(scope="module")
def qapp():
    return create_app([sys.argv[0]])


@pytest.fixture()
def source(qapp):
    src = LiveDataSource()
    # Inject a minimal snapshot so paint paths run with realistic data.
    src.snapshot.update({
        "speed_kmh": 142.7,
        "gear": 4,
        "rpm": 7250,
        "rpm_max": 9500,
    })
    return src


@pytest.mark.parametrize("cls", [SpeedWindow, GearWindow, RpmWindow])
def test_render_to_image_shape(source, cls):
    win = cls(source)
    try:
        img = win.render_to_image()
        assert isinstance(img, QImage)
        assert not img.isNull()
        assert img.width() == win.width()
        assert img.height() == win.height()
        assert img.format() == QImage.Format.Format_ARGB32_Premultiplied
    finally:
        win.deleteLater()


def test_render_to_image_independent_of_visibility(source):
    win = SpeedWindow(source)
    try:
        # Never call show(): rendering must still work.
        img = win.render_to_image()
        assert not img.isNull()
        # Top-left pixel should be transparent (translucent background).
        px = img.pixelColor(0, 0)
        assert px.alpha() < 255
    finally:
        win.deleteLater()
