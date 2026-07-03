"""Tests for the OpenVR overlay sink graceful-degradation path.

These tests verify that when ``openvr`` is not installed (or SteamVR is
not running), the sink reports ``available=False`` and every API call
becomes a safe no-op, so the rest of the studio keeps running.

The "happy path" of actually pushing pixels into SteamVR is covered by
manual VR testing — it requires a live ``vrserver.exe`` and a headset.
"""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture()
def no_openvr(monkeypatch):
    """Ensure ``import openvr`` fails inside the sink."""
    # Hide any real openvr import.
    monkeypatch.setitem(sys.modules, "openvr", None)
    # Reload the module so the lazy importer re-runs with openvr blocked.
    import importlib

    from lfs_telemetry.studio.vr import openvr_overlay
    return importlib.reload(openvr_overlay)


def test_sink_reports_unavailable_without_openvr(no_openvr):
    sink = no_openvr.OpenVROverlaySink()
    assert sink.available is False
    assert sink.init_error is not None


def test_api_is_safe_noop_without_openvr(no_openvr):
    sink = no_openvr.OpenVROverlaySink()
    # ensure_overlay returns False, never raises
    assert sink.ensure_overlay("speed") is False
    # upload returns False, never raises (image arg is unused on no-op path)
    assert sink.upload("speed", image=types.SimpleNamespace()) is False
    # set_visible / destroy_overlay / shutdown are no-ops
    sink.set_visible("speed", True)
    sink.destroy_overlay("speed")
    sink.shutdown()


def test_overlay_pose_matrix_is_identity_rotation():
    from lfs_telemetry.studio.vr.openvr_overlay import OverlayPose

    pose = OverlayPose(x=0.1, y=-0.3, z=-2.0, width_m=0.5)
    m = pose.to_matrix34()
    assert m[0][:3] == (1.0, 0.0, 0.0)
    assert m[1][:3] == (0.0, 1.0, 0.0)
    assert m[2][:3] == (0.0, 0.0, 1.0)
    assert (m[0][3], m[1][3], m[2][3]) == (0.1, -0.3, -2.0)


# ---------------------------------------------------------------------------
# Live-conversion regression tests.
#
# These exercise the exact arguments the sink hands to pyopenvr, using
# real ``openvr`` ctypes types but a fake IVROverlay so no SteamVR /
# headset is needed. They guard the class of bug where a Python tuple or
# ``bytes`` object is passed to a pyopenvr method that internally calls
# ``ctypes.byref`` (which raises ``TypeError`` on non-ctypes objects and
# silently disabled the whole VR overlay path).
# ---------------------------------------------------------------------------


class _FakeOverlay:
    """Minimal IVROverlay stand-in that records the args it receives."""

    def __init__(self) -> None:
        self.transform = None
        self.tracking_origin = None
        self.raw_buffer = None
        self.shown: list[int] = []
        self._next_handle = 100

    def createOverlay(self, key, name):
        self._next_handle += 1
        return self._next_handle

    def setOverlayWidthInMeters(self, handle, width):
        self.width = width

    def setOverlayTransformAbsolute(
        self, handle, tracking_origin, transform,
    ):
        self.tracking_origin = tracking_origin
        self.transform = transform

    def setOverlayRaw(self, handle, buffer, w, h, bpp):
        self.raw_buffer = buffer

    def showOverlay(self, handle):
        self.shown.append(handle)


def _make_live_sink(openvr_mod, fake_overlay):
    """Build a sink with real openvr types but a fake IVROverlay.

    Bypasses ``__init__`` (which needs a live SteamVR) so the pixel and
    transform conversion logic can be exercised deterministically.
    """
    import threading

    from lfs_telemetry.studio.vr.openvr_overlay import OpenVROverlaySink

    sink = OpenVROverlaySink.__new__(OpenVROverlaySink)
    sink._lock = threading.Lock()
    sink._openvr = openvr_mod
    sink._vr_overlay = fake_overlay
    sink._initialized = True
    sink._init_error = None
    sink._entries = {}
    return sink


def test_ensure_overlay_passes_ctypes_transform():
    """Overlays anchor in seated space with a real ``HmdMatrix34_t``.

    Using an absolute (seated) transform keeps each panel fixed in the
    cockpit instead of following the gaze. pyopenvr calls
    ``ctypes.byref`` on the matrix; a plain tuple would raise and leave
    the overlay created-but-invisible.
    """
    import ctypes

    openvr = pytest.importorskip("openvr")
    fake = _FakeOverlay()
    sink = _make_live_sink(openvr, fake)

    assert sink.ensure_overlay("speed") is True
    assert fake.tracking_origin == openvr.TrackingUniverseSeated
    assert isinstance(fake.transform, openvr.HmdMatrix34_t)
    # Must not raise — this is what pyopenvr does internally.
    ctypes.byref(fake.transform)


def test_upload_passes_ctypes_pixel_buffer():
    """The pixel buffer must be a ctypes array, not raw ``bytes``.

    pyopenvr calls ``ctypes.byref`` on it before handing it to
    ``setOverlayRaw``; raw ``bytes`` would raise and drop the frame.
    """
    import ctypes

    openvr = pytest.importorskip("openvr")
    QImage = pytest.importorskip("PySide6.QtGui").QImage

    fake = _FakeOverlay()
    sink = _make_live_sink(openvr, fake)
    assert sink.ensure_overlay("speed") is True

    img = QImage(8, 4, QImage.Format.Format_RGBA8888)
    img.fill(0xFF112233)
    assert sink.upload("speed", img) is True

    # Must not raise, and must carry the full RGBA payload.
    ctypes.byref(fake.raw_buffer)
    assert len(bytes(fake.raw_buffer)) == 8 * 4 * 4
    # First upload also shows the overlay.
    assert fake.shown
