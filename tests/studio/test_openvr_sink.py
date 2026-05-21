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
