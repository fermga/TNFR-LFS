"""Tests for :class:`VrMirror` graceful degradation.

Verifies that the mirror cleanly reports unavailability when ``openvr``
is missing, and that enable/disable are safe no-ops in that case. The
real upload tick is exercised manually with a live SteamVR session.
"""

from __future__ import annotations

import os
import sys

import pytest

PySide6 = pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from lfs_telemetry.studio.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return create_app([sys.argv[0]])


@pytest.fixture()
def no_openvr(monkeypatch, qapp):
    """Make ``import openvr`` fail and reload the sink module."""
    monkeypatch.setitem(sys.modules, "openvr", None)
    import importlib

    from lfs_telemetry.studio.vr import openvr_overlay, vr_mirror
    importlib.reload(openvr_overlay)
    return importlib.reload(vr_mirror)


def test_mirror_enable_returns_false_without_openvr(no_openvr):
    mirror = no_openvr.VrMirror(provider=dict)
    try:
        assert mirror.enable() is False
        assert mirror.is_enabled is False
        assert mirror.is_available is False
    finally:
        mirror.shutdown()


def test_mirror_disable_is_safe_when_never_enabled(no_openvr):
    mirror = no_openvr.VrMirror(provider=dict)
    # disable() before enable() must not raise.
    mirror.disable()
    mirror.shutdown()


def test_mirror_provider_called_on_demand(no_openvr):
    """Even without VR, the provider isn't called until enable()."""
    calls = {"n": 0}

    def provider():
        calls["n"] += 1
        return {}

    mirror = no_openvr.VrMirror(provider=provider)
    try:
        assert calls["n"] == 0
        mirror.enable()  # returns False, never starts ticking
        assert calls["n"] == 0
    finally:
        mirror.shutdown()
