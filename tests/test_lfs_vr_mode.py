"""Tests for :func:`lfs_telemetry.lfs_config.read_lfs_vr_mode`.

Covers the modern single-line ``G3D_OPTIONS`` format used by current
LFS as well as the legacy discrete ``OpenVR Mode N`` lines. The field
positions in ``G3D_OPTIONS`` (device index 4, VR-system index 5) are
those confirmed against LFS's own language strings.
"""
from __future__ import annotations

from lfs_telemetry import lfs_config


def _write_cfg(lfs_dir, body: str) -> None:
    lfs_config.cfg_path_for(lfs_dir).write_text(body, encoding="latin-1")


def test_returns_none_when_cfg_missing(tmp_path):
    assert lfs_config.read_lfs_vr_mode(tmp_path) is None


def test_modern_openvr_headset(tmp_path):
    # Real-world sample: device=2 (VR headset), vr_system=1 (OpenVR).
    _write_cfg(
        tmp_path,
        "View Smooth 0.2\n"
        "G3D_OPTIONS 0 0 0 0 2 1 1\n"
        "G3D_VR_FLAGS 2\n",
    )
    assert lfs_config.read_lfs_vr_mode(tmp_path) == ("OpenVR", 1)


def test_modern_oculus_headset(tmp_path):
    # device=2 (VR headset), vr_system=0 (Oculus Rift).
    _write_cfg(tmp_path, "G3D_OPTIONS 0 0 0 0 2 0 1\n")
    assert lfs_config.read_lfs_vr_mode(tmp_path) == ("Oculus", 0)


def test_modern_monitor_device_is_not_vr(tmp_path):
    # device=0 (TV / monitor / projector) → no headset.
    _write_cfg(tmp_path, "G3D_OPTIONS 0 0 0 0 0 1 1\n")
    assert lfs_config.read_lfs_vr_mode(tmp_path) is None


def test_modern_3d_display_device_is_not_vr(tmp_path):
    # device=1 (3D display device, e.g. anaglyph / side-by-side).
    _write_cfg(tmp_path, "G3D_OPTIONS 0 0 0 0 1 0 4\n")
    assert lfs_config.read_lfs_vr_mode(tmp_path) is None


def test_modern_line_is_authoritative_over_stale_legacy(tmp_path):
    # A leftover legacy line must not override the modern device flag.
    _write_cfg(
        tmp_path,
        "OpenVR Mode 1\n"
        "G3D_OPTIONS 0 0 0 0 0 1 1\n",
    )
    assert lfs_config.read_lfs_vr_mode(tmp_path) is None


def test_legacy_openvr_mode_on(tmp_path):
    _write_cfg(tmp_path, "OpenVR Mode 1\n")
    assert lfs_config.read_lfs_vr_mode(tmp_path) == ("OpenVR", 1)


def test_legacy_mode_zero_is_off(tmp_path):
    _write_cfg(tmp_path, "OpenVR Mode 0\nOculus Mode 0\n")
    assert lfs_config.read_lfs_vr_mode(tmp_path) is None


def test_malformed_g3d_options_falls_through(tmp_path):
    # Too few fields / non-integer → ignored, legacy line still wins.
    _write_cfg(
        tmp_path,
        "G3D_OPTIONS 0 0\n"
        "G3D_OPTIONS 0 0 0 0 x y z\n"
        "OpenVR Mode 1\n",
    )
    assert lfs_config.read_lfs_vr_mode(tmp_path) == ("OpenVR", 1)
