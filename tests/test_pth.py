"""Tests for the PTH parser."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lfs_telemetry.telemetry.track.pth import (
    DEFAULT_SMX_DIR,
    HEADER_BYTES,
    NODE_BYTES,
    PTH_MAGIC,
    compute_profile,
    list_path_files,
    parse_pth,
)


SMX_DIR = DEFAULT_SMX_DIR


pytestmark = pytest.mark.skipif(
    not SMX_DIR.exists(),
    reason="LFS install not available at C:/LFS — PTH tests skipped.",
)


def test_magic_and_layout_bl1():
    p = parse_pth(SMX_DIR / "BL1.pth")
    assert p.name == "BL1"
    assert p.num_nodes == 548
    assert p.raw_header[:6] == PTH_MAGIC


def test_all_pth_files_parse():
    files = list_path_files(SMX_DIR)
    assert len(files) >= 80, f"only {len(files)} PTH files found"
    for f in files:
        p = parse_pth(f)
        # File size should equal header + N * node_size
        size = f.stat().st_size
        assert size == HEADER_BYTES + p.num_nodes * NODE_BYTES, \
            f"{f.name}: layout mismatch"


def test_directions_are_unit():
    p = parse_pth(SMX_DIR / "BL1.pth")
    norms = np.linalg.norm(p.direction, axis=1)
    # Allow a small slack in case some nodes are zeroed.
    valid = norms > 0.5
    assert valid.sum() > 0
    np.testing.assert_allclose(norms[valid], 1.0, atol=1e-3)


def test_profile_bl1_reasonable():
    p = parse_pth(SMX_DIR / "BL1.pth")
    prof = compute_profile(p)
    # Blackwood GP (BL1) — PTH path length is the AI/centerline path,
    # not the official 2.046 km lap; it lands around 3.3 km after cutting
    # the pit lane teleport.
    assert 1500.0 < prof.total_length_m < 5000.0, \
        f"BL1 length {prof.total_length_m:.0f}m out of range"
    zmin, zmax = prof.elevation_range_m
    assert (zmax - zmin) > 1.0, "BL1 should have elevation change"
    # Should have at least one tight corner (R < 60 m).
    assert prof.radius_m.min() < 60.0
    # End-to-start chord should be short — the path is a closed loop.
    closing = np.linalg.norm(prof.pos[-1] - prof.pos[0])
    assert closing < 30.0, f"BL1 not closed (chord={closing:.1f}m)"


def test_profile_handles_empty():
    # SO7.pth ships with zero nodes.
    so7 = SMX_DIR / "SO7.pth"
    if not so7.exists():
        pytest.skip("SO7.pth not present")
    p = parse_pth(so7)
    prof = compute_profile(p)
    assert prof.total_length_m == 0.0


def test_asymmetric_edges_have_consistent_signs():
    """PTH stores per-node L/R drive/limit offsets with strict sign convention.

    Verified empirically across AS1/AS2/BL1/KY1/KY3/WE1: the 4 trailing
    f32s per node are always ``(limit_left ≤ 0, limit_right ≥ 0,
    drive_left ≤ 0, drive_right ≥ 0)``. Drive edges must lie inside the
    outer limits, and ``width = drive_right - drive_left`` must be
    strictly positive.
    """
    p = parse_pth(SMX_DIR / "BL1.pth")
    prof = compute_profile(p)
    # Non-empty profile after pit-lane teleport cut.
    assert prof.drive_left_m.size > 0
    # Sign convention.
    assert (prof.drive_left_m <= 1e-6).all(), "drive_left must be ≤ 0"
    assert (prof.drive_right_m >= -1e-6).all(), "drive_right must be ≥ 0"
    assert (prof.limit_left_m <= 1e-6).all(), "limit_left must be ≤ 0"
    assert (prof.limit_right_m >= -1e-6).all(), "limit_right must be ≥ 0"
    # Drive edges inside outer limits.
    assert (prof.limit_left_m <= prof.drive_left_m + 1e-6).all()
    assert (prof.limit_right_m >= prof.drive_right_m - 1e-6).all()
    # Total drivable width is positive and matches the property.
    assert (prof.width > 0.0).all()
    np.testing.assert_allclose(
        prof.width, prof.drive_right_m - prof.drive_left_m, atol=1e-9,
    )
    # Reasonable magnitudes for an asphalt circuit (≤ 60 m total).
    assert prof.width.max() < 60.0
    assert prof.width.mean() > 5.0
