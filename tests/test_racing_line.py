"""Tests for racing-line and target-speed estimation."""
from __future__ import annotations

import numpy as np
import pytest

from lfs_telemetry.telemetry.track.enrich import segment_track
from lfs_telemetry.telemetry.track.pth import DEFAULT_SMX_DIR, compute_profile, parse_pth
from lfs_telemetry.telemetry.track.racing_line import (
    compute_edges,
    compute_geometric_line,
    compute_target_speed,
)

SMX = DEFAULT_SMX_DIR
pytestmark = pytest.mark.skipif(
    not SMX.exists(), reason="LFS install not available at C:/LFS",
)


def _bl1():
    return compute_profile(parse_pth(SMX / "BL1.pth"))


def test_edges_match_width():
    prof = _bl1()
    left, right = compute_edges(prof)
    # Edge-to-edge distance ≈ width at every node.
    sep = np.linalg.norm(left - right, axis=1)
    np.testing.assert_allclose(sep, prof.width, atol=1e-6)


def test_target_speed_bounded_and_positive():
    prof = _bl1()
    v = compute_target_speed(prof, mu_lat=1.4, mu_long=1.2, v_cap_ms=80.0)
    assert v.shape == prof.s.shape
    assert (v > 0).all()
    assert (v <= 80.0).all()
    # Sanity: BL1 has long straights, top speed should reach the cap or close.
    assert v.max() > 60.0


def test_geometric_line_stays_within_track():
    prof = _bl1()
    segs = segment_track(prof)
    line = compute_geometric_line(prof, segs, edge_margin_m=0.4)
    # Offset must respect the per-node half-width margin.
    half_w = prof.width / 2.0
    assert (np.abs(line.offset_m) <= half_w + 1e-6).all()
    # Line points lie close to the centerline (within half a track width).
    delta = np.linalg.norm(line.line_xy - prof.pos[:, :2], axis=1)
    assert (delta <= half_w + 1e-6).all()


def test_apex_offsets_have_correct_sign():
    """Inside the apex of a left turn → positive offset; right turn → negative."""
    prof = _bl1()
    segs = segment_track(prof)
    line = compute_geometric_line(prof, segs, edge_margin_m=0.4,
                                  smooth_nodes=0.0)
    # Use raw (un-smoothed) offsets so sign at the pinned apex is exact.
    for seg in segs:
        if seg.kind == "straight":
            continue
        mid = (seg.node_start + seg.node_end) // 2
        off = line.offset_m[mid]
        if seg.kind == "left":
            assert off >= -1e-6, f"left apex {seg.index} offset={off}"
        else:
            assert off <= 1e-6, f"right apex {seg.index} offset={off}"


def test_aero_curve_increases_corner_speed():
    """Positive ``mu_lat_aero_k`` must raise (or equal) cornering speeds vs k=0."""
    prof = _bl1()
    v_flat = compute_target_speed(prof, mu_lat=1.4, mu_long=1.2,
                                  v_cap_ms=100.0, mu_lat_aero_k=0.0)
    v_aero = compute_target_speed(prof, mu_lat=1.4, mu_long=1.2,
                                  v_cap_ms=100.0, mu_lat_aero_k=2e-4)
    # Aero never makes you slower; must beat or match the flat curve.
    assert (v_aero >= v_flat - 1e-6).all()
    # And on at least some nodes it should be strictly higher.
    assert (v_aero > v_flat + 0.1).any()
