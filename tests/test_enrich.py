"""Tests for the spatial-join / segment enrichment pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lfs_telemetry.telemetry.track.enrich import (
    TrackIndex,
    assign_segment,
    detect_track,
    enrich_dataframe,
    segment_track,
)
from lfs_telemetry.telemetry.track.pth import DEFAULT_SMX_DIR, compute_profile, parse_pth

SMX = DEFAULT_SMX_DIR
pytestmark = pytest.mark.skipif(
    not SMX.exists(), reason="LFS install not available at C:/LFS",
)


def _bl1_profile():
    return compute_profile(parse_pth(SMX / "BL1.pth"))


def test_kdtree_recovers_node_indices():
    prof = _bl1_profile()
    idx_, dist = TrackIndex.from_profile(prof).query(
        prof.pos[:, 0], prof.pos[:, 1]
    )
    # Querying with the centerline coordinates must return the same nodes
    # exactly, with zero distance.
    np.testing.assert_array_equal(idx_, np.arange(len(prof.s)))
    np.testing.assert_allclose(dist, 0.0, atol=1e-6)


def test_enrich_dataframe_adds_columns():
    prof = _bl1_profile()
    # Build a fake CSV: car driving along the centerline of BL1.
    sub = pd.DataFrame({
        "pos_x": prof.pos[::10, 0],
        "pos_y": prof.pos[::10, 1],
    })
    enr = enrich_dataframe(sub, prof)
    for col in ("track_node", "track_s_m", "slope_local",
                "radius_local", "width_local", "track_offset_m"):
        assert col in enr.columns
    # Driving on the centerline → offset is ~0.
    assert (enr["track_offset_m"] < 0.5).all()
    # s must be monotonically non-decreasing along the path.
    assert enr["track_s_m"].is_monotonic_increasing


def test_segment_track_produces_meaningful_segments():
    prof = _bl1_profile()
    segs = segment_track(prof, straight_radius_m=250.0, min_segment_m=25.0)
    assert len(segs) >= 4, f"BL1 should have at least 4 segments, got {len(segs)}"
    # Total covered length matches the profile.
    # Sum of segment lengths is close to the lap length (small gaps where
    # consecutive segments share a node are expected).
    total = sum(s.length_m for s in segs)
    assert total >= 0.9 * prof.total_length_m
    assert total <= prof.total_length_m
    # We expect both straights and turns.
    kinds = {s.kind for s in segs}
    assert "straight" in kinds
    assert ("left" in kinds) or ("right" in kinds)


def test_assign_segment_round_trip():
    prof = _bl1_profile()
    segs = segment_track(prof)
    sub = pd.DataFrame({
        "pos_x": prof.pos[:, 0],
        "pos_y": prof.pos[:, 1],
    })
    enr = enrich_dataframe(sub, prof)
    enr = assign_segment(enr, segs)
    assert "segment_id" in enr.columns
    assert "segment_kind" in enr.columns
    assert (enr["segment_id"] >= 0).all()
    # Number of distinct segment_ids matches len(segs).
    assert enr["segment_id"].nunique() == len(segs)


def test_detect_track_via_ctx_track_column():
    df = pd.DataFrame({
        "ctx_track": ["BL1"] * 5,
        "pos_x": [0.0] * 5,
        "pos_y": [0.0] * 5,
    })
    match = detect_track(df)
    assert match is not None
    assert match.name == "BL1"
    assert match.source == "ctx_track"
    assert match.confidence == "high"


def test_detect_track_via_spatial_match():
    prof = _bl1_profile()
    # Drive along BL1 centerline, no race-context column at all.
    idx = np.linspace(0, len(prof.s) - 1, 100).astype(int)
    df = pd.DataFrame({
        "pos_x": prof.pos[idx, 0],
        "pos_y": prof.pos[idx, 1],
    })
    match = detect_track(df)
    assert match is not None
    assert match.name == "BL1", f"expected BL1, got {match.name}"
    assert match.source == "bbox"
    assert match.score < 5.0
    # Top candidate is BL1.
    assert match.candidates[0][0] == "BL1"


def test_detect_track_returns_none_for_unusable_positions():
    df = pd.DataFrame({"pos_x": [0.0] * 10, "pos_y": [0.0] * 10})
    assert detect_track(df) is None
