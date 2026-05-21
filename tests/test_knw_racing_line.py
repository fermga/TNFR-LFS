"""Tests for ``compute_knw_line`` (canonical .knw-derived racing line)."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from lfs_telemetry.telemetry.track.knw import (
    DEFAULT_KNW_DIR,
    KNW_MAGIC,
    KNW_VERSION,
    load_for,
    parse_knw_bytes,
)
from lfs_telemetry.telemetry.track.pth import (
    DEFAULT_SMX_DIR,
    compute_profile,
    parse_pth,
)
from lfs_telemetry.telemetry.track.racing_line import (
    RacingLine,
    compute_knw_line,
)

# ---------------------------------------------------------------------------
# Synthetic profile + knw builder
# ---------------------------------------------------------------------------


class _Profile:
    """Tiny stand-in for ``TrackProfile`` exposing only the fields used."""
    __slots__ = ("direction", "pos", "s", "width")

    def __init__(self, n: int = 50, width: float = 12.0):
        self.s = np.linspace(0.0, n - 1, n)
        self.pos = np.column_stack([self.s, np.zeros(n), np.zeros(n)])
        # Travel along +X → tangent (1, 0, 0); left normal (0, 1).
        self.direction = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
        self.width = np.full(n, width)


def _build_knw(*, pth_nodes: int, segments) -> bytes:
    packed = (pth_nodes & 0xFFFF) | ((len(segments) & 0xFFFF) << 16)
    out = bytearray()
    out += KNW_MAGIC + struct.pack("<H", KNW_VERSION) + b"\x00\x00\x00\x00"
    out += struct.pack("<IffffI", 0, 0.5, 60.0, 50.0, 42.0, packed)
    for flags, n0, n1, delta, lat in segments:
        out += struct.pack("<IIIIff", 0, flags, n0, n1, delta, lat)
    return bytes(out)


def test_compute_knw_line_basic_offsets():
    prof = _Profile(n=50)
    # 3 chained segments (no wrap): straight track, +Y is "left".
    data = _build_knw(pth_nodes=50, segments=[
        (0, 0, 10, 0.0, +1.0),  # +1 m left around node 5
        (0, 10, 25, 0.0, -2.0),  # -2 m right around node 17
        (0, 25, 49, 0.0, +0.5),  # +0.5 m left around node 37
    ])
    knw = parse_knw_bytes(data)
    line = compute_knw_line(prof, knw, smooth_nodes=0.0, edge_margin_m=0.0)
    assert isinstance(line, RacingLine)
    assert line.offset_m.shape == (50,)
    # At each segment midpoint the offset must match the segment value
    # exactly (smoothing disabled). Midpoint = (s[n0]+s[n1])/2 so seg with
    # node range 0..10 anchors at s=5, 10..25 anchors at s=17.5, 25..49 at 37.
    assert line.offset_m[5] == pytest.approx(+1.0, abs=1e-6)
    # s grid is integer, so to hit the 17.5 anchor we check the nearest two
    # nodes bracket the segment value monotonically.
    assert line.offset_m[17] < 0 and line.offset_m[18] < 0
    assert min(line.offset_m[17], line.offset_m[18]) < -1.9
    assert line.offset_m[37] == pytest.approx(+0.5, abs=1e-6)
    # Sign convention: positive offset == +Y (left of +X travel).
    assert line.line_xy[5, 1] == pytest.approx(+1.0, abs=1e-6)
    assert line.line_xy[18, 1] < 0


def test_compute_knw_line_clips_to_track_width():
    prof = _Profile(n=20, width=4.0)  # half-width = 2 m
    data = _build_knw(pth_nodes=20, segments=[
        (0, 0, 10, 0.0, +5.0),   # would be 5 m left, must clip to 1.8
        (0, 10, 19, 0.0, -5.0),
    ])
    knw = parse_knw_bytes(data)
    line = compute_knw_line(prof, knw, smooth_nodes=0.0, edge_margin_m=0.2)
    max_allowed = 4.0 / 2.0 - 0.2 + 1e-6
    assert (np.abs(line.offset_m) <= max_allowed).all()


def test_compute_knw_line_empty_segments_returns_centerline():
    prof = _Profile(n=10)
    data = _build_knw(pth_nodes=10, segments=[])
    knw = parse_knw_bytes(data)
    line = compute_knw_line(prof, knw)
    assert np.all(line.offset_m == 0.0)
    np.testing.assert_allclose(line.line_xy, prof.pos[:, :2])


def test_compute_knw_line_handles_wrap_segment():
    """A segment whose ``node_end < node_start`` wraps past the loop end."""
    prof = _Profile(n=40)
    data = _build_knw(pth_nodes=40, segments=[
        (0, 35, 5, 0.0, +1.0),   # wraps: end of loop → start
        (0, 5, 20, 0.0, -1.0),
        (0, 20, 35, 0.0, +1.0),
    ])
    knw = parse_knw_bytes(data)
    # Must not raise and must yield bounded offsets.
    line = compute_knw_line(prof, knw, smooth_nodes=0.0, edge_margin_m=0.0)
    assert np.isfinite(line.offset_m).all()
    assert np.abs(line.offset_m).max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Real install smoke (skipped without LFS install)
# ---------------------------------------------------------------------------


_HAS_INSTALL = DEFAULT_KNW_DIR.exists() and DEFAULT_SMX_DIR.exists()
install = pytest.mark.skipif(
    not _HAS_INSTALL, reason="LFS install not available")


@install
def test_install_bl1_fbm_line_stays_within_track():
    knw = load_for("BL1", "FBM")
    if knw is None:
        pytest.skip("BL1_FBM.knw missing")
    prof = compute_profile(parse_pth(DEFAULT_SMX_DIR / "BL1.pth"))
    line = compute_knw_line(prof, knw)
    half_w = prof.width / 2.0
    assert (np.abs(line.offset_m) <= half_w + 1e-3).all()
    delta = np.linalg.norm(line.line_xy - prof.pos[:, :2], axis=1)
    assert (delta <= half_w + 1e-3).all()


@install
def test_install_knw_segments_indices_fit_pth():
    """Every .knw segment's node_end stays within the PTH node count."""
    from lfs_telemetry.telemetry.track.knw import list_knw_files, parse_knw
    bad: list[tuple[str, int, int]] = []
    sampled = 0
    for p in list_knw_files()[:50]:    # smoke sample
        info = parse_knw(p)
        layout = info.layout
        pth_path = DEFAULT_SMX_DIR / f"{layout}.pth"
        if not pth_path.exists():
            continue
        sampled += 1
        track = parse_pth(pth_path)
        n_nodes = len(track.nodes)
        max_end = max((s.node_end for s in info.segments), default=0)
        if max_end >= n_nodes:
            bad.append((p.name, max_end, n_nodes))
    assert sampled > 0
    assert not bad, f"knw segments overflow PTH for: {bad[:5]}"
