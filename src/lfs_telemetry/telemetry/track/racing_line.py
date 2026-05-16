"""Geometric racing-line and quasi-static target-speed estimation.

Honest scope
------------
The PTH centerline is the LFS *AI driving line*, not an optimised racing
line. From the per-node geometry (position, tangent, width, radius) we can
derive two physically meaningful quantities:

1. **Target speed v(s)** — the maximum speed a vehicle can carry at every
   node assuming a quasi-static friction circle::

        v_max(s) = sqrt(mu_lat · g · |R(s)|)

   followed by a backward pass to limit braking deceleration to
   ``mu_long · g`` and a forward pass to limit traction acceleration to the
   same bound (standard QSS lap-time approximation).

2. **Heuristic geometric racing line** — for each turn segment (left or
   right, classified by :func:`lfs_telemetry.telemetry.track.enrich.segment_track`) we set a
   lateral offset relative to the centerline that goes from the *outside*
   edge at entry, hits the *inside* edge at the apex, and returns to the
   outside at exit. Straight segments interpolate linearly between
   neighbouring turn offsets. The resulting offset profile is then
   smoothed with a Gaussian filter so the line stays continuous.

This is **not** a minimum-curvature optimal trajectory — that would
require a constrained QP over the lateral offsets. It is a sensible
visual reference and a starting point for the velocity profile.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .enrich import TrackSegment
from .knw import KnwInfo
from .pth import TrackProfile


GRAVITY = 9.81


# ---------------------------------------------------------------------------
# Track edges
# ---------------------------------------------------------------------------


def _unit_normals_xy(profile: TrackProfile) -> np.ndarray:
    """Per-node unit normal in the XY plane, pointing **left** of travel.

    LFS uses a right-handed frame (X east, Y north, Z up). Rotating the XY
    tangent (dx, dy) by +90° gives (-dy, dx), the left-hand normal.
    """
    tan = profile.direction[:, :2].astype(float)
    norm = np.linalg.norm(tan, axis=1, keepdims=True)
    norm = np.where(norm < 1e-9, 1.0, norm)
    tan = tan / norm
    left = np.column_stack([-tan[:, 1], tan[:, 0]])
    return left


def compute_edges(profile: TrackProfile) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(left_edge, right_edge)`` arrays of shape ``(N, 2)``.

    Each edge is the centerline shifted along the lateral normal using the
    per-side asymmetric drive limits stored in the PTH:
    ``left_edge = center + drive_right_m * n``  (LFS-right side, n points left
    in our convention so we use ``drive_right_m`` which is positive),
    ``right_edge = center + drive_left_m * n`` (``drive_left_m`` is negative).
    Separation equals ``drive_right_m - drive_left_m`` = ``profile.width``.
    """
    n = _unit_normals_xy(profile)
    center_xy = profile.pos[:, :2]
    left = center_xy + n * profile.drive_right_m[:, None]
    right = center_xy + n * profile.drive_left_m[:, None]
    return left, right


# ---------------------------------------------------------------------------
# Quasi-static target speed
# ---------------------------------------------------------------------------


def compute_target_speed(
    profile: TrackProfile,
    *,
    mu_lat: float = 1.4,
    mu_long: float = 1.2,
    v_cap_ms: float = 80.0,
    g: float = GRAVITY,
    mu_lat_aero_k: float = 0.0,
) -> np.ndarray:
    """Quasi-static target speed (m/s) at every centerline node.

    The result is the elementwise minimum of:
      * cornering limit ``sqrt(mu_lat(v) · g · |R|)`` (capped at ``v_cap_ms``);
      * a backward sweep limiting braking by ``mu_long·g``;
      * a forward sweep limiting traction by ``mu_long·g``.

    When ``mu_lat_aero_k > 0``, μ_lat grows linearly with v² (downforce
    model). The cornering limit is then resolved by fixed-point iteration
    so the speed used to compute the effective μ matches the speed it
    yields.
    """
    s = profile.s.astype(float)
    radius = np.abs(profile.radius_m).astype(float)
    n = len(s)
    if n < 2:
        return np.zeros(n)

    # Cornering limit. With aero, solve v² = μ(v)·g·R = (μ0 + k·v²)·g·R
    #  → v² (1 - k·g·R) = μ0·g·R  → v² = μ0·g·R / (1 - k·g·R)
    # Fall back to constant-μ formula when k=0 or denominator non-positive.
    if mu_lat_aero_k > 0.0:
        denom = 1.0 - mu_lat_aero_k * g * radius
        v_corner = np.where(
            denom > 1e-6,
            np.sqrt(np.maximum(mu_lat * g * radius / np.maximum(denom, 1e-6),
                               0.0)),
            v_cap_ms,
        )
    else:
        v_corner = np.sqrt(mu_lat * g * radius)
    v_corner = np.minimum(v_corner, v_cap_ms)

    # Treat as cyclic if the path closes back on itself (start ≈ end).
    closed = bool(np.linalg.norm(profile.pos[0, :2] - profile.pos[-1, :2]) < 5.0)

    ds = np.diff(s)
    ds = np.append(ds, ds[-1])  # length n
    a_long = mu_long * g

    # --- Backward pass (braking): walk the path in reverse so v[i] does
    #     not exceed what we can decelerate to from v[i+1].
    v_brake = v_corner.copy()
    iters = 2 if closed else 1
    for _ in range(iters):
        for i in range(n - 2, -1, -1):
            v_brake[i] = min(v_brake[i],
                             np.sqrt(v_brake[i + 1] ** 2 + 2 * a_long * ds[i]))

    # --- Forward pass (traction).
    v_acc = v_brake.copy()
    for _ in range(iters):
        for i in range(1, n):
            v_acc[i] = min(v_acc[i],
                           np.sqrt(v_acc[i - 1] ** 2 + 2 * a_long * ds[i - 1]))

    return v_acc


# ---------------------------------------------------------------------------
# Heuristic geometric racing line
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RacingLine:
    """Per-node racing-line offset and absolute (X, Y) coordinates."""
    profile: TrackProfile
    offset_m: np.ndarray            # (N,) lateral offset; +left, -right
    line_xy: np.ndarray             # (N, 2)


def compute_geometric_line(
    profile: TrackProfile,
    segments: list[TrackSegment],
    *,
    edge_margin_m: float = 0.4,
    smooth_nodes: float = 6.0,
) -> RacingLine:
    """Build a smoothed outside-apex-outside line through the segments.

    For each turn segment:
      * "left" turn  (curvature > 0) → outside is the **right** edge,
        inside is the **left** edge. Offset goes ``-half_w → +half_w → -half_w``.
      * "right" turn (curvature < 0) → mirrored.
      * "straight" → offset is linearly interpolated between the surrounding
        turn-exit and turn-entry offsets.

    Offsets are clipped to ``±(half_width − edge_margin_m)`` and the final
    profile is smoothed with a Gaussian filter of width ``smooth_nodes``.
    """
    n = profile.s.size
    half_w = profile.width / 2.0
    max_offset = np.maximum(half_w - edge_margin_m, 0.0)

    target_offset = np.full(n, np.nan)

    if not segments:
        line_xy = profile.pos[:, :2].copy()
        return RacingLine(profile=profile,
                          offset_m=np.zeros(n), line_xy=line_xy)

    # Pin offsets at segment start/middle/end for turn segments.
    for seg in segments:
        i0, i1 = seg.node_start, seg.node_end
        if seg.kind == "straight":
            continue
        # Inside is +1 for left turn, -1 for right turn.
        inside_sign = +1.0 if seg.kind == "left" else -1.0
        outside_sign = -inside_sign
        mid = (i0 + i1) // 2
        target_offset[i0] = outside_sign * max_offset[i0]
        target_offset[mid] = inside_sign * max_offset[mid]
        target_offset[i1] = outside_sign * max_offset[i1]

    # Anchor straights with 0 offset midpoint to avoid flat-line drift.
    for seg in segments:
        if seg.kind != "straight":
            continue
        i0, i1 = seg.node_start, seg.node_end
        if i1 - i0 < 4:
            continue
        mid = (i0 + i1) // 2
        if np.isnan(target_offset[mid]):
            target_offset[mid] = 0.0

    # Linear interpolation across NaN gaps over the s axis.
    s = profile.s
    finite = ~np.isnan(target_offset)
    if finite.sum() < 2:
        offsets = np.zeros(n)
    else:
        offsets = np.interp(s, s[finite], target_offset[finite])

    # Smooth to remove kinks (units = nodes, not metres).
    if smooth_nodes > 0:
        offsets = gaussian_filter1d(offsets, sigma=float(smooth_nodes),
                                    mode="nearest")

    # Re-clip after smoothing.
    offsets = np.clip(offsets, -max_offset, max_offset)

    # Place along left-hand normal (positive = left).
    left_n = _unit_normals_xy(profile)
    line_xy = profile.pos[:, :2] + left_n * offsets[:, None]

    return RacingLine(profile=profile, offset_m=offsets, line_xy=line_xy)


# ---------------------------------------------------------------------------
# Canonical .knw-derived racing line (per car, from LFS AI knowledge data)
# ---------------------------------------------------------------------------


def compute_knw_line(
    profile: TrackProfile,
    knw: KnwInfo,
    *,
    smooth_nodes: float = 4.0,
    edge_margin_m: float = 0.2,
) -> RacingLine:
    """Build a per-node racing line from a parsed ``.knw`` AI knowledge file.

    The ``.knw`` chain describes the LFS canonical AI line for one car as
    a sequence of segments ``(node_start, node_end, lateral_offset_m)``
    over the PTH centerline. We anchor each segment's offset at its
    midpoint along the path arc-length, then linearly interpolate between
    anchors and apply a light Gaussian smoothing to keep the line
    continuous at segment boundaries.

    Sign convention matches :func:`compute_geometric_line`:
    positive offset = **left** of travel.
    """
    n = int(profile.s.size)
    if n < 2 or not knw.segments:
        line_xy = profile.pos[:, :2].copy()
        return RacingLine(profile=profile,
                          offset_m=np.zeros(n), line_xy=line_xy)

    half_w = profile.width / 2.0
    max_offset = np.maximum(half_w - edge_margin_m, 0.0)
    s = profile.s.astype(float)
    total = float(s[-1] - s[0]) if s[-1] > s[0] else 0.0

    # Anchor each segment at its midpoint (handling wrap around node 0).
    anchor_s: list[float] = []
    anchor_o: list[float] = []
    for seg in knw.segments:
        n0 = max(0, min(seg.node_start, n - 1))
        n1 = max(0, min(seg.node_end, n - 1))
        if n0 <= n1:
            s_mid = 0.5 * (s[n0] + s[n1])
        elif total > 0.0:
            # Wraps past the end of the loop: walk from n0 → end → 0 → n1.
            length = (s[-1] - s[n0]) + (s[n1] - s[0])
            s_mid_raw = s[n0] + 0.5 * length
            s_mid = s_mid_raw if s_mid_raw <= s[-1] else s_mid_raw - total
        else:
            s_mid = s[n0]
        anchor_s.append(float(s_mid))
        anchor_o.append(float(seg.lateral_offset_m))

    # Sort anchors by s so np.interp can use them.
    order = np.argsort(anchor_s)
    a_s = np.asarray(anchor_s, dtype=float)[order]
    a_o = np.asarray(anchor_o, dtype=float)[order]

    # Periodic interpolation: pad on both sides so np.interp wraps cleanly.
    if total > 0.0 and len(a_s) >= 2:
        a_s_pad = np.concatenate([a_s[-1:] - total, a_s, a_s[:1] + total])
        a_o_pad = np.concatenate([a_o[-1:], a_o, a_o[:1]])
        offsets = np.interp(s, a_s_pad, a_o_pad)
    else:
        offsets = np.interp(s, a_s, a_o)

    if smooth_nodes > 0:
        offsets = gaussian_filter1d(offsets, sigma=float(smooth_nodes),
                                    mode="wrap")

    offsets = np.clip(offsets, -max_offset, max_offset)
    left_n = _unit_normals_xy(profile)
    line_xy = profile.pos[:, :2] + left_n * offsets[:, None]
    return RacingLine(profile=profile, offset_m=offsets, line_xy=line_xy)
