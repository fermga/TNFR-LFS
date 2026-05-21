"""Spatial join: telemetry CSV ↔ track geometry from PTH.

Builds a KDTree over the (X, Y) centerline nodes of an LFS track and
augments a telemetry DataFrame with track-relative columns:

    track_node       int    nearest centerline node index
    track_s_m        float  cumulative arc length along the path (m)
    slope_local      float  local slope (%)
    radius_local     float  local turning radius (m)  (clipped at 1e6)
    curvature_local  float  signed curvature (1/m)    (+ left)
    width_local      float  total drivable width at the node (m)
    drive_left_local  float PTH drive-edge offset, LFS-left  (≤ 0, m)
    drive_right_local float PTH drive-edge offset, LFS-right (≥ 0, m)
    limit_left_local  float PTH outer-limit offset, LFS-left  (≤ 0, m)
    limit_right_local float PTH outer-limit offset, LFS-right (≥ 0, m)
    track_offset_m   float  perpendicular distance to centerline (m)

Usage
-----

>>> from lfs_telemetry.telemetry.track.pth import parse_pth, compute_profile
>>> from lfs_telemetry.telemetry.track.enrich import enrich_dataframe
>>> path = parse_pth("C:/LFS/data/smx/BL1.pth")
>>> profile = compute_profile(path)
>>> df = enrich_dataframe(df, profile)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path as _Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .pin import PinInfo
from .pin import load_all as _load_pins
from .pth import (
    DEFAULT_SMX_DIR,
    TrackProfile,
    compute_profile,
    list_path_files,
    parse_pth,
)

# Map smx_dir -> {env: PinInfo}. Built lazily per directory.
_PIN_CACHE: dict[_Path, dict[str, PinInfo]] = {}


def _pins_for(smx_dir: _Path) -> dict[str, PinInfo]:
    key = _Path(smx_dir).resolve()
    if key not in _PIN_CACHE:
        try:
            _PIN_CACHE[key] = _load_pins(smx_dir)
        except (OSError, ValueError):
            # SMX dir missing or PINs malformed — cache empty result.
            _PIN_CACHE[key] = {}
    return _PIN_CACHE[key]


@dataclass(slots=True)
class TrackIndex:
    """KDTree over a profile's (X, Y) centerline."""
    profile: TrackProfile
    tree: cKDTree

    @classmethod
    def from_profile(cls, profile: TrackProfile) -> TrackIndex:
        if profile.pos.shape[0] == 0:
            raise ValueError(f"profile {profile.name!r} has no nodes")
        xy = profile.pos[:, :2]
        return cls(profile=profile, tree=cKDTree(xy))

    @classmethod
    def from_pth(cls, pth_path: str | _Path) -> TrackIndex:
        return cls.from_profile(compute_profile(parse_pth(pth_path)))

    def query(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (node_idx, distance_to_centerline) for each (x, y) point."""
        pts = np.column_stack([np.asarray(x, dtype=float),
                               np.asarray(y, dtype=float)])
        dist, idx = self.tree.query(pts, k=1)
        return idx.astype(np.int64), dist.astype(np.float64)


def enrich_dataframe(df: pd.DataFrame, profile: TrackProfile) -> pd.DataFrame:
    """Return a copy of *df* augmented with track-relative columns.

    Requires ``pos_x`` and ``pos_y`` columns (LFS world frame, metres).
    """
    if "pos_x" not in df.columns or "pos_y" not in df.columns:
        raise KeyError("dataframe needs 'pos_x' and 'pos_y' columns")
    if profile.pos.shape[0] == 0:
        raise ValueError(f"profile {profile.name!r} has no nodes")

    idx_, dist = TrackIndex.from_profile(profile).query(
        df["pos_x"].to_numpy(), df["pos_y"].to_numpy()
    )

    out = df.copy()
    out["track_node"] = idx_
    out["track_s_m"] = profile.s[idx_]
    out["slope_local"] = profile.slope_pct[idx_]
    out["curvature_local"] = profile.curvature_1_per_m[idx_]
    out["radius_local"] = profile.radius_m[idx_]
    out["width_local"] = profile.width[idx_]
    out["drive_left_local"] = profile.drive_left_m[idx_]
    out["drive_right_local"] = profile.drive_right_m[idx_]
    out["limit_left_local"] = profile.limit_left_m[idx_]
    out["limit_right_local"] = profile.limit_right_m[idx_]
    out["track_offset_m"] = dist
    # Optional BVH-derived barriers (populated by
    # ``geom3d.enrich_profile_with_smx``). When present, expose a
    # per-sample margin-to-wall channel so it shows up in ChartsDock
    # alongside the standard track-relative columns.
    if profile.barrier_left_m is not None:
        out["barrier_left_local"] = profile.barrier_left_m[idx_]
    if profile.barrier_right_m is not None:
        out["barrier_right_local"] = profile.barrier_right_m[idx_]
    if (profile.barrier_left_m is not None
            and profile.barrier_right_m is not None):
        out["margin_to_wall_m"] = np.minimum(
            profile.barrier_left_m[idx_], profile.barrier_right_m[idx_]
        )
    if profile.effective_width_m is not None:
        out["effective_width_local"] = profile.effective_width_m[idx_]
    return out


def enrich_csv(in_csv: str | _Path,
               profile: TrackProfile,
               out_csv: str | _Path | None = None) -> _Path:
    """Read *in_csv*, enrich with track columns, write to *out_csv* (or
    ``<stem>_enriched.csv`` next to it)."""
    in_p = _Path(in_csv)
    out_p = (_Path(out_csv) if out_csv else
             in_p.with_name(f"{in_p.stem}_enriched{in_p.suffix}"))
    df = pd.read_csv(in_p)
    enriched = enrich_dataframe(df, profile)
    enriched.to_csv(out_p, index=False)
    return out_p


# ---------------------------------------------------------------------------
# Segment generation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrackSegment:
    """One physical segment of the track."""
    index: int
    kind: str                # "straight" | "left" | "right" | "brake"
    s_start_m: float
    s_end_m: float
    length_m: float
    node_start: int
    node_end: int            # inclusive
    mean_radius_m: float
    mean_slope_pct: float
    mean_curvature: float


def segment_track(
    profile: TrackProfile,
    *,
    straight_radius_m: float = 250.0,
    min_segment_m: float = 25.0,
) -> list[TrackSegment]:
    """Split a track profile into physical segments by curvature regime.

    A node is classified as ``straight`` when ``|R| > straight_radius_m``,
    otherwise ``left`` (curvature > 0) or ``right`` (curvature < 0). Adjacent
    nodes with the same classification are merged into a single segment.
    Segments shorter than ``min_segment_m`` are absorbed into the previous one.
    """
    if profile.s.size < 2:
        return []
    s = profile.s
    r = profile.radius_m
    k = profile.curvature_1_per_m
    slope = profile.slope_pct

    n = len(s)
    kind_per_node = np.where(
        np.abs(r) > straight_radius_m, "straight",
        np.where(k > 0, "left", "right"),
    )

    segments: list[TrackSegment] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and kind_per_node[j + 1] == kind_per_node[i]:
            j += 1
        seg = TrackSegment(
            index=len(segments),
            kind=str(kind_per_node[i]),
            s_start_m=float(s[i]),
            s_end_m=float(s[j]),
            length_m=float(s[j] - s[i]),
            node_start=int(i),
            node_end=int(j),
            mean_radius_m=float(np.mean(r[i:j + 1])),
            mean_slope_pct=float(np.mean(slope[i:j + 1])),
            mean_curvature=float(np.mean(k[i:j + 1])),
        )
        segments.append(seg)
        i = j + 1

    # Merge tiny segments into the previous one (recompute means).
    merged: list[TrackSegment] = []
    for seg in segments:
        if merged and seg.length_m < min_segment_m:
            prev = merged[-1]
            i0 = prev.node_start
            j1 = seg.node_end
            merged[-1] = TrackSegment(
                index=prev.index,
                kind=prev.kind,
                s_start_m=prev.s_start_m,
                s_end_m=float(s[j1]),
                length_m=float(s[j1] - s[i0]),
                node_start=i0,
                node_end=j1,
                mean_radius_m=float(np.mean(r[i0:j1 + 1])),
                mean_slope_pct=float(np.mean(slope[i0:j1 + 1])),
                mean_curvature=float(np.mean(k[i0:j1 + 1])),
            )
        else:
            merged.append(seg)

    # Re-index after merging.
    return [
        TrackSegment(
            index=k_, kind=s_.kind,
            s_start_m=s_.s_start_m, s_end_m=s_.s_end_m, length_m=s_.length_m,
            node_start=s_.node_start, node_end=s_.node_end,
            mean_radius_m=s_.mean_radius_m, mean_slope_pct=s_.mean_slope_pct,
            mean_curvature=s_.mean_curvature,
        )
        for k_, s_ in enumerate(merged)
    ]


def assign_segment(df_enriched: pd.DataFrame,
                   segments: list[TrackSegment]) -> pd.DataFrame:
    """Add a ``segment_id`` column to an enriched DataFrame.

    Assignment is by ``track_node`` index, which is robust regardless of how
    many laps the CSV covers.
    """
    if "track_node" not in df_enriched.columns:
        raise KeyError("dataframe must be enriched first (missing 'track_node')")
    if not segments:
        out = df_enriched.copy()
        out["segment_id"] = -1
        out["segment_kind"] = ""
        return out

    # Build a per-node lookup: node_id -> segment index, kind.
    max_node = max(s.node_end for s in segments) + 1
    seg_id_lut = np.full(max_node, -1, dtype=np.int64)
    seg_kind_lut = np.empty(max_node, dtype=object)
    for seg in segments:
        seg_id_lut[seg.node_start:seg.node_end + 1] = seg.index
        seg_kind_lut[seg.node_start:seg.node_end + 1] = seg.kind

    nodes = df_enriched["track_node"].to_numpy()
    nodes = np.clip(nodes, 0, max_node - 1)

    out = df_enriched.copy()
    out["segment_id"] = seg_id_lut[nodes]
    out["segment_kind"] = seg_kind_lut[nodes]
    return out


# ---------------------------------------------------------------------------
# Track auto-detection
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrackMatch:
    """Result of an auto-detection lookup against the LFS smx directory."""
    name: str                # short variant name (e.g. "BL1")
    source: str              # "ctx_track" | "bbox" | "centroid"
    score: float             # mean nearest-neighbour distance (m); lower=better
    confidence: str          # "high" | "medium" | "low"
    candidates: list[tuple[str, float]]  # (name, score) sorted by score asc


def _ctx_track_value(df: pd.DataFrame) -> str | None:
    """Return the most common non-empty value of ``ctx_track`` in *df*."""
    if "ctx_track" not in df.columns:
        return None
    vals = df["ctx_track"].dropna()
    vals = vals[vals.astype(str).str.strip() != ""]
    if vals.empty:
        return None
    # Mode (most frequent) — robust if track changes mid-stint.
    return str(vals.mode().iloc[0]).strip().upper()


def _normalize_variant(name: str) -> str:
    """Strip extension and normalise to upper-case stem."""
    p = _Path(name)
    return p.stem.upper()


def detect_track(
    df: pd.DataFrame,
    *,
    smx_dir: _Path | str = DEFAULT_SMX_DIR,
    sample_points: int = 256,
    require_pth: bool = True,
) -> TrackMatch | None:
    """Auto-detect the LFS track variant for an enriched/raw telemetry DataFrame.

    Strategy
    --------
    1. If ``ctx_track`` column has a non-empty value (recorded via InSim
       ``RaceContext``), that wins. We still verify the corresponding
       ``<name>.pth`` exists when ``require_pth`` is True; otherwise we
       fall back to the spatial match.
    2. Spatial match: for every available ``.pth`` in ``smx_dir``, build a
       KDTree over its centerline and compute the mean distance from up
       to ``sample_points`` evenly-spaced (X, Y) telemetry points. Pick
       the variant with the smallest mean distance.

    Returns ``None`` if positions are unusable (all zero / no pos columns).
    """
    smx = _Path(smx_dir)

    # --- Step 1: trust the recorded race context if present ---------------
    ctx_name = _ctx_track_value(df)
    if ctx_name:
        pth_file = smx / f"{ctx_name}.pth"
        if pth_file.exists() or not require_pth:
            return TrackMatch(
                name=ctx_name, source="ctx_track",
                score=0.0, confidence="high", candidates=[(ctx_name, 0.0)],
            )

    # --- Step 2: spatial match against every PTH --------------------------
    if "pos_x" not in df.columns or "pos_y" not in df.columns:
        return None

    xy = df[["pos_x", "pos_y"]].to_numpy(dtype=float)
    # Filter out all-zero / sentinel rows.
    finite = np.isfinite(xy).all(axis=1)
    nonzero = (np.abs(xy).sum(axis=1) > 1e-6)
    xy = xy[finite & nonzero]
    if len(xy) < 4:
        return None
    if len(xy) > sample_points:
        idx = np.linspace(0, len(xy) - 1, sample_points).astype(int)
        xy = xy[idx]

    if not smx.exists():
        return None

    # PIN bbox prefilter: only test PTHs whose environment bbox plausibly
    # contains the telemetry positions (50 m margin). Falls back to all PTHs
    # if no PIN matches (e.g. test fixtures pointing at a custom dir).
    pins = _pins_for(smx)
    candidate_envs: set[str] | None = None
    if pins:
        x_med = float(np.median(xy[:, 0]))
        y_med = float(np.median(xy[:, 1]))
        candidate_envs = {
            env for env, info in pins.items()
            if info.contains_xy(x_med, y_med, margin_m=50.0)
        }
        if not candidate_envs:
            candidate_envs = None  # no env matched -> don't filter

    scores: list[tuple[str, float]] = []
    for pth_file in list_path_files(smx):
        if candidate_envs is not None:
            env_prefix = pth_file.stem[:2].upper()
            if env_prefix not in candidate_envs:
                continue
        try:
            prof = compute_profile(parse_pth(pth_file))
        except (OSError, ValueError):
            # Skip unreadable / malformed PTH files.
            continue
        if prof.pos.shape[0] < 2:
            continue
        tree = cKDTree(prof.pos[:, :2])
        dist, _ = tree.query(xy, k=1)
        scores.append((_normalize_variant(pth_file.stem), float(np.mean(dist))))

    if not scores:
        return None

    scores.sort(key=lambda t: t[1])
    best_name, best_score = scores[0]

    # Confidence:
    #   - "high" if absolute fit is excellent (positions land on the path),
    #     even when several variants share the same centerline (e.g. BL1/BL1R).
    #   - "high" when best is well separated from the runner-up.
    #   - "medium" when fit is ok and somewhat separated.
    #   - "low" otherwise.
    runner = scores[1][1] if len(scores) > 1 else best_score * 10
    ratio = runner / max(best_score, 1e-6)
    if best_score < 2.0 or (best_score < 5.0 and ratio > 3.0):
        confidence = "high"
    elif best_score < 25.0 and ratio > 1.5:
        confidence = "medium"
    else:
        confidence = "low"

    return TrackMatch(
        name=best_name, source="bbox",
        score=best_score, confidence=confidence,
        candidates=scores[:5],
    )
