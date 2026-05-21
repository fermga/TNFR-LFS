"""Per-row lookup of track geometry from a precomputed racing_lines CSV.

Loads ``racing_lines/<TRACK>_racing.csv`` (produced by
``scripts/racing_line_view.py``) and exposes a KDTree over the
centerline so any (pos_x, pos_y) sample can be projected to its
nearest centerline node.

Used by :mod:`lfs_telemetry.telemetry.derived` to add slope-corrected
longitudinal acceleration, yaw-misalignment and per-segment columns to
the enriched DataFrame whenever a racing-line file exists for the
captured track.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ...app_paths import (
    candidate_racing_lines_dirs as _candidate_racing_lines_dirs,
)

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filesystem discovery
# ---------------------------------------------------------------------------


def candidate_racing_lines_dirs() -> list[Path]:
    """Return every plausible directory holding ``<TRACK>_racing.csv`` files.

    Thin wrapper around :func:`lfs_telemetry.app_paths.candidate_racing_lines_dirs`,
    kept for backwards compatibility with the public import path.
    """
    return _candidate_racing_lines_dirs()


def find_racing_line_csv(track: str) -> Path | None:
    """Locate ``<dir>/<TRACK>_racing.csv`` under any candidate dir."""
    if not track:
        return None
    name = f"{track.upper()}_racing.csv"
    for base in candidate_racing_lines_dirs():
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# TrackGeometry
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrackGeometry:
    """All per-node columns of a racing_lines CSV plus a KDTree over (X, Y)."""
    track: str
    s_m: np.ndarray              # (N,)
    xy: np.ndarray               # (N, 2)
    z_m: np.ndarray
    heading_rad: np.ndarray
    curvature_1_per_m: np.ndarray
    radius_m: np.ndarray
    slope_pct: np.ndarray
    width_m: np.ndarray
    segment_id: np.ndarray       # int64
    segment_kind: np.ndarray     # object (strings)
    _tree: cKDTree

    @property
    def num_nodes(self) -> int:
        return self.s_m.size

    def lookup(self, pos_x, pos_y) -> dict[str, np.ndarray]:
        """Return nearest-node values for an array-like of (x, y) samples.

        Output dict keys: ``track_node``, ``track_s_m``, ``track_z_m``,
        ``track_heading_rad``, ``track_curvature_1_per_m``,
        ``track_radius_m``, ``track_slope_pct``, ``track_width_m``,
        ``segment_id``, ``segment_kind``, ``track_offset_m``.
        """
        x = np.asarray(pos_x, dtype=float)
        y = np.asarray(pos_y, dtype=float)
        pts = np.column_stack([x, y])
        dist, idx = self._tree.query(pts, k=1)
        idx = idx.astype(np.int64)
        return {
            "track_node": idx,
            "track_s_m": self.s_m[idx],
            "track_z_m": self.z_m[idx],
            "track_heading_rad": self.heading_rad[idx],
            "track_curvature_1_per_m": self.curvature_1_per_m[idx],
            "track_radius_m": self.radius_m[idx],
            "track_slope_pct": self.slope_pct[idx],
            "track_width_m": self.width_m[idx],
            "segment_id": self.segment_id[idx],
            "segment_kind": self.segment_kind[idx],
            "track_offset_m": dist.astype(np.float64),
        }


def load_track_geometry(track: str, *,
                        path: Path | str | None = None) -> TrackGeometry | None:
    """Load ``<TRACK>_racing.csv`` and return a :class:`TrackGeometry`.

    Returns ``None`` if no CSV is found, the file is empty or any
    required column is missing (callers treat that as "no geometry
    available — skip track-derived columns").
    """
    if path is None:
        path = find_racing_line_csv(track)
        if path is None:
            return None
    p = Path(path)
    try:
        df = pd.read_csv(p)
    except Exception as exc:
        _LOG.warning("could not read %s: %s", p, exc)
        return None
    required = (
        "x_center_m", "y_center_m", "s_m", "z_center_m",
        "heading_rad", "curvature_1_per_m", "radius_m",
        "slope_pct", "width_m", "segment_id", "segment_kind",
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        _LOG.info("racing line %s lacks columns %s; "
                  "regenerate with scripts/racing_line_view.py --all",
                  p.name, missing)
        return None
    if len(df) < 2:
        return None
    xy = df[["x_center_m", "y_center_m"]].to_numpy(dtype=float)
    return TrackGeometry(
        track=(track or p.stem.replace("_racing", "")).upper(),
        s_m=df["s_m"].to_numpy(dtype=float),
        xy=xy,
        z_m=df["z_center_m"].to_numpy(dtype=float),
        heading_rad=df["heading_rad"].to_numpy(dtype=float),
        curvature_1_per_m=df["curvature_1_per_m"].to_numpy(dtype=float),
        radius_m=df["radius_m"].to_numpy(dtype=float),
        slope_pct=df["slope_pct"].to_numpy(dtype=float),
        width_m=df["width_m"].to_numpy(dtype=float),
        segment_id=df["segment_id"].to_numpy(dtype=np.int64),
        segment_kind=df["segment_kind"].fillna("").astype(str).to_numpy(),
        _tree=cKDTree(xy),
    )


# Module-level cache so that repeated calls (lap reloads, comparisons) don't
# re-read and re-tree the same file. Keyed by absolute path.
_CACHE: dict[Path, TrackGeometry | None] = {}


def cached_track_geometry(track: str) -> TrackGeometry | None:
    """Memoized variant of :func:`load_track_geometry` keyed by track code."""
    if not track:
        return None
    csv = find_racing_line_csv(track)
    if csv is None:
        return None
    key = csv.resolve()
    if key not in _CACHE:
        _CACHE[key] = load_track_geometry(track, path=csv)
    return _CACHE[key]


__all__ = [
    "TrackGeometry",
    "cached_track_geometry",
    "candidate_racing_lines_dirs",
    "find_racing_line_csv",
    "load_track_geometry",
]
