"""Parser for LFS PTH (path) files in C:/LFS/data/smx/*.pth.

Format (reverse-engineered, validated against all 89 ship paths and
verified visually against the BL1 layout):

  Bytes 0..5   : ASCII magic 'SRPATH'
  Byte 6       : 0x00
  Byte 7       : version byte (e.g. 0xFC for current LFS)
  Bytes 8..55  : header — version dwords plus other metadata
                 (we surface as raw bytes; only magic + version + node count
                 are required to read the geometry).
  Bytes 56..   : N nodes of 44 bytes each.

Per-node layout (44 B), little-endian:
  bytes  0..3  : int32 — node flags / surface index (often 0; not decoded)
  bytes  4..7  : int32 centerline X * 65536   (Q16.16 fixed-point, metres)
  bytes  8..11 : int32 centerline Y * 65536   (Q16.16 fixed-point, metres)
  bytes 12..15 : int32 centerline Z * 65536   (Q16.16 fixed-point, metres)
  bytes 16..27 : 3 × float32 — direction unit tangent (dx, dy, dz)
  bytes 28..31 : float32 — outer track limit, LFS-left side  (always ≤ 0,
                 magnitude can reach hundreds of metres on tracks with
                 large infield run-off; “off-limits” boundary)
  bytes 32..35 : float32 — outer track limit, LFS-right side (always ≥ 0)
  bytes 36..39 : float32 — drive limit, LFS-left side  (always ≤ 0,
                 magnitude is the actual asphalt half-width to that side)
  bytes 40..43 : float32 — drive limit, LFS-right side (always ≥ 0)

Note: PTH does NOT encode banking. Edge values are scalar lateral
offsets in the horizontal plane perpendicular to the tangent; Z of the
edge is implicitly the centerline Z. Banking has to be recovered from
elsewhere (mesh files / OutSim car roll in steady-state cornering).

LFS coordinate system (matches OutSim pos_x/pos_y/pos_z):
  X = east (m), Y = north (m), Z = up (m).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import Iterable, Optional

import numpy as np

PTH_MAGIC = b"SRPATH"
HEADER_BYTES = 56
NODE_BYTES = 44
FIXED_POINT_DIVISOR = 65536.0  # Q16.16


@dataclass(slots=True, frozen=True)
class PthNode:
    """One sample along the centreline."""
    flags: int                  # bytes 0..3
    pos: np.ndarray             # shape (3,) — x, y, z (m)
    direction: np.ndarray       # shape (3,) unit tangent
    limit_left: float           # outer track limit, LFS-left side (≤ 0)
    limit_right: float          # outer track limit, LFS-right side (≥ 0)
    drive_left: float           # drivable asphalt edge, LFS-left side (≤ 0)
    drive_right: float          # drivable asphalt edge, LFS-right side (≥ 0)

    @property
    def width(self) -> float:
        """Total drivable width = ``drive_right - drive_left`` (metres)."""
        return float(self.drive_right - self.drive_left)


@dataclass(slots=True)
class Path:
    """Parsed PTH file."""
    name: str
    version: int
    raw_header: bytes
    nodes: list[PthNode]

    @property
    def pos(self) -> np.ndarray:
        if not self.nodes:
            return np.empty((0, 3))
        return np.array([n.pos for n in self.nodes])

    @property
    def direction(self) -> np.ndarray:
        if not self.nodes:
            return np.empty((0, 3))
        return np.array([n.direction for n in self.nodes])

    @property
    def drive_left(self) -> np.ndarray:
        if not self.nodes:
            return np.empty(0)
        return np.array([n.drive_left for n in self.nodes])

    @property
    def drive_right(self) -> np.ndarray:
        if not self.nodes:
            return np.empty(0)
        return np.array([n.drive_right for n in self.nodes])

    @property
    def limit_left(self) -> np.ndarray:
        if not self.nodes:
            return np.empty(0)
        return np.array([n.limit_left for n in self.nodes])

    @property
    def limit_right(self) -> np.ndarray:
        if not self.nodes:
            return np.empty(0)
        return np.array([n.limit_right for n in self.nodes])

    @property
    def width(self) -> np.ndarray:
        """Total drivable width per node (metres)."""
        if not self.nodes:
            return np.empty(0)
        return self.drive_right - self.drive_left

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)


def parse_pth(path: str | _Path) -> Path:
    """Parse a PTH file from disk."""
    p = _Path(path)
    data = p.read_bytes()
    return parse_pth_bytes(data, name=p.stem)


def parse_pth_bytes(data: bytes, name: str = "") -> Path:
    if len(data) < HEADER_BYTES:
        raise ValueError(f"PTH too short ({len(data)} bytes)")
    if data[:6] != PTH_MAGIC:
        raise ValueError(f"PTH bad magic: {data[:6]!r}")
    version = data[7]
    body = data[HEADER_BYTES:]
    if len(body) % NODE_BYTES != 0:
        raise ValueError(
            f"PTH body {len(body)} bytes is not a multiple of {NODE_BYTES}"
        )
    n_nodes = len(body) // NODE_BYTES
    nodes: list[PthNode] = []
    for i in range(n_nodes):
        off = i * NODE_BYTES
        flags, ix, iy, iz = struct.unpack_from("<i3i", body, off)
        dx, dy, dz = struct.unpack_from("<3f", body, off + 16)
        lim_l, lim_r, drv_l, drv_r = struct.unpack_from("<4f", body, off + 28)
        nodes.append(
            PthNode(
                flags=int(flags),
                pos=np.array(
                    [ix / FIXED_POINT_DIVISOR,
                     iy / FIXED_POINT_DIVISOR,
                     iz / FIXED_POINT_DIVISOR],
                    dtype=np.float64,
                ),
                direction=np.array([dx, dy, dz], dtype=np.float64),
                limit_left=float(lim_l),
                limit_right=float(lim_r),
                drive_left=float(drv_l),
                drive_right=float(drv_r),
            )
        )
    return Path(name=name, version=version,
                raw_header=data[:HEADER_BYTES], nodes=nodes)


# ----------------------------------------------------------------------------
# Derived geometric profile
# ----------------------------------------------------------------------------

@dataclass(slots=True)
class TrackProfile:
    """Per-node geometric properties derived from a Path."""
    name: str
    s: np.ndarray                  # cumulative arc-length (m)
    pos: np.ndarray                # (N, 3)
    direction: np.ndarray          # (N, 3) unit tangent
    slope_pct: np.ndarray          # 100 * dz/ds (positive = uphill)
    curvature_1_per_m: np.ndarray  # signed; positive = left turn
    radius_m: np.ndarray           # 1/|curvature|; clipped at 1e6 in straights
    heading_rad: np.ndarray        # atan2(dir_y, dir_x) in radians
    drive_left_m: np.ndarray       # drive edge, LFS-left  (≤ 0)
    drive_right_m: np.ndarray      # drive edge, LFS-right (≥ 0)
    limit_left_m: np.ndarray       # outer limit, LFS-left
    limit_right_m: np.ndarray      # outer limit, LFS-right
    # Optional enrichments populated from a parsed SMX mesh
    # (see ``lfs_telemetry.telemetry.track.geom3d.enrich_profile_with_smx``).
    banking_rad: Optional[np.ndarray] = None  # (N,) signed transverse slope
    surface_fractions: Optional[np.ndarray] = None  # (N, 5) per class
    surface_classes: Optional[tuple[str, ...]] = None  # names of the cols
    apex_visibility_m: Optional[np.ndarray] = None  # (N,) line-of-sight
    # 3D bounding / wall enrichments (BVH-style scan of the SMX mesh).
    # Distance from the centreline to the first non-drivable surface
    # (grass / runoff / other) on each side, in metres. ``effective_width_m``
    # = barrier_left_m + barrier_right_m. ``los_apex_m`` is the
    # line-of-sight distance to the next apex (mirrors ``apex_visibility_m``
    # but updated when wall-aware logic is in place).
    barrier_left_m: Optional[np.ndarray] = None     # (N,) positive
    barrier_right_m: Optional[np.ndarray] = None    # (N,) positive
    effective_width_m: Optional[np.ndarray] = None  # (N,) positive
    los_apex_m: Optional[np.ndarray] = None         # (N,) positive

    @property
    def width(self) -> np.ndarray:
        """Total drivable width per node = ``drive_right - drive_left``."""
        return self.drive_right_m - self.drive_left_m

    @property
    def total_length_m(self) -> float:
        return float(self.s[-1]) if self.s.size else 0.0

    @property
    def total_climb_m(self) -> float:
        if self.pos.shape[0] < 2:
            return 0.0
        dz = np.diff(self.pos[:, 2])
        return float(np.sum(np.clip(dz, 0.0, None)))

    @property
    def elevation_range_m(self) -> tuple[float, float]:
        if self.pos.size == 0:
            return (0.0, 0.0)
        return (float(self.pos[:, 2].min()), float(self.pos[:, 2].max()))


def compute_profile(path: Path, *, max_segment_m: float = 50.0) -> TrackProfile:
    """Derive arc length, slope, curvature, heading from a Path.

    PTH files often include the pit lane appended to the main racing line,
    separated by a large "teleport" segment.  `max_segment_m` truncates the
    path at the first inter-node segment longer than this threshold so the
    derived length and arc-distance reflect the racing line only.
    """
    if path.num_nodes < 2:
        return TrackProfile(
            name=path.name, s=np.empty(0), pos=np.empty((0, 3)),
            direction=np.empty((0, 3)), slope_pct=np.empty(0),
            curvature_1_per_m=np.empty(0), radius_m=np.empty(0),
            heading_rad=np.empty(0),
            drive_left_m=np.empty(0), drive_right_m=np.empty(0),
            limit_left_m=np.empty(0), limit_right_m=np.empty(0),
        )
    pos = path.pos
    dirs = path.direction
    drv_l = path.drive_left
    drv_r = path.drive_right
    lim_l = path.limit_left
    lim_r = path.limit_right

    # Detect a teleport-style discontinuity and cut there.
    deltas = np.diff(pos, axis=0)
    seg_all = np.linalg.norm(deltas, axis=1)
    big = np.where(seg_all > max_segment_m)[0]
    # Only cut if it leaves at least 5 usable nodes; otherwise the file is
    # likely an open / short path (autocross, drag) where every segment is
    # legitimate.
    if big.size and int(big[0]) + 1 >= 5:
        cut = int(big[0]) + 1
        pos = pos[:cut]
        dirs = dirs[:cut]
        drv_l = drv_l[:cut]
        drv_r = drv_r[:cut]
        lim_l = lim_l[:cut]
        lim_r = lim_r[:cut]
        deltas = deltas[: cut - 1]
        seg = seg_all[: cut - 1]
    else:
        seg = seg_all
    s = np.concatenate(([0.0], np.cumsum(seg)))

    # Force strictly increasing s for stable np.gradient.
    s_safe = s.copy()
    for i in range(1, len(s_safe)):
        if s_safe[i] <= s_safe[i - 1]:
            s_safe[i] = s_safe[i - 1] + 1e-3

    z = pos[:, 2]
    slope_pct = np.gradient(z, s_safe) * 100.0

    heading = np.arctan2(dirs[:, 1], dirs[:, 0])
    h_unw = np.unwrap(heading)
    curvature = np.gradient(h_unw, s_safe)
    abs_k = np.abs(curvature)
    radius = np.where(abs_k > 1e-6, 1.0 / np.maximum(abs_k, 1e-12), 1e6)

    return TrackProfile(
        name=path.name, s=s, pos=pos, direction=dirs,
        slope_pct=slope_pct, curvature_1_per_m=curvature,
        radius_m=radius, heading_rad=heading,
        drive_left_m=drv_l, drive_right_m=drv_r,
        limit_left_m=lim_l, limit_right_m=lim_r,
    )


# ----------------------------------------------------------------------------
# Discovery helpers
# ----------------------------------------------------------------------------

DEFAULT_SMX_DIR = _Path(r"C:\LFS\data\smx")


def list_path_files(smx_dir: _Path | str = DEFAULT_SMX_DIR) -> list[_Path]:
    """Return all *.pth files in the LFS smx directory, sorted."""
    return sorted(_Path(smx_dir).glob("*.pth"))


def load_all(smx_dir: _Path | str = DEFAULT_SMX_DIR) -> dict[str, Path]:
    """Load every PTH and return {variant_name: Path}."""
    out: dict[str, Path] = {}
    for f in list_path_files(smx_dir):
        try:
            out[f.stem] = parse_pth(f)
        except Exception as exc:  # noqa: BLE001
            out[f.stem] = exc  # type: ignore[assignment]
    return out


def summary_table(paths: Iterable[Path]) -> list[dict]:
    """Build a summary row per Path."""
    rows = []
    for p in paths:
        prof = compute_profile(p)
        zmin, zmax = prof.elevation_range_m
        rows.append({
            "variant": p.name,
            "nodes": p.num_nodes,
            "length_m": prof.total_length_m,
            "elev_min_m": zmin,
            "elev_max_m": zmax,
            "elev_delta_m": zmax - zmin,
            "total_climb_m": prof.total_climb_m,
            "min_radius_m": float(np.min(prof.radius_m))
                if prof.radius_m.size else 0.0,
            "max_slope_pct": float(np.max(np.abs(prof.slope_pct)))
                if prof.slope_pct.size else 0.0,
            "mean_width": float(np.mean(prof.width))
                if prof.width.size else 0.0,
        })
    return rows
