"""Parser for LFS SMX (Simple Mesh eXport) files in ``C:/LFS/data/smx``.

The format spec below is reproduced verbatim from the ``SMX.txt`` shipped
by LFS in its own ``data/smx`` directory:

  X, Y, Z fixed-point: ``1 m == 65536`` (Q16.16, signed int32).
  Vertex colours: ``ARGB`` (4 × byte = opacity, R, G, B).

  HEADER  (64 B)
    0   6 B  ASCII  ``LFSSMX``           — magic; reject if mismatch.
    6   1 B  game version                — informational.
    7   1 B  game revision               — informational.
    8   1 B  SMX version (must be 0)     — reject if higher.
    9   1 B  dimensions (must be 3)      — reject if != 3.
    10  1 B  resolution (0 high, 1 low)  — informational.
    11  1 B  vertex colours (must be 1)  — reject if 0.
    12  4 B  reserved (zeros).
    16 32 B  track text label.
    48  1 B  ground colour R.
    49  1 B  ground colour G.
    50  1 B  ground colour B.
    51  9 B  reserved.
    60  4 B  int32 — number of objects (N).

  OBJECT BLOCK header (24 B), repeated N times:
    0   4 B  int32  centre X (Q16.16).
    4   4 B  int32  centre Y.
    8   4 B  int32  centre Z.
    12  4 B  int32  radius (Q16.16).
    16  4 B  int32  num_points (P).
    20  4 B  int32  num_tris   (T).
    .. P × 16 B  point blocks  (X, Y, Z, ARGB).
    .. T ×  8 B  triangle blocks (a, b, c, _pad — uint16 each).

  FOOTER (after every object):
    0   4 B  int32  num_cp_objects (M).
    4   M × 4 B  int32  object indices of checkpoint objects.

The parser is stdlib + numpy only. No Qt / scipy import here.

Coordinate convention: world frame matches OutSim (X = east, Y = north,
Z = up), in metres after the Q16.16 divide.
"""
from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "SmxObject",
    "SmxMesh",
    "parse_smx",
    "DEFAULT_SMX_DIR",
    "list_smx_files",
    "find_smx_for_track",
    "elevation_envelope",
    "cross_section_at",
]

SMX_MAGIC = b"LFSSMX"
HEADER_BYTES = 64
OBJECT_HEADER_BYTES = 24
POINT_BYTES = 16
TRI_BYTES = 8
FIXED_POINT_DIVISOR = 65536.0  # Q16.16 — 1 m == 65536

# Numpy dtypes for bulk-decoding point and triangle blocks. Little-endian.
_POINT_DTYPE = np.dtype([
    ("x",  "<i4"),
    ("y",  "<i4"),
    ("z",  "<i4"),
    ("a",  "u1"),
    ("r",  "u1"),
    ("g",  "u1"),
    ("b",  "u1"),
])
_TRI_DTYPE = np.dtype([
    ("a",   "<u2"),
    ("b",   "<u2"),
    ("c",   "<u2"),
    ("pad", "<u2"),
])

# Default SMX directory — the live LFS install path. Tests and dev tools
# pass an explicit ``smx_dir`` when they need to point elsewhere.
DEFAULT_SMX_DIR = Path(r"C:\LFS\data\smx")


@dataclass(slots=True, frozen=True)
class SmxObject:
    """One object block from the SMX file.

    Each object has its own local vertex/triangle pool, but for ease of
    consumption :class:`SmxMesh` concatenates them and remembers each
    object's slice via ``vertex_start / vertex_end / tri_start / tri_end``.
    """
    index: int                  # 0-based object index within the file
    centre: np.ndarray          # shape (3,) — x, y, z (m)
    radius_m: float
    vertex_start: int           # first vertex index in SmxMesh.vertices
    vertex_end: int             # exclusive
    tri_start: int              # first triangle index in SmxMesh.triangles
    tri_end: int                # exclusive

    @property
    def num_points(self) -> int:
        return self.vertex_end - self.vertex_start

    @property
    def num_triangles(self) -> int:
        return self.tri_end - self.tri_start


@dataclass(slots=True)
class SmxMesh:
    """Parsed SMX file with global vertex / triangle pools.

    ``vertices`` and ``colors`` are aligned (one row per vertex).
    ``triangles`` indices are global (already offset across objects), so
    the mesh can be rendered as a single draw call.
    """
    name: str                       # file stem (e.g. "Blackwood_3DH")
    track_label: str                # the 32-char text from the header
    smx_version: int
    game_version: int
    game_revision: int
    resolution: int                 # 0 high, 1 low
    ground_rgb: tuple[int, int, int]
    vertices: np.ndarray            # shape (V, 3)  float64 (m)
    colors: np.ndarray              # shape (V, 4)  uint8 — A,R,G,B
    triangles: np.ndarray           # shape (T, 3)  uint32 — global indices
    objects: list[SmxObject] = field(default_factory=list)
    cp_object_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64))

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_triangles(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    def bounds_xyz(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(min_xyz, max_xyz)`` over all vertices."""
        if self.vertices.size == 0:
            zeros = np.zeros(3)
            return zeros, zeros
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def elevation_range_m(self) -> tuple[float, float]:
        """Min/max Z across all vertices (metres)."""
        lo, hi = self.bounds_xyz()
        return float(lo[2]), float(hi[2])


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_smx(path: str | Path) -> SmxMesh:
    """Parse an SMX file from disk."""
    p = Path(path)
    return parse_smx_bytes(p.read_bytes(), name=p.stem)


def parse_smx_bytes(data: bytes, name: str = "") -> SmxMesh:
    if len(data) < HEADER_BYTES:
        raise ValueError(f"SMX too short ({len(data)} B)")
    if data[:6] != SMX_MAGIC:
        raise ValueError(f"SMX bad magic: {data[:6]!r}")

    game_ver = data[6]
    game_rev = data[7]
    smx_ver = data[8]
    if smx_ver != 0:
        raise ValueError(f"SMX version {smx_ver} unsupported (expected 0)")
    dims = data[9]
    if dims != 3:
        raise ValueError(f"SMX dimensions {dims} != 3")
    resolution = data[10]
    vcolours = data[11]
    if vcolours != 1:
        raise ValueError(f"SMX vertex_colours {vcolours} != 1")

    track_label = data[16:48].split(b"\x00", 1)[0].decode(
        "latin-1", errors="replace")
    gr = data[48]
    gg = data[49]
    gb = data[50]
    num_objects = struct.unpack_from("<i", data, 60)[0]
    if num_objects < 0:
        raise ValueError(f"SMX negative num_objects: {num_objects}")

    # Walk objects, accumulating per-object vertex / triangle counts so we
    # can allocate the global pools once and copy block-by-block.
    cursor = HEADER_BYTES
    object_headers: list[tuple[int, int, int, int, int, int]] = []
    for _ in range(num_objects):
        if cursor + OBJECT_HEADER_BYTES > len(data):
            raise ValueError("SMX truncated inside object header")
        cx, cy, cz, rad, np_, nt = struct.unpack_from(
            "<iiiiII", data, cursor)
        object_headers.append((cx, cy, cz, rad, int(np_), int(nt)))
        cursor += OBJECT_HEADER_BYTES + np_ * POINT_BYTES + nt * TRI_BYTES
    if cursor > len(data):
        raise ValueError(
            f"SMX truncated: would need {cursor} B, have {len(data)}")

    # Total counts.
    total_v = sum(h[4] for h in object_headers)
    total_t = sum(h[5] for h in object_headers)
    vertices = np.empty((total_v, 3), dtype=np.float64)
    colors = np.empty((total_v, 4), dtype=np.uint8)
    triangles = np.empty((total_t, 3), dtype=np.uint32)
    objects: list[SmxObject] = []

    vcur = 0
    tcur = 0
    cursor = HEADER_BYTES
    for i, (cx, cy, cz, rad, np_, nt) in enumerate(object_headers):
        # Skip the object header.
        cursor += OBJECT_HEADER_BYTES

        # --- vertices ---
        if np_:
            pts = np.frombuffer(
                data, dtype=_POINT_DTYPE, count=np_, offset=cursor)
            vertices[vcur:vcur + np_, 0] = pts["x"] / FIXED_POINT_DIVISOR
            vertices[vcur:vcur + np_, 1] = pts["y"] / FIXED_POINT_DIVISOR
            vertices[vcur:vcur + np_, 2] = pts["z"] / FIXED_POINT_DIVISOR
            colors[vcur:vcur + np_, 0] = pts["a"]
            colors[vcur:vcur + np_, 1] = pts["r"]
            colors[vcur:vcur + np_, 2] = pts["g"]
            colors[vcur:vcur + np_, 3] = pts["b"]
        cursor += np_ * POINT_BYTES

        # --- triangles --- (local 0..np_-1 indices → offset by vcur)
        if nt:
            tris = np.frombuffer(
                data, dtype=_TRI_DTYPE, count=nt, offset=cursor)
            # Validate before offsetting so callers see clear errors.
            local_max = max(int(tris["a"].max()),
                            int(tris["b"].max()),
                            int(tris["c"].max())) if nt else -1
            if local_max >= np_:
                raise ValueError(
                    f"SMX object {i}: triangle index {local_max} "
                    f"out of range (num_points={np_})")
            triangles[tcur:tcur + nt, 0] = tris["a"].astype(np.uint32) + vcur
            triangles[tcur:tcur + nt, 1] = tris["b"].astype(np.uint32) + vcur
            triangles[tcur:tcur + nt, 2] = tris["c"].astype(np.uint32) + vcur
        cursor += nt * TRI_BYTES

        objects.append(SmxObject(
            index=i,
            centre=np.array([cx / FIXED_POINT_DIVISOR,
                             cy / FIXED_POINT_DIVISOR,
                             cz / FIXED_POINT_DIVISOR], dtype=np.float64),
            radius_m=float(rad) / FIXED_POINT_DIVISOR,
            vertex_start=vcur, vertex_end=vcur + np_,
            tri_start=tcur, tri_end=tcur + nt,
        ))
        vcur += np_
        tcur += nt

    # --- footer (checkpoint object indices) ---
    cp_indices = np.empty(0, dtype=np.int64)
    if cursor + 4 <= len(data):
        num_cp = struct.unpack_from("<i", data, cursor)[0]
        cursor += 4
        if num_cp < 0 or cursor + num_cp * 4 > len(data):
            # Tolerate a missing / malformed footer: many community-built
            # SMX exporters skip it. The mesh itself is already parsed.
            cp_indices = np.empty(0, dtype=np.int64)
        else:
            cp_indices = np.frombuffer(
                data, dtype="<i4", count=num_cp, offset=cursor
            ).astype(np.int64)

    return SmxMesh(
        name=name,
        track_label=track_label,
        smx_version=smx_ver,
        game_version=game_ver,
        game_revision=game_rev,
        resolution=resolution,
        ground_rgb=(int(gr), int(gg), int(gb)),
        vertices=vertices,
        colors=colors,
        triangles=triangles,
        objects=objects,
        cp_object_indices=cp_indices,
    )


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

def list_smx_files(smx_dir: str | Path | None = None) -> list[Path]:
    """List ``*.smx`` files in ``smx_dir`` (defaults to ``DEFAULT_SMX_DIR``)."""
    base = Path(smx_dir) if smx_dir else DEFAULT_SMX_DIR
    if not base.exists():
        return []
    return sorted(base.glob("*.smx"))


def find_smx_for_track(
    track: str,
    smx_dir: str | Path | None = None,
) -> Path | None:
    """Locate the SMX whose stem starts with the LFS track id.

    LFS official SMX filenames are e.g. ``Blackwood_3DH.smx`` — the prefix
    is the full track *name*, not the 3-char id. We match conservatively:

    * exact stem match
    * stem starts with the track id followed by ``_``
    * stem matches the human-readable env table (e.g. ``BL`` → ``Blackwood``).
    """
    base = Path(smx_dir) if smx_dir else DEFAULT_SMX_DIR
    if not base.exists():
        return None
    tid = track.strip().upper()
    candidates = sorted(base.glob("*.smx"))
    # Strict id-based match: filenames produced by ``scripts/track_view`` or
    # custom exporters often follow ``<id>.smx``.
    for p in candidates:
        stem = p.stem.upper()
        if stem == tid or stem.startswith(tid + "_"):
            return p
    # Fall back to the canonical LFS naming (env id → folder prefix).
    env_map = {
        "BL": "Blackwood",
        "SO": "South City",
        "FE": "Fern Bay",
        "AU": "Autocross",
        "KY": "Kyoto Ring",
        "WE": "Westhill",
        "AS": "Aston",
        "RO": "Rockingham",
        "LA": "Las Vegas",
    }
    env = env_map.get(tid[:2])
    if env is None:
        return None
    for p in candidates:
        if p.stem.lower().startswith(env.lower()):
            return p
    return None


# ---------------------------------------------------------------------------
# Silhouette / elevation helpers
# ---------------------------------------------------------------------------

def elevation_envelope(
    mesh: SmxMesh,
    centreline_xy: np.ndarray,
    s: np.ndarray,
    *,
    half_width_m: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project SMX vertices onto a centreline and return (z_lo, z_hi) bands.

    Parameters
    ----------
    mesh
        Parsed :class:`SmxMesh`.
    centreline_xy
        Shape ``(N, 2)`` array of centreline XY samples (metres).
    s
        Shape ``(N,)`` cumulative arc-length, aligned with ``centreline_xy``.
    half_width_m
        Vertices farther than this from the centreline (XY distance to
        the nearest node) are discarded — they belong to background
        scenery, not the track corridor.

    Returns
    -------
    (s_out, z_lo, z_hi)
        Three arrays of shape ``(N,)`` — the input ``s`` plus the
        min / max Z over the vertices binned to each station. Stations
        with no vertices in range inherit the previous bin (or NaN at the
        very start). ``z_lo`` and ``z_hi`` form a fill band suitable for
        a side-elevation silhouette plot.

    Implementation: a ``scipy.spatial.cKDTree`` over the centreline XY
    gives an O((V + N) log N) lookup. A linear-scan fallback runs when
    scipy is unavailable so the helper still works in minimal environments.
    """
    if centreline_xy.shape[1] != 2:
        raise ValueError("centreline_xy must be (N, 2)")
    if centreline_xy.shape[0] != s.shape[0]:
        raise ValueError("centreline_xy and s must have the same length")
    n = centreline_xy.shape[0]
    if mesh.vertices.size == 0 or n == 0:
        nan_arr = np.full(n, np.nan)
        return s, nan_arr, nan_arr.copy()

    verts = mesh.vertices
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(centreline_xy)
        dist, idx = tree.query(verts[:, :2], k=1)
    except ImportError:                         # pragma: no cover
        # Vectorised fallback: O(V × N). Acceptable for N < ~1000.
        d2 = ((verts[:, None, 0] - centreline_xy[None, :, 0]) ** 2
              + (verts[:, None, 1] - centreline_xy[None, :, 1]) ** 2)
        idx = d2.argmin(axis=1)
        dist = np.sqrt(d2[np.arange(verts.shape[0]), idx])

    mask = dist <= half_width_m
    z_lo = np.full(n, np.nan)
    z_hi = np.full(n, np.nan)
    if not mask.any():
        return s, z_lo, z_hi
    bin_idx = idx[mask].astype(np.int64)
    z_vals = verts[mask, 2]
    # Use +inf / -inf as reduction sentinels so np.minimum.at / maximum.at
    # work correctly (np.minimum propagates NaN, which would corrupt the
    # accumulator). Translate untouched bins back to NaN at the end.
    lo_acc = np.full(n, np.inf)
    hi_acc = np.full(n, -np.inf)
    np.minimum.at(lo_acc, bin_idx, z_vals)
    np.maximum.at(hi_acc, bin_idx, z_vals)
    touched = np.isfinite(lo_acc)
    z_lo[touched] = lo_acc[touched]
    z_hi[touched] = hi_acc[touched]
    # No gap interpolation here — callers can use pandas if they want a
    # solid band: ``pd.Series(z_lo).interpolate().to_numpy()``.
    return s, z_lo, z_hi


def cross_section_at(
    mesh: SmxMesh,
    centre_xy: np.ndarray,
    tangent_xy: np.ndarray,
    *,
    half_width_m: float = 15.0,
    slice_thickness_m: float = 2.0,
) -> np.ndarray:
    """Return the SMX cross-section at a given station as ``(t, z)`` pairs.

    A slab of thickness ``slice_thickness_m`` perpendicular to ``tangent_xy``
    is intersected with the mesh; vertices inside the slab and within
    ``half_width_m`` laterally are returned, sorted by lateral offset ``t``
    (positive = LFS-right). This is the building block for banking analysis.
    """
    if mesh.vertices.size == 0:
        return np.empty((0, 2))
    tx, ty = float(tangent_xy[0]), float(tangent_xy[1])
    norm = (tx * tx + ty * ty) ** 0.5
    if norm < 1e-9:
        return np.empty((0, 2))
    tx /= norm
    ty /= norm
    # Right-hand normal in the XY plane (rotate tangent -90°).
    nx, ny = ty, -tx
    dx = mesh.vertices[:, 0] - float(centre_xy[0])
    dy = mesh.vertices[:, 1] - float(centre_xy[1])
    s_along = dx * tx + dy * ty
    t_perp = dx * nx + dy * ny
    mask = (np.abs(s_along) <= 0.5 * slice_thickness_m) \
        & (np.abs(t_perp) <= half_width_m)
    if not mask.any():
        return np.empty((0, 2))
    out = np.column_stack((t_perp[mask], mesh.vertices[mask, 2]))
    order = np.argsort(out[:, 0])
    return out[order]


def iter_smx_directory(
    smx_dir: str | Path | None = None,
) -> Iterable[tuple[Path, SmxMesh]]:
    """Yield ``(path, mesh)`` for every parseable SMX in ``smx_dir``."""
    for p in list_smx_files(smx_dir):
        try:
            yield p, parse_smx(p)
        except (OSError, ValueError):
            continue
