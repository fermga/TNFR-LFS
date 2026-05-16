"""Track geometry features derived from a parsed SMX mesh.

The PTH file gives us the centreline plus *scalar horizontal* drive
limits. The SMX mesh adds the real 3D geometry of the track surface
(per-vertex ARGB colour, grouped into objects, checkpoint object
indices in the footer). From those two sources combined we can compute:

* **banking** — transverse slope of the asphalt cross-section
  (radians, signed: positive = peraltado hacia LFS-right).
* **surface classification per vertex** — ``asphalt`` / ``kerb`` /
  ``runoff`` / ``grass`` / ``other`` from an RGB heuristic palette.
* **surface fractions per station** — ``(N_s, 5)`` array summing to 1
  in the asphalt corridor; lets us flag kerb usage and validate PTH.
* **checkpoint geometry** — convex-hull polygons of the actual physics
  checkpoint objects, ready to drive sector splitting from ground truth.
* **corridor heightmap** — regular ``(N_s, n_t)`` Z grid of the track
  corridor, persistable as ``.npz`` for fast lookup at runtime.
* **apex visibility distance** — line-of-sight clipping over crests
  along the centreline (1D, no mesh needed beyond the PTH Z).

All helpers are stdlib + numpy + scipy. No Qt, no pandas.

The classifier palette is a heuristic calibrated against the seven
shipped LFS environments. It is documented at the top of
:func:`classify_surface` and is intentionally generous on the asphalt
side (low-saturation greys) so the banking fit gets enough points.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .smx import SmxMesh, cross_section_at
from .pth import TrackProfile

__all__ = [
    "SURFACE_CLASSES",
    "SURFACE_INDEX",
    "classify_surface",
    "compute_banking_profile",
    "surface_distribution_along",
    "kerb_mask_along",
    "CheckpointGeom",
    "extract_checkpoint_geometry",
    "CorridorHeightmap",
    "corridor_heightmap",
    "apex_visibility_distance",
    "enrich_profile_with_smx",
    "compute_barrier_offsets",
]

# ---------------------------------------------------------------------------
# Surface classification
# ---------------------------------------------------------------------------

SURFACE_CLASSES: tuple[str, ...] = (
    "asphalt", "kerb", "runoff", "grass", "other",
)
SURFACE_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(SURFACE_CLASSES)
}


def classify_surface(mesh: SmxMesh) -> np.ndarray:
    """Classify every SMX vertex by surface type from its RGB colour.

    Returns ``np.ndarray`` of shape ``(V,)`` and dtype ``uint8`` with
    indices into :data:`SURFACE_CLASSES`.

    LFS uses two different effective palettes:

    * **Real shipped tracks** — Blue channel is universally pinned to 255
      and the surface class is encoded in ``(R, G)``: greyscale-ish
      ``R ≈ G`` rows are tarmac (lighting/normal shading varies the
      brightness), ``R >> G`` is red kerb, ``G >> R`` is grass, and
      ``R = G = 255`` is white concrete / pit lane.
    * **Standalone / synthetic** colours (used in our tests and ad-hoc
      meshes) follow the obvious RGB intuition — saturated red is a
      kerb, saturated green is grass, near-grey is asphalt, etc.

    The classifier dispatches per-vertex on ``B >= 240`` to pick the
    appropriate rule set, so both palettes work in one call.
    """
    if mesh.colors.size == 0:
        return np.empty(0, dtype=np.uint8)
    rgb = mesh.colors[:, 1:4].astype(np.int16)  # drop alpha
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    out = np.full(rgb.shape[0], SURFACE_INDEX["other"], dtype=np.uint8)

    lfs = b >= 240          # real LFS shipped meshes use B = 255 as sentinel
    syn = ~lfs

    # --- LFS real-track palette (B = 255) ---
    lfs_white = lfs & (r >= 240) & (g >= 240)
    lfs_red_kerb = lfs & ((r - g) >= 60) & (r >= 150) & ~lfs_white
    lfs_grass = lfs & ((g - r) >= 30) & (g >= 80) & ~lfs_white
    lfs_runoff = lfs & ((r - g) >= 20) & ((r - g) < 60) & (r >= 150) \
        & ~lfs_white & ~lfs_red_kerb
    lfs_asphalt = lfs & (np.abs(r - g) <= 25) & ~lfs_white

    # --- Synthetic / standalone palette ---
    syn_red_kerb = syn & (r >= 170) & (g <= 90) & (b <= 90)
    syn_white_kerb = syn & (r >= 210) & (g >= 210) & (b >= 210)
    syn_grass = syn & (g > r + 15) & (g > b + 15) & (g >= 80)
    syn_runoff = syn & (r >= 140) & (g >= 110) & (g <= 200) & (b <= r - 30)
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    syn_asphalt = syn & ((cmax - cmin) <= 25) & (cmax <= 160) \
        & ~syn_red_kerb & ~syn_white_kerb & ~syn_grass & ~syn_runoff

    out[lfs_asphalt | syn_asphalt] = SURFACE_INDEX["asphalt"]
    out[lfs_runoff | syn_runoff] = SURFACE_INDEX["runoff"]
    out[lfs_grass | syn_grass] = SURFACE_INDEX["grass"]
    out[lfs_red_kerb | lfs_white | syn_red_kerb | syn_white_kerb] = \
        SURFACE_INDEX["kerb"]
    return out


# ---------------------------------------------------------------------------
# Banking from cross-sections
# ---------------------------------------------------------------------------

def _centreline_tangents(centreline_xy: np.ndarray) -> np.ndarray:
    """Return unit tangents per centreline node by central difference."""
    n = centreline_xy.shape[0]
    if n < 2:
        return np.zeros((n, 2))
    t = np.empty_like(centreline_xy)
    t[1:-1] = centreline_xy[2:] - centreline_xy[:-2]
    t[0] = centreline_xy[1] - centreline_xy[0]
    t[-1] = centreline_xy[-1] - centreline_xy[-2]
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    norm[norm < 1e-9] = 1.0
    return t / norm


def compute_banking_profile(
    centreline_xyz: np.ndarray,
    mesh: SmxMesh,
    classes: np.ndarray | None = None,
    *,
    half_width_m: float = 12.0,
    slice_thickness_m: float = 3.0,
    min_points: int = 6,
) -> np.ndarray:
    """Return banking angle (rad) per centreline station.

    Banking = transverse slope of the asphalt cross-section, signed so
    that positive values mean the surface tilts up toward LFS-right
    (the same sign convention as PTH ``drive_right`` offsets).

    Parameters
    ----------
    centreline_xyz
        Shape ``(N, 3)`` — the PTH centreline (e.g. ``profile.pos``).
    mesh
        Parsed :class:`SmxMesh`.
    classes
        Optional precomputed output of :func:`classify_surface`. If
        ``None``, it is computed here.
    half_width_m
        Lateral half-width used to slice the cross-section.
    slice_thickness_m
        Longitudinal thickness of the slab perpendicular to the tangent.
    min_points
        Minimum asphalt vertices required for the fit; stations with
        fewer return ``NaN``.

    Returns
    -------
    np.ndarray
        Shape ``(N,)`` — banking in radians. ``NaN`` where the fit
        could not be performed (no asphalt found near that station).
    """
    n = centreline_xyz.shape[0]
    out = np.full(n, np.nan)
    if mesh.vertices.size == 0 or n == 0:
        return out

    if classes is None:
        classes = classify_surface(mesh)
    asphalt_mask = (classes == SURFACE_INDEX["asphalt"])
    if not asphalt_mask.any():
        return out
    asphalt_verts = mesh.vertices[asphalt_mask]
    shim_mesh = SmxMesh(
        name=mesh.name, track_label=mesh.track_label,
        smx_version=mesh.smx_version, game_version=mesh.game_version,
        game_revision=mesh.game_revision, resolution=mesh.resolution,
        ground_rgb=mesh.ground_rgb,
        vertices=asphalt_verts,
        colors=mesh.colors[asphalt_mask],
        triangles=np.empty((0, 3), dtype=np.uint32),
        objects=[], cp_object_indices=np.empty(0, dtype=np.int64),
    )

    tangents = _centreline_tangents(centreline_xyz[:, :2])
    for i in range(n):
        cs = cross_section_at(
            shim_mesh,
            centreline_xyz[i, :2],
            tangents[i],
            half_width_m=half_width_m,
            slice_thickness_m=slice_thickness_m,
        )
        if cs.shape[0] < min_points:
            continue
        # np.polyfit returns highest-order coeff first → slope = coeff[0].
        slope, _intercept = np.polyfit(cs[:, 0], cs[:, 1], 1)
        out[i] = float(np.arctan(slope))
    return out


# ---------------------------------------------------------------------------
# Surface distribution and kerb mask along the centreline
# ---------------------------------------------------------------------------

def _nearest_station_index(
    centreline_xy: np.ndarray,
    points_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(idx, dist)`` of the nearest centreline node per point."""
    try:
        from scipy.spatial import cKDTree   # type: ignore
        tree = cKDTree(centreline_xy)
        dist, idx = tree.query(points_xy, k=1)
        return idx.astype(np.int64), dist
    except ImportError:                                  # pragma: no cover
        d2 = ((points_xy[:, None, 0] - centreline_xy[None, :, 0]) ** 2
              + (points_xy[:, None, 1] - centreline_xy[None, :, 1]) ** 2)
        idx = d2.argmin(axis=1).astype(np.int64)
        dist = np.sqrt(d2[np.arange(points_xy.shape[0]), idx])
        return idx, dist


def surface_distribution_along(
    centreline_xy: np.ndarray,
    mesh: SmxMesh,
    classes: np.ndarray | None = None,
    *,
    half_width_m: float = 12.0,
) -> np.ndarray:
    """Per-station fraction of vertices belonging to each surface class.

    Returns shape ``(N, len(SURFACE_CLASSES))``. Each row sums to 1 where
    at least one vertex lay within ``half_width_m`` of that station; rows
    with no hits are all zero.
    """
    n = centreline_xy.shape[0]
    out = np.zeros((n, len(SURFACE_CLASSES)), dtype=np.float64)
    if mesh.vertices.size == 0 or n == 0:
        return out
    if classes is None:
        classes = classify_surface(mesh)
    idx, dist = _nearest_station_index(centreline_xy, mesh.vertices[:, :2])
    mask = dist <= half_width_m
    if not mask.any():
        return out
    bin_idx = idx[mask]
    cls_idx = classes[mask].astype(np.int64)
    # Vectorised bincount per class.
    for ci in range(len(SURFACE_CLASSES)):
        m = cls_idx == ci
        if not m.any():
            continue
        counts = np.bincount(bin_idx[m], minlength=n)
        out[:, ci] = counts
    totals = out.sum(axis=1, keepdims=True)
    nz = totals[:, 0] > 0
    out[nz] = out[nz] / totals[nz]
    return out


def kerb_mask_along(
    centreline_xy: np.ndarray,
    mesh: SmxMesh,
    classes: np.ndarray | None = None,
    *,
    side: int,
    half_width_m: float = 12.0,
    min_kerb_vertices: int = 3,
) -> np.ndarray:
    """Boolean ``(N,)`` mask: kerb present on the given side per station.

    ``side`` is ``-1`` for LFS-left and ``+1`` for LFS-right (matches
    the sign of the lateral offset ``t`` returned by
    :func:`cross_section_at`).
    """
    if side not in (-1, +1):
        raise ValueError("side must be -1 (left) or +1 (right)")
    n = centreline_xy.shape[0]
    out = np.zeros(n, dtype=bool)
    if mesh.vertices.size == 0 or n == 0:
        return out
    if classes is None:
        classes = classify_surface(mesh)
    kerb_mask_v = classes == SURFACE_INDEX["kerb"]
    if not kerb_mask_v.any():
        return out
    kerb_xy = mesh.vertices[kerb_mask_v, :2]
    idx, dist = _nearest_station_index(centreline_xy, kerb_xy)
    keep = dist <= half_width_m
    if not keep.any():
        return out

    # Determine the lateral offset sign by projecting onto the right normal.
    tangents = _centreline_tangents(centreline_xy)
    bin_idx = idx[keep]
    pts = kerb_xy[keep]
    centre_pts = centreline_xy[bin_idx]
    tan = tangents[bin_idx]
    # right-hand normal of the 2D tangent: (ty, -tx)
    nrm = np.column_stack((tan[:, 1], -tan[:, 0]))
    d = pts - centre_pts
    t_perp = d[:, 0] * nrm[:, 0] + d[:, 1] * nrm[:, 1]
    side_mask = (t_perp > 0) if side > 0 else (t_perp < 0)
    if not side_mask.any():
        return out
    counts = np.bincount(bin_idx[side_mask], minlength=n)
    out = counts >= min_kerb_vertices
    return out


# ---------------------------------------------------------------------------
# Checkpoint geometry
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class CheckpointGeom:
    """3D geometry of one LFS physics checkpoint object."""
    object_index: int
    centre: np.ndarray              # (3,)
    polygon_xy: np.ndarray          # (K, 2) convex hull, CCW
    normal_xy: np.ndarray           # (2,) unit, principal axis of the polygon
    half_width_m: float             # half of the major-axis extent


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain, returns CCW hull (no duplicate endpoint)."""
    if points.shape[0] <= 2:
        return points

    def _cross2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    pts = points[np.lexsort((points[:, 1], points[:, 0]))]
    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and _cross2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull)


def extract_checkpoint_geometry(mesh: SmxMesh) -> list[CheckpointGeom]:
    """Return one :class:`CheckpointGeom` per checkpoint object in the SMX.

    The SMX footer lists checkpoint object indices. For each, we take
    all of its XY vertices, compute the 2D convex hull, run a tiny PCA
    to get the principal axis (= the gate direction), and report the
    centre and half-width along that axis.
    """
    out: list[CheckpointGeom] = []
    if mesh.cp_object_indices.size == 0:
        return out
    for cp_idx in mesh.cp_object_indices.tolist():
        if cp_idx < 0 or cp_idx >= len(mesh.objects):
            continue
        obj = mesh.objects[cp_idx]
        if obj.num_points == 0:
            continue
        v = mesh.vertices[obj.vertex_start:obj.vertex_end]
        xy = v[:, :2]
        hull = _convex_hull_2d(xy)
        # 2D PCA on the hull to recover the principal axis.
        centred = xy - xy.mean(axis=0)
        cov = centred.T @ centred / max(1, centred.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # eigh returns ascending; principal axis = last column.
        axis = eigvecs[:, -1]
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        projections = centred @ axis
        half_width = float(0.5 * (projections.max() - projections.min()))
        out.append(CheckpointGeom(
            object_index=cp_idx,
            centre=obj.centre.copy(),
            polygon_xy=hull,
            normal_xy=axis,
            half_width_m=half_width,
        ))
    return out


# ---------------------------------------------------------------------------
# Corridor heightmap
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CorridorHeightmap:
    """Regular ``(N_s, n_t)`` Z grid over the track corridor.

    ``s`` is the centreline arc-length axis (m). ``t`` is the lateral
    offset axis (m, positive = LFS-right). ``z`` is altitude (m).
    NaN where no SMX asphalt vertex was found nearby.
    """
    s: np.ndarray            # (N_s,)
    t: np.ndarray            # (n_t,)
    z: np.ndarray            # (N_s, n_t)
    half_width_m: float

    def save_npz(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p, s=self.s, t=self.t, z=self.z,
            half_width_m=np.array([self.half_width_m]),
        )
        return p

    @classmethod
    def load_npz(cls, path: str | Path) -> "CorridorHeightmap":
        data = np.load(Path(path))
        return cls(
            s=data["s"], t=data["t"], z=data["z"],
            half_width_m=float(data["half_width_m"][0]),
        )


def corridor_heightmap(
    centreline_xyz: np.ndarray,
    s: np.ndarray,
    mesh: SmxMesh,
    classes: np.ndarray | None = None,
    *,
    n_t: int = 21,
    half_width_m: float = 12.0,
    slice_thickness_m: float = 4.0,
    surface_filter: tuple[str, ...] = ("asphalt", "kerb"),
) -> CorridorHeightmap:
    """Sample the SMX surface on a regular ``(N_s, n_t)`` corridor grid.

    For each station we slice the mesh perpendicular to the tangent,
    keep vertices belonging to the requested surface classes, and
    interpolate Z linearly onto ``t ∈ linspace(-hw, +hw, n_t)``.
    Stations with too few hits return a row of NaNs.
    """
    if centreline_xyz.shape[0] != s.shape[0]:
        raise ValueError("centreline_xyz and s must have the same length")
    n = s.shape[0]
    t_grid = np.linspace(-half_width_m, half_width_m, n_t)
    z = np.full((n, n_t), np.nan)
    if mesh.vertices.size == 0 or n == 0:
        return CorridorHeightmap(s=s, t=t_grid, z=z, half_width_m=half_width_m)

    if classes is None:
        classes = classify_surface(mesh)
    keep_idx = np.zeros_like(classes, dtype=bool)
    for cname in surface_filter:
        keep_idx |= classes == SURFACE_INDEX[cname]
    if not keep_idx.any():
        return CorridorHeightmap(s=s, t=t_grid, z=z, half_width_m=half_width_m)
    sub_verts = mesh.vertices[keep_idx]
    sub_mesh = SmxMesh(
        name=mesh.name, track_label=mesh.track_label,
        smx_version=mesh.smx_version, game_version=mesh.game_version,
        game_revision=mesh.game_revision, resolution=mesh.resolution,
        ground_rgb=mesh.ground_rgb, vertices=sub_verts,
        colors=mesh.colors[keep_idx],
        triangles=np.empty((0, 3), dtype=np.uint32),
        objects=[], cp_object_indices=np.empty(0, dtype=np.int64),
    )

    tangents = _centreline_tangents(centreline_xyz[:, :2])
    for i in range(n):
        cs = cross_section_at(
            sub_mesh, centreline_xyz[i, :2], tangents[i],
            half_width_m=half_width_m,
            slice_thickness_m=slice_thickness_m,
        )
        if cs.shape[0] < 2:
            continue
        # Sort by t and interpolate Z on the regular grid (clip outside).
        order = np.argsort(cs[:, 0])
        t_sorted = cs[order, 0]
        z_sorted = cs[order, 1]
        # Drop duplicate t to keep np.interp happy.
        keep = np.concatenate(([True], np.diff(t_sorted) > 1e-6))
        t_sorted = t_sorted[keep]
        z_sorted = z_sorted[keep]
        if t_sorted.size < 2:
            continue
        mask = (t_grid >= t_sorted[0]) & (t_grid <= t_sorted[-1])
        if mask.any():
            z[i, mask] = np.interp(t_grid[mask], t_sorted, z_sorted)
    return CorridorHeightmap(s=s, t=t_grid, z=z, half_width_m=half_width_m)


# ---------------------------------------------------------------------------
# Apex visibility distance (1D, centreline-only)
# ---------------------------------------------------------------------------

def compute_barrier_offsets(
    centreline_xy: np.ndarray,
    tangents_xy: np.ndarray,
    mesh: SmxMesh,
    classes: np.ndarray | None = None,
    *,
    max_search_m: float = 40.0,
    step_m: float = 0.5,
    sample_radius_m: float = 1.5,
    drivable: tuple[str, ...] = ("asphalt", "kerb"),
) -> tuple[np.ndarray, np.ndarray]:
    """Distance to the first non-drivable surface on each side.

    For every centreline node, march in perpendicular steps (left and
    right of the travel direction) and classify each sample by the
    nearest mesh vertex within ``sample_radius_m``. The first step
    whose nearest-vertex class falls outside ``drivable`` (i.e. grass /
    runoff / other / no nearby surface) sets the barrier offset on
    that side. Returns ``(left_m, right_m)`` — both positive distances
    in metres, capped at ``max_search_m``.

    This is the 2D analogue of a BVH ray-cast: instead of intersecting
    triangles, we sample the classified vertex cloud at fixed steps.
    For tracks where the wall starts a few metres beyond the runoff
    (e.g. tyre stacks) the returned value is conservative — it stops
    at the grass edge — which is exactly what a driver perceives as
    "track width".
    """
    n = centreline_xy.shape[0]
    left = np.full(n, max_search_m, dtype=np.float64)
    right = np.full(n, max_search_m, dtype=np.float64)
    if n == 0 or mesh.vertices.size == 0:
        return left, right
    if classes is None:
        classes = classify_surface(mesh)
    drivable_idx = {SURFACE_INDEX[c] for c in drivable if c in SURFACE_INDEX}
    # KDTree over mesh vertices for O(log V) nearest-neighbour lookup.
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(mesh.vertices[:, :2])
    except ImportError:  # pragma: no cover
        tree = None
    # Build (S, 2) array of right-hand normals (LFS-right is +).
    nrm = np.column_stack((tangents_xy[:, 1], -tangents_xy[:, 0]))
    nrm_norm = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, np.where(nrm_norm > 1e-9, nrm_norm, 1.0))
    n_steps = int(max_search_m / step_m)
    if n_steps <= 0:
        return left, right
    # Pre-build sample distances 1..n_steps.
    step_dists = np.arange(1, n_steps + 1, dtype=np.float64) * step_m
    for side, out in ((+1, right), (-1, left)):
        # Offsets shape (N, n_steps, 2): centre + side * step * normal.
        # Process in chunks to limit peak memory on long tracks.
        chunk = 256
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            c = centreline_xy[i0:i1][:, None, :]
            nv = nrm[i0:i1][:, None, :]
            samples = c + side * step_dists[None, :, None] * nv
            flat = samples.reshape(-1, 2)
            if tree is not None:
                dist, idx = tree.query(flat, k=1)
            else:  # pragma: no cover
                d2 = ((flat[:, None, 0] - mesh.vertices[None, :, 0]) ** 2
                      + (flat[:, None, 1] - mesh.vertices[None, :, 1]) ** 2)
                idx = d2.argmin(axis=1)
                dist = np.sqrt(d2[np.arange(flat.shape[0]), idx])
            cls = classes[idx]
            is_drivable = np.isin(cls, list(drivable_idx))
            far = dist > sample_radius_m
            # A sample is "off the drivable surface" if either no nearby
            # vertex was found OR the nearest vertex is non-drivable.
            off = (~is_drivable) | far
            off = off.reshape(i1 - i0, n_steps)
            # First True along each row sets the barrier distance.
            first = np.argmax(off, axis=1)
            any_off = off.any(axis=1)
            for k, (hit, row) in enumerate(zip(any_off, first)):
                if hit:
                    out[i0 + k] = float(step_dists[row])
    return left, right


def apex_visibility_distance(
    s: np.ndarray,
    z: np.ndarray,
    *,
    eye_height_m: float = 1.0,
    look_height_m: float = 0.5,
    max_lookahead_m: float = 800.0,
) -> np.ndarray:
    """Return the line-of-sight distance from each station to the next crest.

    Models the driver's eye at ``z[i] + eye_height_m`` looking at a
    target at ``z[j] + look_height_m``. The straight line between them
    must clear every intermediate Z by ``>= 0`` (i.e. no point pokes
    above the chord). The visibility distance is the largest ``s[j]-s[i]``
    for which this holds, capped at ``max_lookahead_m``.

    Runs in O(N · k) where ``k`` is the average number of nodes within
    ``max_lookahead_m`` — fast enough for any PTH (N < 2000).
    """
    n = s.shape[0]
    out = np.zeros(n)
    if n < 2 or z.shape[0] != n:
        return out
    for i in range(n):
        eye = z[i] + eye_height_m
        s_i = s[i]
        last_ok_dist = 0.0
        for j in range(i + 1, n):
            ds = s[j] - s_i
            if ds > max_lookahead_m:
                break
            target = z[j] + look_height_m
            # Chord between (s_i, eye) and (s[j], target);
            # check every intermediate k.

            blocked = False
            if j > i + 1:
                ks = slice(i + 1, j)
                frac = (s[ks] - s_i) / max(ds, 1e-9)
                chord = eye + (target - eye) * frac
                # If any z[k] (the surface) rises above the chord, blocked.
                if np.any(z[ks] > chord):
                    blocked = True
            if blocked:
                break
            last_ok_dist = ds
        out[i] = last_ok_dist
    return out


# ---------------------------------------------------------------------------
# Convenience iterator (mainly for tests)
# ---------------------------------------------------------------------------

def iter_surface_classes(mesh: SmxMesh) -> Iterable[tuple[str, int]]:
    """Yield ``(class_name, vertex_count)`` for each surface class."""
    classes = classify_surface(mesh)
    if classes.size == 0:
        return
    for name, idx in SURFACE_INDEX.items():
        yield name, int((classes == idx).sum())


# ---------------------------------------------------------------------------
# One-shot enrichment of a TrackProfile
# ---------------------------------------------------------------------------

def enrich_profile_with_smx(
    profile: TrackProfile,
    mesh: SmxMesh,
    *,
    half_width_m: float = 12.0,
    banking_slice_thickness_m: float = 3.0,
    apex_eye_height_m: float = 1.0,
    apex_look_height_m: float = 0.5,
) -> TrackProfile:
    """Return a new :class:`TrackProfile` enriched with SMX-derived fields.

    The input profile is not mutated. The returned profile carries the
    original PTH-derived fields plus ``banking_rad``, ``surface_fractions``,
    ``surface_classes`` and ``apex_visibility_m`` populated from ``mesh``.
    """
    classes = classify_surface(mesh)
    banking = compute_banking_profile(
        profile.pos, mesh, classes,
        half_width_m=half_width_m,
        slice_thickness_m=banking_slice_thickness_m,
    )
    fractions = surface_distribution_along(
        profile.pos[:, :2], mesh, classes, half_width_m=half_width_m,
    )
    visibility = apex_visibility_distance(
        profile.s, profile.pos[:, 2],
        eye_height_m=apex_eye_height_m,
        look_height_m=apex_look_height_m,
    )
    # BVH-style barrier scan (left + right corridor margins).
    tangents = _centreline_tangents(profile.pos[:, :2])
    barrier_left, barrier_right = compute_barrier_offsets(
        profile.pos[:, :2], tangents, mesh, classes,
        max_search_m=max(half_width_m * 3.0, 30.0),
    )
    effective_width = barrier_left + barrier_right
    return TrackProfile(
        name=profile.name,
        s=profile.s, pos=profile.pos, direction=profile.direction,
        slope_pct=profile.slope_pct,
        curvature_1_per_m=profile.curvature_1_per_m,
        radius_m=profile.radius_m, heading_rad=profile.heading_rad,
        drive_left_m=profile.drive_left_m, drive_right_m=profile.drive_right_m,
        limit_left_m=profile.limit_left_m, limit_right_m=profile.limit_right_m,
        banking_rad=banking,
        surface_fractions=fractions,
        surface_classes=SURFACE_CLASSES,
        apex_visibility_m=visibility,
        barrier_left_m=barrier_left,
        barrier_right_m=barrier_right,
        effective_width_m=effective_width,
        los_apex_m=visibility,
    )
