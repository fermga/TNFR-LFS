"""Tests for the SMX-derived 3D geometry features (geom3d module)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lfs_telemetry.telemetry.track import geom3d
from lfs_telemetry.telemetry.track.pth import TrackProfile
from lfs_telemetry.telemetry.track.smx import (
    DEFAULT_SMX_DIR,
    SmxMesh,
    SmxObject,
    parse_smx,
)

# ---------------------------------------------------------------------------
# Helpers: synthetic meshes
# ---------------------------------------------------------------------------

def _make_synthetic_mesh(
    vertices: np.ndarray,
    colors: np.ndarray,
    *,
    cp_indices: np.ndarray | None = None,
    objects: list[SmxObject] | None = None,
) -> SmxMesh:
    """Build a minimal SmxMesh for unit tests."""
    if objects is None:
        objects = [SmxObject(
            index=0, centre=vertices.mean(axis=0), radius_m=1.0,
            vertex_start=0, vertex_end=vertices.shape[0],
            tri_start=0, tri_end=0,
        )]
    return SmxMesh(
        name="synthetic", track_label="synthetic",
        smx_version=0, game_version=0, game_revision=0,
        resolution=0, ground_rgb=(0, 0, 0),
        vertices=vertices, colors=colors,
        triangles=np.empty((0, 3), dtype=np.uint32),
        objects=objects,
        cp_object_indices=(np.empty(0, dtype=np.int64)
                           if cp_indices is None else cp_indices),
    )


def _plane_strip(
    *, n_s: int = 40, n_t: int = 11,
    s_max: float = 100.0, half_width: float = 6.0,
    z_func=None, color=(255, 80, 80, 80),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a flat XY rectangular strip aligned with +X.

    Returns (vertices, colors, centreline_xyz).
    """
    s = np.linspace(0, s_max, n_s)
    t = np.linspace(-half_width, half_width, n_t)
    ss, tt = np.meshgrid(s, t, indexing="ij")
    zz = (np.zeros_like(ss) if z_func is None
          else np.vectorize(z_func)(ss, tt))
    verts = np.column_stack([ss.ravel(), tt.ravel(), zz.ravel()])
    cols = np.tile(np.array(color, dtype=np.uint8), (verts.shape[0], 1))
    centre = np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])
    return verts, cols, centre


# ---------------------------------------------------------------------------
# classify_surface
# ---------------------------------------------------------------------------

def test_classify_surface_palette():
    # ARGB
    colors = np.array([
        [255, 80, 80, 80],     # mid grey -> asphalt
        [255, 30, 30, 30],     # dark grey -> asphalt
        [255, 220, 40, 40],    # bright red -> kerb
        [255, 230, 230, 230],  # near-white -> kerb
        [255, 60, 160, 60],    # green -> grass
        [255, 200, 170, 90],   # sandy -> runoff
        [255, 0, 0, 200],      # blue -> other
    ], dtype=np.uint8)
    mesh = _make_synthetic_mesh(
        np.zeros((colors.shape[0], 3)), colors,
    )
    out = geom3d.classify_surface(mesh)
    si = geom3d.SURFACE_INDEX
    assert out[0] == si["asphalt"]
    assert out[1] == si["asphalt"]
    assert out[2] == si["kerb"]
    assert out[3] == si["kerb"]
    assert out[4] == si["grass"]
    assert out[5] == si["runoff"]
    assert out[6] == si["other"]


# ---------------------------------------------------------------------------
# Banking — synthetic
# ---------------------------------------------------------------------------

def test_banking_flat_synthetic():
    verts, cols, centre = _plane_strip()
    mesh = _make_synthetic_mesh(verts, cols)
    out = geom3d.compute_banking_profile(
        centre, mesh, half_width_m=8.0, slice_thickness_m=4.0,
    )
    # Interior stations should give ~0 banking.
    interior = out[5:-5]
    finite = interior[np.isfinite(interior)]
    assert finite.size > 0
    assert np.max(np.abs(finite)) < np.deg2rad(0.5)


def test_banking_tilted_synthetic():
    angle = np.deg2rad(5.0)
    # In LFS coords, with tangent +X the right-hand normal is (0, -1),
    # so a vertex at world-Y = +Y lies at lateral t = -Y (LFS-LEFT).
    # To express "peraltado hacia LFS-right (+t) by `angle`" we must
    # raise Z toward -Y in world frame.
    verts, cols, centre = _plane_strip(
        z_func=lambda s, t: -t * np.tan(angle),
    )
    mesh = _make_synthetic_mesh(verts, cols)
    out = geom3d.compute_banking_profile(
        centre, mesh, half_width_m=8.0, slice_thickness_m=4.0,
    )
    interior = out[5:-5]
    finite = interior[np.isfinite(interior)]
    assert finite.size > 0
    assert np.allclose(finite, angle, atol=np.deg2rad(0.5))


# ---------------------------------------------------------------------------
# surface_distribution_along
# ---------------------------------------------------------------------------

def test_surface_distribution_rows_sum_to_one():
    verts, cols, centre = _plane_strip()
    mesh = _make_synthetic_mesh(verts, cols)
    frac = geom3d.surface_distribution_along(
        centre[:, :2], mesh, half_width_m=10.0,
    )
    totals = frac.sum(axis=1)
    touched = totals > 0
    assert touched.any()
    assert np.allclose(totals[touched], 1.0, atol=1e-9)


def test_kerb_mask_side_selection():
    # Asphalt strip + a kerb row only on the LFS-right flank.
    # With tangent +X, LFS-right = world Y < 0.
    verts_a, cols_a, centre = _plane_strip()
    n_s = 40
    s = np.linspace(0, 100, n_s)
    kerb_verts = np.column_stack([s, np.full_like(s, -5.0), np.zeros_like(s)])
    kerb_cols = np.tile(
        np.array([255, 220, 40, 40], dtype=np.uint8), (n_s, 1))
    verts = np.vstack([verts_a, kerb_verts])
    cols = np.vstack([cols_a, kerb_cols])
    mesh = _make_synthetic_mesh(verts, cols)
    right = geom3d.kerb_mask_along(
        centre[:, :2], mesh, side=+1, half_width_m=10.0,
        min_kerb_vertices=1,
    )
    left = geom3d.kerb_mask_along(
        centre[:, :2], mesh, side=-1, half_width_m=10.0,
        min_kerb_vertices=1,
    )
    assert right.any()
    assert not left.any()


# ---------------------------------------------------------------------------
# Checkpoint geometry
# ---------------------------------------------------------------------------

def test_extract_checkpoint_geometry_synthetic_rect():
    # Build a rectangular CP object spanning t∈[-4,4] at s=10.
    pts = np.array([
        [10.0, -4.0, 0.0], [10.0, 4.0, 0.0],
        [10.2, -4.0, 0.0], [10.2, 4.0, 0.0],
    ])
    cols = np.tile(np.array([255, 0, 0, 200], dtype=np.uint8), (4, 1))
    obj = SmxObject(
        index=0, centre=pts.mean(axis=0), radius_m=4.0,
        vertex_start=0, vertex_end=4, tri_start=0, tri_end=0,
    )
    mesh = _make_synthetic_mesh(
        pts, cols, cp_indices=np.array([0], dtype=np.int64), objects=[obj],
    )
    cps = geom3d.extract_checkpoint_geometry(mesh)
    assert len(cps) == 1
    cp = cps[0]
    assert cp.polygon_xy.shape[0] >= 3
    # principal axis should be ~ ±Y (the long side)
    assert abs(cp.normal_xy[1]) > 0.9
    assert cp.half_width_m == pytest.approx(4.0, abs=0.1)


# ---------------------------------------------------------------------------
# Corridor heightmap
# ---------------------------------------------------------------------------

def test_corridor_heightmap_shape_and_roundtrip(tmp_path: Path):
    verts, cols, centre = _plane_strip()
    s = np.arange(centre.shape[0]) * (100.0 / (centre.shape[0] - 1))
    mesh = _make_synthetic_mesh(verts, cols)
    hm = geom3d.corridor_heightmap(
        centre, s, mesh, n_t=11, half_width_m=5.0,
        slice_thickness_m=4.0,
    )
    assert hm.z.shape == (centre.shape[0], 11)
    finite = np.isfinite(hm.z)
    assert finite.sum() > hm.z.size * 0.5
    # Z is ~0 everywhere on a flat plane.
    assert np.nanmax(np.abs(hm.z)) < 1e-6
    out = hm.save_npz(tmp_path / "hm.npz")
    hm2 = geom3d.CorridorHeightmap.load_npz(out)
    assert np.array_equal(hm2.s, hm.s)
    assert np.array_equal(hm2.t, hm.t)
    assert np.allclose(
        np.nan_to_num(hm2.z), np.nan_to_num(hm.z),
    )
    assert hm2.half_width_m == hm.half_width_m


# ---------------------------------------------------------------------------
# Apex visibility (synthetic crest)
# ---------------------------------------------------------------------------

def test_apex_visibility_distance_basic():
    s = np.linspace(0, 400, 201)
    # A flat road: visibility should equal the lookahead cap on interior
    # nodes well within the array.
    vis_flat = geom3d.apex_visibility_distance(
        s, np.zeros_like(s),
        max_lookahead_m=150.0,
    )
    assert vis_flat[50] == pytest.approx(150.0, abs=4.0)
    # A sharp crest at s=100 hiding a descending back side: the
    # approach should be visibility-limited (cannot see past the crest)
    # while the very top of the crest itself sees the whole back side.
    z_hill = np.where(s <= 100.0, 0.05 * s, 5.0 - 0.05 * (s - 100.0))
    vis_h = geom3d.apex_visibility_distance(
        s, z_hill,
        eye_height_m=1.0, look_height_m=0.5,
        max_lookahead_m=300.0,
    )
    # Before the crest (approach), driver cannot see the descending side.
    assert vis_h[30] < 150.0
    # At/after the crest the road opens up.
    assert vis_h[60] > vis_h[30]
    assert vis_h.min() >= 0.0
    assert vis_h.max() <= 300.0


# ---------------------------------------------------------------------------
# enrich_profile_with_smx (end-to-end synthetic)
# ---------------------------------------------------------------------------

def test_enrich_profile_with_smx_synthetic():
    verts, cols, centre = _plane_strip(
        z_func=lambda s, t: t * np.tan(np.deg2rad(3.0)),
    )
    mesh = _make_synthetic_mesh(verts, cols)
    n = centre.shape[0]
    s = np.linspace(0, 100, n)
    profile = TrackProfile(
        name="syn", s=s, pos=centre,
        direction=np.tile([1.0, 0.0, 0.0], (n, 1)),
        slope_pct=np.zeros(n), curvature_1_per_m=np.zeros(n),
        radius_m=np.full(n, 1e6), heading_rad=np.zeros(n),
        drive_left_m=np.full(n, -6.0), drive_right_m=np.full(n, 6.0),
        limit_left_m=np.full(n, -10.0), limit_right_m=np.full(n, 10.0),
    )
    enriched = geom3d.enrich_profile_with_smx(
        profile, mesh, half_width_m=8.0,
    )
    assert enriched.banking_rad is not None
    assert enriched.banking_rad.shape == (n,)
    assert enriched.surface_fractions is not None
    assert enriched.surface_fractions.shape == (n, len(geom3d.SURFACE_CLASSES))
    assert enriched.surface_classes == geom3d.SURFACE_CLASSES
    assert enriched.apex_visibility_m is not None
    assert enriched.apex_visibility_m.shape == (n,)
    # Original profile is unchanged.
    assert profile.banking_rad is None


# ---------------------------------------------------------------------------
# Parametrised over the 7 shipped SMX files
# ---------------------------------------------------------------------------

_SMX_FILES = sorted(DEFAULT_SMX_DIR.glob("*.smx")) if DEFAULT_SMX_DIR.exists() else []


@pytest.mark.skipif(not _SMX_FILES, reason="no bundled SMX files")
@pytest.mark.parametrize("smx_path", _SMX_FILES, ids=lambda p: p.stem)
def test_classify_and_checkpoints_on_real_smx(smx_path: Path):
    mesh = parse_smx(smx_path)
    classes = geom3d.classify_surface(mesh)
    assert classes.shape == (mesh.num_vertices,)
    # Every real track should contain at least some asphalt vertices.
    asphalt_count = int((classes == geom3d.SURFACE_INDEX["asphalt"]).sum())
    assert asphalt_count > 0, f"no asphalt detected in {smx_path.name}"
    # Each track should have at least one checkpoint object.
    cps = geom3d.extract_checkpoint_geometry(mesh)
    if mesh.cp_object_indices.size > 0:
        assert len(cps) >= 1
        bmin, bmax = mesh.bounds_xyz()
        for cp in cps:
            assert np.all(cp.centre >= bmin - 1.0)
            assert np.all(cp.centre <= bmax + 1.0)


@pytest.mark.skipif(not _SMX_FILES, reason="no bundled SMX files")
@pytest.mark.parametrize("smx_path", _SMX_FILES, ids=lambda p: p.stem)
def test_banking_on_real_smx_reasonable(smx_path: Path):
    mesh = parse_smx(smx_path)
    # Build a coarse centreline by sampling 200 points along the mesh
    # bounding box X axis at mid-Y.
    bmin, bmax = mesh.bounds_xyz()
    centre = np.column_stack([
        np.linspace(bmin[0], bmax[0], 80),
        np.full(80, 0.5 * (bmin[1] + bmax[1])),
        np.full(80, 0.5 * (bmin[2] + bmax[2])),
    ])
    banking = geom3d.compute_banking_profile(
        centre, mesh, half_width_m=15.0, slice_thickness_m=4.0,
    )
    # Note: this centreline is a straight diagonal across the bounding
    # box, not the actual racing line, so the slices may catch off-track
    # ramps. We only require that the function runs end-to-end and that
    # at least some stations produce a finite (non-NaN) banking value.
    finite = banking[np.isfinite(banking)]
    assert finite.size > 0, f"no banking samples on {smx_path.name}"
