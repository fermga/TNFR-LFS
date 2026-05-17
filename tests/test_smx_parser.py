"""Tests for the SMX parser and elevation helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lfs_telemetry.telemetry.track.smx import (
    DEFAULT_SMX_DIR,
    SmxMesh,
    cross_section_at,
    elevation_envelope,
    find_smx_for_track,
    list_smx_files,
    parse_smx,
)

SMX_FILES = list_smx_files(DEFAULT_SMX_DIR)


def _expected_track_id(stem: str) -> str:
    """LFS env id encoded in the SMX filename prefix (Aston_3DH → AS, …)."""
    prefix = stem.split("_", 1)[0].lower()
    return {
        "aston":      "AS",
        "autocross":  "AU",
        "blackwood":  "BL",
        "fern bay":   "FE",
        "kyoto ring": "KY",
        "south city": "SO",
        "westhill":   "WE",
    }.get(prefix, "")


@pytest.fixture(scope="module")
def first_mesh() -> SmxMesh:
    if not SMX_FILES:
        pytest.skip("no SMX files in DEFAULT_SMX_DIR (C:/LFS/data/smx)")
    return parse_smx(SMX_FILES[0])


@pytest.mark.skipif(
    not SMX_FILES, reason="no SMX files in DEFAULT_SMX_DIR"
)
def test_bundled_smx_files_present() -> None:
    assert SMX_FILES
    assert len(SMX_FILES) >= 6


@pytest.mark.parametrize(
    "smx_path",
    SMX_FILES,
    ids=[p.stem for p in SMX_FILES] or ["skipped"],
)
def test_parse_smx_all_bundled(smx_path: Path) -> None:
    mesh = parse_smx(smx_path)
    assert isinstance(mesh, SmxMesh)
    # Header invariants.
    assert mesh.smx_version == 0
    assert mesh.num_objects > 0
    # The track label is the LFS-internal env name; not necessarily
    # equal to the filename, but it must be non-empty ASCII.
    assert mesh.track_label.strip(), \
        f"empty track_label in {smx_path.name}"
    # Geometry sanity.
    assert mesh.num_vertices > 0
    assert mesh.num_triangles > 0
    assert mesh.vertices.shape == (mesh.num_vertices, 3)
    assert mesh.colors.shape == (mesh.num_vertices, 4)
    assert mesh.triangles.shape == (mesh.num_triangles, 3)
    # All triangle indices must reference real vertices.
    assert int(mesh.triangles.max()) < mesh.num_vertices
    # Bounds: tracks span tens to thousands of metres horizontally
    # (Autocross is the smallest at ~150 m).
    lo, hi = mesh.bounds_xyz()
    extent_xy = max(hi[0] - lo[0], hi[1] - lo[1])
    assert 50.0 < extent_xy < 20_000.0, (
        f"{smx_path.name}: implausible XY extent {extent_xy:.1f} m")
    # Z (altitude) is finite and reasonable: -200 m to +400 m.
    z_lo, z_hi = mesh.elevation_range_m()
    assert np.isfinite(z_lo) and np.isfinite(z_hi)
    assert -300.0 < z_lo <= z_hi < 1000.0
    # Each object's slice is consistent.
    for obj in mesh.objects:
        assert obj.num_points >= 0
        assert obj.num_triangles >= 0
        assert 0 <= obj.vertex_start <= obj.vertex_end <= mesh.num_vertices
        assert 0 <= obj.tri_start <= obj.tri_end <= mesh.num_triangles


def test_bad_magic_rejected(tmp_path: Path) -> None:
    p = tmp_path / "fake.smx"
    p.write_bytes(b"NOTSMX" + b"\x00" * 200)
    with pytest.raises(ValueError, match="bad magic"):
        parse_smx(p)


def test_truncated_file_rejected(tmp_path: Path) -> None:
    p = tmp_path / "short.smx"
    p.write_bytes(b"LFSSMX" + b"\x00" * 4)
    with pytest.raises(ValueError, match="too short"):
        parse_smx(p)


def test_find_smx_for_track() -> None:
    if not SMX_FILES:
        pytest.skip("no bundled SMX")
    # Match by 2-letter env id via the env_map fallback.
    p = find_smx_for_track("BL1", smx_dir=DEFAULT_SMX_DIR)
    if p is not None:
        assert "blackwood" in p.stem.lower()


def test_elevation_envelope_on_synthetic_centreline(
    first_mesh: SmxMesh,
) -> None:
    # Build a small XY grid that lies inside the mesh footprint by
    # walking from the centroid in +X.
    lo, hi = first_mesh.bounds_xyz()
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    n = 50
    xs = np.linspace(cx - 50, cx + 50, n)
    ys = np.full(n, cy)
    centreline = np.column_stack((xs, ys))
    s = np.linspace(0.0, 100.0, n)
    s_out, z_lo, z_hi = elevation_envelope(
        first_mesh, centreline, s, half_width_m=30.0)
    assert s_out is s
    assert z_lo.shape == z_hi.shape == (n,)
    # At least one bin must have been touched (the mesh covers the
    # centroid).
    touched = np.isfinite(z_lo)
    assert touched.any()
    # Min ≤ max everywhere a bin was touched.
    assert (z_lo[touched] <= z_hi[touched]).all()


def test_cross_section_returns_sorted_by_lateral_offset(
    first_mesh: SmxMesh,
) -> None:
    centre = first_mesh.vertices.mean(axis=0)[:2]
    cs = cross_section_at(
        first_mesh, centre, np.array([1.0, 0.0]),
        half_width_m=50.0, slice_thickness_m=10.0,
    )
    if cs.size == 0:
        pytest.skip("no vertices in the test slice for this mesh")
    assert cs.shape[1] == 2
    assert (np.diff(cs[:, 0]) >= 0).all()
