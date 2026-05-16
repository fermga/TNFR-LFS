"""Tests for the racing_lines-backed track geometry enrichment."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lfs_telemetry.telemetry.derived import enrich_dataframe
from lfs_telemetry.telemetry.observables import car_spec_for
from lfs_telemetry.telemetry.track import loader as tg_loader


# ---------------------------------------------------------------------------
# Synthetic racing-line CSV fixture
# ---------------------------------------------------------------------------


def _write_synthetic_racing_csv(path):
    """Write a minimal racing_lines CSV: a 100 m straight along +X with 1% climb."""
    n = 21
    s = np.linspace(0.0, 100.0, n)
    df = pd.DataFrame({
        "s_m": s,
        "x_center_m": s,                # heading along +X
        "y_center_m": np.zeros(n),
        "z_center_m": s * 0.01,         # 1% climb
        "x_line_m": s,
        "y_line_m": np.zeros(n),
        "offset_m": np.zeros(n),
        "heading_rad": np.zeros(n),     # straight along +X => heading 0
        "curvature_1_per_m": np.zeros(n),
        "radius_m": np.full(n, 1e6),
        "slope_pct": np.full(n, 1.0),   # 1% uphill
        "width_m": np.full(n, 12.0),
        "segment_id": np.zeros(n, dtype=np.int64),
        "segment_kind": np.array(["straight"] * n, dtype=object),
        "v_target_ms": np.full(n, 80.0),
        "v_target_kmh": np.full(n, 288.0),
    })
    df.to_csv(path, index=False)


@pytest.fixture
def patched_racing_lines(tmp_path, monkeypatch):
    """Point the loader at a temp dir with a synthetic ``XX1_racing.csv``."""
    rl_dir = tmp_path / "racing_lines"
    rl_dir.mkdir()
    _write_synthetic_racing_csv(rl_dir / "XX1_racing.csv")

    def _candidates() -> list:
        return [rl_dir]

    monkeypatch.setattr(tg_loader, "candidate_racing_lines_dirs", _candidates)
    # Reset module-level memo so previous tests don't pollute.
    tg_loader._CACHE.clear()
    yield rl_dir
    tg_loader._CACHE.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _base_df(track: str = "XX1") -> pd.DataFrame:
    """Synthetic 5-row stint sitting on the racing line."""
    n = 5
    return pd.DataFrame({
        "time_ms": np.arange(n) * 100,
        "ctx_track": [track] * n,
        "car": ["FBM"] * n,
        "pos_x": np.linspace(10.0, 50.0, n),
        "pos_y": np.zeros(n),
        "pos_z": np.linspace(0.10, 0.50, n),
        "vel_x": np.full(n, 30.0),       # 30 m/s along +X
        "vel_y": np.zeros(n),
        "speed_ms": np.full(n, 30.0),
        "accel_x": np.full(n, 2.0),      # 2 m/s² measured
        "accel_y": np.zeros(n),
        "ang_vel_z": np.zeros(n),
    })


def test_track_columns_added_when_csv_exists(patched_racing_lines):
    df = enrich_dataframe(_base_df(), car_spec_for("FBM"))
    for col in ("track_s_m", "track_z_m", "track_heading_rad",
                "track_slope_pct", "segment_id", "segment_kind",
                "track_offset_m"):
        assert col in df.columns, f"missing {col}"
    # Centerline runs along +X with width 12 → offset must be ~0.
    assert np.allclose(df["track_offset_m"], 0.0, atol=1e-6)
    # Slope is constant 1%.
    assert np.allclose(df["track_slope_pct"], 1.0)
    # Segment classification.
    assert (df["segment_kind"] == "straight").all()


def test_slope_corrected_accel_removes_gravity(patched_racing_lines):
    df = enrich_dataframe(_base_df(), car_spec_for("FBM"))
    # Uphill 1% slope with car pointed uphill: gravity drags backward by
    # g·sin(arctan(0.01)) ≈ 9.80665 * 0.0099995 ≈ 0.0980 m/s². The road-frame
    # acceleration is therefore measured (2.0) + that drag = ~2.098 m/s².
    expected = 2.0 + 9.80665 * np.sin(np.arctan(0.01))
    assert "accel_x_road_mps2" in df.columns
    assert np.allclose(df["accel_x_road_mps2"], expected, atol=1e-6)


def test_yaw_misalignment_zero_when_aligned(patched_racing_lines):
    df = enrich_dataframe(_base_df(), car_spec_for("FBM"))
    # Velocity is +X, track heading is 0 rad → misalignment ≈ 0.
    assert "yaw_misalign_rad" in df.columns
    assert np.allclose(df["yaw_misalign_rad"], 0.0, atol=1e-9)


def test_yaw_misalignment_detects_sideways_drift(patched_racing_lines):
    raw = _base_df()
    raw["vel_y"] = 30.0  # 45° drift
    raw["vel_x"] = 30.0
    df = enrich_dataframe(raw, car_spec_for("FBM"))
    expected = np.pi / 4
    assert np.allclose(df["yaw_misalign_rad"], expected, atol=1e-6)


def test_enrich_skips_track_columns_when_no_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(tg_loader, "candidate_racing_lines_dirs",
                        lambda: [tmp_path / "missing"])
    tg_loader._CACHE.clear()
    df = enrich_dataframe(_base_df("UNKNOWN_TRACK"), car_spec_for("FBM"))
    # No racing line → no track columns; existing physics columns still there.
    assert "track_s_m" not in df.columns
    assert "accel_x_road_mps2" not in df.columns
    assert "yaw_misalign_rad" not in df.columns


def test_grid_sentinel_positions_become_nan(patched_racing_lines):
    raw = _base_df()
    # First two rows are sentinel zeros (e.g. pre-grid).
    raw.loc[0, ["pos_x", "pos_y"]] = (0.0, 0.0)
    raw.loc[1, ["pos_x", "pos_y"]] = (0.0, 0.0)
    df = enrich_dataframe(raw, car_spec_for("FBM"))
    assert np.isnan(df["track_s_m"].iloc[0])
    assert np.isnan(df["track_s_m"].iloc[1])
    assert df["segment_id"].iloc[0] == -1
    assert df["segment_kind"].iloc[0] == ""
    # Real samples preserved.
    assert np.isfinite(df["track_s_m"].iloc[-1])


def test_real_install_fixture_loads_bl1():
    """Smoke test against the bundled racing_lines/BL1_racing.csv."""
    geom = tg_loader.cached_track_geometry("BL1")
    if geom is None:
        pytest.skip("racing_lines/BL1_racing.csv not bundled in this checkout")
    assert geom.num_nodes > 100
    # Lookup at the centroid → must return a valid finite s.
    cx, cy = geom.xy.mean(axis=0)
    out = geom.lookup(np.array([cx]), np.array([cy]))
    assert np.isfinite(out["track_s_m"][0])
    assert out["segment_kind"][0] in {"straight", "left", "right"}
