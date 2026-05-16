"""Tests for the MoTeC-style helpers: channels, catalog, comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lfs_telemetry.telemetry import (
    CHANNELS,
    CaptureInfo,
    ChannelInfo,
    LapComparison,
    LapTelemetry,
    channel_info,
    channels_by_group,
    discover_captures,
    inspect_capture,
    write_csv_replay,
)
from lfs_telemetry.telemetry.live import TelemetrySample
from lfs_telemetry.telemetry.protocol.packets import WHEEL_ORDER


# ---------------------------------------------------------------------------
# Synthetic lap helpers (mirrors test_stint.py)
# ---------------------------------------------------------------------------


def _synthetic_lap_df(
    *,
    n: int = 600,
    speed_ms: float = 30.0,
    time_scale: float = 1.0,
) -> pd.DataFrame:
    """Build a minimal but enrich-compatible lap DataFrame at 100 Hz.

    ``time_scale`` stretches the time axis (≥1 → slower lap, same
    distance), which is what we need for the comparison tests.
    """
    t_ms = (np.arange(n, dtype=np.int64) * 10 * time_scale).astype(np.int64)
    cols: dict[str, np.ndarray] = {
        "time_ms": t_ms,
        "ang_vel_x": np.zeros(n), "ang_vel_y": np.zeros(n), "ang_vel_z": np.full(n, 0.05),
        "heading": np.zeros(n), "pitch": np.zeros(n), "roll": np.zeros(n),
        "accel_x": np.full(n, 0.5), "accel_y": np.full(n, 0.3), "accel_z": np.full(n, 9.81),
        "vel_x": np.full(n, speed_ms), "vel_y": np.zeros(n), "vel_z": np.zeros(n),
        "pos_x": np.arange(n, dtype=float), "pos_y": np.zeros(n), "pos_z": np.zeros(n),
        "car": np.array(["FBM"] * n, dtype=object),
        "gear": np.full(n, 4), "speed_ms": np.full(n, speed_ms), "rpm": np.full(n, 8000.0),
        "throttle": np.full(n, 0.7), "brake": np.zeros(n), "clutch": np.zeros(n),
        "fuel": np.full(n, 0.5), "eng_temp_c": np.full(n, 85.0), "oil_temp_c": np.full(n, 92.0),
        "oil_pressure_bar": np.full(n, 4.5), "turbo_bar": np.zeros(n),
        "og_flags": np.zeros(n, dtype=int), "dash_lights": np.zeros(n, dtype=int),
        "show_lights": np.zeros(n, dtype=int), "og_player_id": np.ones(n, dtype=int),
        "current_lap_dist_m": np.arange(n, dtype=float),
        "indexed_distance_m": np.arange(n, dtype=float),
        "steer_torque_nm": np.full(n, 5.0),
        "engine_ang_vel_rads": np.full(n, 800.0),
        "max_torque_at_vel_nm": np.full(n, 200.0),
        "input_throttle": np.full(n, 0.7), "input_brake": np.zeros(n),
        "input_steer": np.full(n, 0.05), "input_clutch": np.zeros(n),
        "input_handbrake": np.zeros(n),
        "ctx_track": np.array(["BL1"] * n, dtype=object),
    }
    for c in WHEEL_ORDER:
        cols[f"wheel_{c}_susp_deflect_m"] = np.full(n, 0.02)
        cols[f"wheel_{c}_vertical_load_n"] = np.full(n, 1500.0)
        cols[f"wheel_{c}_slip_ratio"] = np.zeros(n)
        cols[f"wheel_{c}_tan_slip_angle"] = np.zeros(n)
        cols[f"wheel_{c}_x_force_n"] = np.zeros(n)
        cols[f"wheel_{c}_y_force_n"] = np.zeros(n)
        cols[f"wheel_{c}_ang_vel_rads"] = np.full(n, 20.0)
        cols[f"wheel_{c}_lean_rel_road_rad"] = np.zeros(n)
        cols[f"wheel_{c}_air_temp_c"] = np.full(n, 25)
        cols[f"wheel_{c}_slip_fraction"] = np.zeros(n)
        cols[f"wheel_{c}_touching"] = np.ones(n, dtype=int)
        cols[f"wheel_{c}_steer_rad"] = np.zeros(n)
    return pd.DataFrame(cols)


def _make_lap(**kw) -> LapTelemetry:
    return LapTelemetry.from_dataframe(_synthetic_lap_df(**kw), car="FBM")


# ---------------------------------------------------------------------------
# channels.py
# ---------------------------------------------------------------------------


def test_channel_info_known_column() -> None:
    info = channel_info("speed_ms")
    assert isinstance(info, ChannelInfo)
    assert info.units == "m/s"
    assert info.group == "Vehicle"


def test_channel_info_unknown_falls_back_gracefully() -> None:
    info = channel_info("totally_made_up_column")
    assert info.label == "totally_made_up_column"
    assert info.units == ""
    assert info.group == "Other"


def test_channel_info_per_wheel_template_expanded() -> None:
    info = channel_info("wheel_FL_vertical_load_n")
    assert info.label.endswith("FL")
    assert info.units == "N"
    info = channel_info("friction_use_RR")
    assert info.label.endswith("RR")
    assert info.group == "Derived"


def test_channels_by_group_filters_to_columns() -> None:
    cols = ["speed_ms", "throttle", "brake", "wheel_FL_vertical_load_n"]
    groups = channels_by_group(cols)
    assert "Vehicle" in groups
    assert "Driver" in groups
    assert "Suspension" in groups
    flat = {info.column for items in groups.values() for info in items}
    assert flat == set(cols)


def test_channels_registry_is_populated() -> None:
    # Sanity: well over the base table count once wheel templates expand.
    assert len(CHANNELS) > 60


# ---------------------------------------------------------------------------
# LapTelemetry: channels + distance grid
# ---------------------------------------------------------------------------


def test_lap_channels_lists_all_enriched_columns() -> None:
    lap = _make_lap()
    # Should at least include the enriched derived signals.
    cols = {info.column for info in lap.channels}
    assert "speed_ms" in cols
    assert "friction_use_FL" in cols
    assert "ffb_load_pct" in cols


def test_lap_channel_vs_distance_returns_aligned_series() -> None:
    lap = _make_lap()
    s = lap.channel_vs_distance("speed_ms", n_points=200)
    assert s.index.name == "distance_m"
    assert s.size == 200
    # Constant speed in the synthetic lap → constant resampled values.
    assert np.allclose(s.to_numpy(), 30.0)


def test_lap_distance_grid_monotonic_zero_anchored() -> None:
    lap = _make_lap()
    grid = lap.distance_grid_m(n_points=100)
    assert grid.size == 100
    assert grid[0] == 0.0
    assert np.all(np.diff(grid) > 0)


# ---------------------------------------------------------------------------
# LapComparison
# ---------------------------------------------------------------------------


def test_lap_comparison_delta_time_for_slower_candidate() -> None:
    fast = _make_lap(time_scale=1.0)
    slow = _make_lap(time_scale=1.10)  # 10 % slower at every distance
    cmp = LapComparison.from_laps(fast, slow, n_points=500)
    d = cmp.delta_time_s
    assert d.size == 500
    # Slower candidate → strictly positive (and growing) delta.
    assert d[-1] > d[0]
    assert d[-1] > 0
    # End delta ≈ 10 % of fast lap_time.
    fast_t = fast.summary["lap_time_s"]
    assert cmp.total_delta_s == pytest.approx(0.10 * fast_t, rel=0.02)


def test_lap_comparison_channel_overlay_returns_two_columns() -> None:
    a = _make_lap(speed_ms=30.0)
    b = _make_lap(speed_ms=35.0)
    cmp = LapComparison.from_laps(a, b, n_points=100)
    df = cmp.channel("speed_ms")
    assert list(df.columns) == ["reference", "candidate"]
    assert df.index.name == "distance_m"
    assert np.allclose(df["reference"].to_numpy(), 30.0)
    assert np.allclose(df["candidate"].to_numpy(), 35.0)


def test_lap_comparison_summary_has_swing_metrics() -> None:
    cmp = LapComparison.from_laps(_make_lap(), _make_lap(time_scale=1.05), n_points=100)
    s = cmp.summary
    assert s["n_points"] == 100
    assert s["max_loss_s"] >= s["max_gain_s"] * -1 - 1e-9  # both finite
    assert "lap_time_delta_s" in s


# ---------------------------------------------------------------------------
# Catalog (workspace browser)
# ---------------------------------------------------------------------------


def _write_real_csv(path: Path, df: pd.DataFrame) -> None:
    """Persist via the real replay schema to keep the catalog test honest."""
    samples: list[TelemetrySample] = []  # build minimal samples from df
    # Easier: just write a CSV that mirrors the schema preamble + header so
    # inspect_capture sees a valid file.
    from lfs_telemetry.telemetry.replay import _SCHEMA_HEADER  # type: ignore[attr-defined]
    with path.open("w", encoding="utf-8", newline="") as fp:
        fp.write(f"{_SCHEMA_HEADER}\n")
        df.to_csv(fp, index=False)


def test_inspect_capture_reads_metadata(tmp_path: Path) -> None:
    df = _synthetic_lap_df()
    p = tmp_path / "stint_lap01.csv"
    _write_real_csv(p, df)
    info = inspect_capture(p)
    assert info is not None
    assert info.car == "FBM"
    assert info.track == "BL1"
    assert info.samples == len(df)
    assert info.lap_time_s == pytest.approx((len(df) - 1) * 0.01, abs=1e-3)
    assert info.distance_m == pytest.approx(len(df) - 1, abs=1.0)
    assert info.schema_version == "1.1"


def test_discover_captures_finds_and_skips_foreign(tmp_path: Path) -> None:
    _write_real_csv(tmp_path / "lap01.csv", _synthetic_lap_df())
    _write_real_csv(tmp_path / "lap02.csv", _synthetic_lap_df())
    # Foreign CSV without ``time_ms``: must be skipped, not raise.
    pd.DataFrame({"foo": [1, 2, 3]}).to_csv(tmp_path / "junk.csv", index=False)
    items = discover_captures(tmp_path)
    assert len(items) == 2
    assert all(isinstance(i, CaptureInfo) for i in items)


def test_inspect_capture_returns_none_on_missing_file(tmp_path: Path) -> None:
    assert inspect_capture(tmp_path / "nope.csv") is None
