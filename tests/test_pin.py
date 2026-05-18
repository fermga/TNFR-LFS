"""Tests for the PIN parser."""
from __future__ import annotations

import struct

import pytest

from lfs_telemetry.telemetry.track.pin import (
    PIN_MAGIC,
    find_env_for_xy,
    list_pin_files,
    load_all,
    parse_pin,
    parse_pin_bytes,
)
from lfs_telemetry.telemetry.track.pth import DEFAULT_SMX_DIR, FIXED_POINT_DIVISOR

# ---------------------------------------------------------------------------
# Synthetic-data tests (no LFS install required)
# ---------------------------------------------------------------------------


def _build_pin(layout_count: int, xmin: float, xmax: float,
               ymin: float, ymax: float) -> bytes:
    return (
        PIN_MAGIC
        + b"\x00" * 6
        + struct.pack("<I", layout_count)
        + struct.pack(
            "<4i",
            int(xmin * FIXED_POINT_DIVISOR),
            int(xmax * FIXED_POINT_DIVISOR),
            int(ymin * FIXED_POINT_DIVISOR),
            int(ymax * FIXED_POINT_DIVISOR),
        )
    )


def test_parse_pin_bytes_synthetic():
    data = _build_pin(4, -1152.0, 512.0, -832.0, 832.0)
    info = parse_pin_bytes(data, env="BL")
    assert info.env == "BL"
    assert info.layout_count == 4
    assert info.x_min_m == pytest.approx(-1152.0)
    assert info.x_max_m == pytest.approx(512.0)
    assert info.y_min_m == pytest.approx(-832.0)
    assert info.y_max_m == pytest.approx(832.0)
    assert info.width_m == pytest.approx(1664.0)
    assert info.height_m == pytest.approx(1664.0)
    assert info.bbox == (-1152.0, -832.0, 512.0, 832.0)


def test_pin_contains_xy_with_margin():
    info = parse_pin_bytes(_build_pin(4, -100.0, 100.0, -50.0, 50.0), env="X")
    assert info.contains_xy(0.0, 0.0)
    assert info.contains_xy(100.0, 50.0)
    assert not info.contains_xy(101.0, 0.0)
    assert info.contains_xy(101.0, 0.0, margin_m=5.0)
    assert not info.contains_xy(0.0, 60.0)
    assert info.contains_xy(0.0, 60.0, margin_m=15.0)


def test_parse_pin_rejects_bad_magic():
    bad = b"NOPE!!" + b"\x00" * 26
    with pytest.raises(ValueError, match="bad magic"):
        parse_pin_bytes(bad)


def test_parse_pin_rejects_bad_length():
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        parse_pin_bytes(b"LFSPIN" + b"\x00")


def test_parse_pin_rejects_nonzero_reserved():
    bad = PIN_MAGIC + b"\x00\x01\x00\x00\x00\x00" + b"\x00" * 20
    with pytest.raises(ValueError, match="reserved bytes"):
        parse_pin_bytes(bad)


def test_find_env_for_xy_filters_correctly():
    pins = [
        parse_pin_bytes(_build_pin(1, -100.0, 100.0, -100.0, 100.0), env="A"),
        parse_pin_bytes(_build_pin(1, 500.0, 600.0, 500.0, 600.0), env="B"),
    ]
    matches = find_env_for_xy(0.0, 0.0, pins=pins)
    assert {p.env for p in matches} == {"A"}
    matches = find_env_for_xy(550.0, 550.0, pins=pins)
    assert {p.env for p in matches} == {"B"}
    matches = find_env_for_xy(1000.0, 1000.0, pins=pins)
    assert matches == []


# ---------------------------------------------------------------------------
# Real-install tests (skip when C:/LFS missing)
# ---------------------------------------------------------------------------


_install = pytest.mark.skipif(
    not DEFAULT_SMX_DIR.exists(),
    reason="LFS install not available — PIN install tests skipped.",
)


@_install
def test_load_all_pin_files_from_install():
    files = list_pin_files()
    pins = load_all()
    assert len(files) >= 7  # ship at least 7 envs
    # Ship envs we expect to see (RO/LA may be missing on some installs).
    expected_subset = {"AS", "AU", "BL", "FE", "KY", "SO", "WE"}
    assert expected_subset.issubset(pins.keys())


@_install
def test_pin_layout_counts_match_pth_inventory():
    """For every shipped env, the PIN layout_count must equal the number of
    base (non-reverse) PTH files for that env in the same directory."""
    pins = load_all()
    pth_files = list(DEFAULT_SMX_DIR.glob("*.pth"))
    for env, info in pins.items():
        base = [
            f for f in pth_files
            if f.stem.upper().startswith(env) and not f.stem.upper().endswith("R")
        ]
        # AU is "AU" prefix but variant stems are "AU1..AU4"; same prefix logic
        # applies to all envs. R-suffixed reverse layouts are derived in-engine
        # and excluded from layout_count.
        assert len(base) == info.layout_count, (
            f"{env}: PIN says {info.layout_count} base layouts, "
            f"found {len(base)} PTH files: {[f.stem for f in base]}"
        )


@_install
def test_blackwood_pin_bbox_matches_published_layout():
    """BL.pin world bbox should match the well-known Blackwood cell."""
    info = parse_pin(DEFAULT_SMX_DIR / "BL.pin")
    assert info.env == "BL"
    assert info.layout_count == 4
    assert info.x_min_m == pytest.approx(-1152.0, abs=1.0)
    assert info.x_max_m == pytest.approx(512.0, abs=1.0)
    assert info.y_min_m == pytest.approx(-832.0, abs=1.0)
    assert info.y_max_m == pytest.approx(832.0, abs=1.0)


@_install
def test_pin_bbox_contains_pth_centerline():
    """Every PTH centerline should fit within its env's PIN bbox."""
    from lfs_telemetry.telemetry.track.pth import parse_pth
    pins = load_all()
    for env, info in pins.items():
        for pth_file in DEFAULT_SMX_DIR.glob(f"{env}*.pth"):
            try:
                path = parse_pth(pth_file)
            except Exception:
                continue
            if path.num_nodes == 0:
                continue
            xy = path.pos[:, :2]
            # Allow a small tolerance — some PTH segments (pit lane) can poke
            # marginally outside the published bbox.
            assert info.contains_xy(
                float(xy[:, 0].min()), float(xy[:, 1].min()), margin_m=20.0
            ), f"{pth_file.stem} min outside {env} bbox"
            assert info.contains_xy(
                float(xy[:, 0].max()), float(xy[:, 1].max()), margin_m=20.0
            ), f"{pth_file.stem} max outside {env} bbox"
