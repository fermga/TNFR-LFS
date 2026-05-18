"""Tests for the LFS ``.knw`` (AI knowledge) parser."""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from lfs_telemetry.telemetry.track.knw import (
    DEFAULT_KNW_DIR,
    HEADER_SIZE,
    KNW_MAGIC,
    KNW_VERSION,
    RECORD_SIZE,
    KnwSegment,
    list_knw_files,
    load_all_for_layout,
    load_for,
    parse_knw,
    parse_knw_bytes,
)

# ---------------------------------------------------------------------------
# Synthetic file builder
# ---------------------------------------------------------------------------


def _build_knw(
    *,
    version: int = KNW_VERSION,
    build_stamp: bytes = b"\x01\x02\x03\x04",
    ai_seed: int = 12345,
    lap_factor: float = 0.5,
    sm_a: float = 60.0,
    sm_b: float = 50.0,
    car_constant: float = 42.0,
    pth_nodes: int = 100,
    segments: list[tuple[int, int, int, float, float]] | None = None,
) -> bytes:
    """Pack a synthetic .knw file. ``segments`` items: (flags, n_start, n_end, delta, lat)."""
    if segments is None:
        segments = [(0xF8, 4, 13, 0.01, -0.4),
                    (0x108, 13, 28, -0.02, 0.8),
                    (0x208, 28, 4, 0.0, 1.6)]   # closes the loop
    packed = (pth_nodes & 0xFFFF) | ((len(segments) & 0xFFFF) << 16)
    out = bytearray()
    out += KNW_MAGIC
    out += struct.pack("<H", version)
    out += build_stamp
    out += struct.pack("<IffffI", ai_seed, lap_factor, sm_a, sm_b,
                       car_constant, packed)
    for flags, n_start, n_end, delta, lat in segments:
        out += struct.pack("<IIIIff", 0, flags, n_start, n_end, delta, lat)
    return bytes(out)


# ---------------------------------------------------------------------------
# Synthetic tests
# ---------------------------------------------------------------------------


def test_parse_round_trip_basic():
    data = _build_knw()
    info = parse_knw_bytes(data, layout="XX1", car="FBM")
    assert info.version == KNW_VERSION
    assert info.layout == "XX1"
    assert info.car == "FBM"
    assert info.pth_node_count == 100
    assert info.segment_count == 3
    assert len(info.segments) == 3
    assert info.car_constant == pytest.approx(42.0)
    assert info.speed_metric_a_ms == pytest.approx(60.0)
    assert info.speed_metric_a_kmh == pytest.approx(60.0 * 3.6)
    seg0 = info.segments[0]
    assert isinstance(seg0, KnwSegment)
    assert seg0.node_start == 4
    assert seg0.node_end == 13
    assert seg0.lateral_offset_m == pytest.approx(-0.4)


def test_parse_bad_magic_rejected():
    bad = b"NOPE!!" + b"\x00" * (HEADER_SIZE + RECORD_SIZE)
    with pytest.raises(ValueError, match="not an LFS"):
        parse_knw_bytes(bad)


def test_parse_too_short_rejected():
    with pytest.raises(ValueError, match="too short"):
        parse_knw_bytes(b"LFSKNW\x07\x00")


def test_parse_unaligned_body_rejected():
    data = _build_knw() + b"\x00" * 3  # break alignment
    with pytest.raises(ValueError, match="not a multiple"):
        parse_knw_bytes(data)


def test_unexpected_version_warns_but_parses(caplog):
    # 0x4242 is not a known LFS .knw version → should warn but parse.
    data = _build_knw(version=0x4242)
    with caplog.at_level("WARNING"):
        info = parse_knw_bytes(data)
    assert info.version == 0x4242
    assert any("unexpected .knw version" in r.message for r in caplog.records)


def test_known_alternate_version_silent(caplog):
    # 0x0600 is the FE/KY format observed in the canonical install.
    data = _build_knw(version=0x0600)
    with caplog.at_level("WARNING"):
        info = parse_knw_bytes(data)
    assert info.version == 0x0600
    assert not any("unexpected .knw version" in r.message for r in caplog.records)


def test_chain_continuity_synthetic():
    data = _build_knw()
    info = parse_knw_bytes(data)
    # node_end[i] must equal node_start[i+1] for an interior chain.
    for a, b in zip(info.segments, info.segments[1:], strict=False):
        assert a.node_end == b.node_start


def test_parse_file_round_trip(tmp_path: Path):
    p = tmp_path / "ZZ9_TST.knw"
    p.write_bytes(_build_knw())
    info = parse_knw(p)
    assert info.layout == "ZZ9"
    assert info.car == "TST"


# ---------------------------------------------------------------------------
# Real install (skipped if .knw dir absent)
# ---------------------------------------------------------------------------

_HAS_INSTALL = DEFAULT_KNW_DIR.exists()
pytestmark_install = pytest.mark.skipif(
    not _HAS_INSTALL, reason="LFS install knw dir not found"
)


@pytestmark_install
def test_install_full_inventory_parses():
    """Every .knw in the canonical install parses without error."""
    files = list_knw_files()
    assert len(files) > 1000, "expected the canonical 1718-file install"
    failures: list[tuple[Path, str]] = []
    for p in files:
        try:
            parse_knw(p)
        except Exception as exc:  # noqa: BLE001
            failures.append((p, str(exc)))
    assert not failures, f"{len(failures)} files failed: {failures[:3]}"


@pytestmark_install
def test_install_chain_continuity_holds_everywhere():
    """node_end[i] == node_start[i+1] across all install files."""
    bad: list[tuple[str, int]] = []
    for p in list_knw_files():
        info = parse_knw(p)
        for i, (a, b) in enumerate(zip(info.segments, info.segments[1:], strict=False)):
            if a.node_end != b.node_start:
                bad.append((p.name, i))
                break
    assert not bad, f"chain broken in {len(bad)} files (sample: {bad[:5]})"


@pytestmark_install
def test_install_segment_count_matches_header():
    """packed segment_count from the header record matches len(segments)."""
    for p in list_knw_files():
        info = parse_knw(p)
        assert info.segment_count == len(info.segments), p.name


@pytestmark_install
def test_install_per_car_constant_is_car_invariant():
    """The header ``car_constant`` field depends on the car, not the layout.

    Across every layout for which we have both AS1_FBM and AS1_BF1, the
    FBM file should show the same car_constant as any other FBM file,
    and ditto for BF1 — confirming our field-5 identification.
    """
    by_car: dict[str, set[float]] = {}
    for p in list_knw_files():
        info = parse_knw(p)
        by_car.setdefault(info.car, set()).add(round(info.car_constant, 3))
    # Every car should have ONE unique car_constant value across all layouts.
    multi = {c: vs for c, vs in by_car.items() if len(vs) > 1}
    assert not multi, f"car_constant varies per car for: {multi}"


@pytestmark_install
def test_install_load_all_for_layout_as1():
    cars = load_all_for_layout("AS1")
    if not cars:
        pytest.skip("AS1 layout not installed")
    assert "FBM" in cars or "XFG" in cars
    for car, info in cars.items():
        assert info.layout == "AS1"
        assert info.car == car


@pytestmark_install
def test_install_load_for_bl1_fbm():
    info = load_for("BL1", "FBM")
    if info is None:
        pytest.skip("BL1_FBM.knw not installed")
    assert info.layout == "BL1"
    assert info.car == "FBM"
    assert info.pth_node_count > 100  # BL1 has hundreds of PTH nodes
    assert info.segment_count == len(info.segments)
