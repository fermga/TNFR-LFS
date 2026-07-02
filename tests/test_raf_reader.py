"""Tests for the LFS RAF (Replay Analyser File) v2 reader.

These tests build minimal but spec-shaped RAF byte blobs in memory,
parse them through :mod:`lfs_telemetry.telemetry.raf`, and check that:

* the header parser reads the variable header offsets, strings and
  splits correctly;
* the per-block decoder converts G quantisation, position scaling and
  forward-vector heading per the documented formulas;
* :func:`split_into_laps` segments by track-ruler wrap;
* :func:`raf_to_lap_csvs` writes CSVs that :meth:`LapTelemetry.from_csv`
  loads without error.
"""

from __future__ import annotations

import math
import struct

import pytest

from lfs_telemetry.telemetry.raf import (
    parse_raf,
    parse_raf_header,
    raf_to_lap_csvs,
    split_into_laps,
)

_HEADER_SIZE = 1024
_BLOCK_SIZE = 192
_WHEEL_BLOCK_OFFSET = 64
_WHEEL_BLOCK_SIZE = 32


def _build_header(
    *,
    num_blocks: int,
    update_interval_ms: int = 10,
    short_track: str = "BL1",
    track: str = "Blackwood GP",
    car: str = "FBM",
    player: str = "Tester",
    weather: str = "Sunny",
    lfs_version: str = "0.7E",
    track_ruler_length_m: float = 5000.0,
    num_splits: int = 4,
    splits_ms: tuple[int, int, int, int] = (30000, 60000, 90000, 120000),
) -> bytes:
    buf = bytearray(_HEADER_SIZE)
    buf[0:6] = b"LFSRAF"
    buf[6] = 0  # game version
    buf[7] = 0  # game revision
    buf[8] = 2  # RAF version
    buf[9] = update_interval_ms
    struct.pack_into("<H", buf, 12, _HEADER_SIZE)
    struct.pack_into("<H", buf, 14, _BLOCK_SIZE)
    struct.pack_into("<H", buf, 16, _WHEEL_BLOCK_SIZE)
    struct.pack_into("<H", buf, 18, _WHEEL_BLOCK_OFFSET)
    struct.pack_into("<i", buf, 20, num_blocks)
    buf[24:24 + len(short_track)] = short_track.encode("latin-1")
    struct.pack_into("<f", buf, 28, track_ruler_length_m)
    buf[32:32 + len(player)] = player.encode("latin-1")
    buf[64:64 + len(car)] = car.encode("latin-1")
    buf[96:96 + len(track)] = track.encode("latin-1")
    # config @128, weather @144, lfs version @160
    buf[144:144 + len(weather)] = weather.encode("latin-1")
    buf[160:160 + len(lfs_version)] = lfs_version.encode("latin-1")
    buf[168] = 0  # player flags
    buf[169] = 4  # num wheels
    buf[170] = 1  # HLVC legal
    buf[171] = num_splits
    struct.pack_into("<iiii", buf, 172, *splits_ms)
    struct.pack_into("<f", buf, 188, 600.0)  # mass
    struct.pack_into("<f", buf, 192, 550.0)  # sprung mass
    struct.pack_into("<f", buf, 196, 50000.0)  # rear antiroll
    struct.pack_into("<f", buf, 200, 50000.0)  # front antiroll
    struct.pack_into("<f", buf, 204, 3.5)  # final drive
    buf[208] = 6  # num gears
    struct.pack_into(
        "<7f", buf, 212,
        3.5, 2.5, 1.8, 1.4, 1.1, 0.9, 0.0,
    )
    return bytes(buf)


def _build_block(
    *,
    throttle: float = 0.5,
    brake: float = 0.1,
    input_steer: float = 0.0,
    clutch: float = 0.0,
    handbrake: float = 0.0,
    gear: int = 3,
    lat_g_q: int = 20,    # -> 1 g
    fwd_g_q: int = -40,   # -> -2 g (braking)
    up_g_q: int = 20,     # -> 1 g (vertical)
    speed_ms: float = 40.0,
    car_distance_m: float = 0.0,
    pos_x_m: float = 100.0,
    pos_y_m: float = 200.0,
    pos_z_m: float = 0.5,
    engine_rads: float = 800.0,
    index_distance_m: float = 0.0,
    heading_rad: float = 0.0,
) -> bytes:
    buf = bytearray(_BLOCK_SIZE)
    struct.pack_into("<f", buf, 0, throttle)
    struct.pack_into("<f", buf, 4, brake)
    struct.pack_into("<f", buf, 8, input_steer)
    struct.pack_into("<f", buf, 12, clutch)
    struct.pack_into("<f", buf, 16, handbrake)
    buf[20] = gear & 0xFF
    struct.pack_into("<b", buf, 21, lat_g_q)
    struct.pack_into("<b", buf, 22, fwd_g_q)
    struct.pack_into("<b", buf, 23, up_g_q)
    struct.pack_into("<f", buf, 24, speed_ms)
    struct.pack_into("<f", buf, 28, car_distance_m)
    struct.pack_into("<iii", buf, 32,
                     int(pos_x_m * 65536),
                     int(pos_y_m * 65536),
                     int(pos_z_m * 65536))
    struct.pack_into("<f", buf, 44, engine_rads)
    struct.pack_into("<f", buf, 48, index_distance_m)
    # Right vector = (cos(h), sin(h), 0), forward = (-sin(h), cos(h), 0)
    # Heading uses atan2(-fx, fy) — so for heading=0 we want fx=0, fy=1.
    rx = int(math.cos(heading_rad) * 32767)
    ry = int(math.sin(heading_rad) * 32767)
    fx = int(-math.sin(heading_rad) * 32767)
    fy = int(math.cos(heading_rad) * 32767)
    struct.pack_into("<hhh", buf, 52, rx, ry, 0)
    struct.pack_into("<hhh", buf, 58, fx, fy, 0)
    # 4× wheel dynamic blocks: leave zero except small markers
    for i in range(4):
        wb = _WHEEL_BLOCK_OFFSET + i * _WHEEL_BLOCK_SIZE
        struct.pack_into("<f", buf, wb + 0, -0.05)   # susp deflect
        struct.pack_into("<f", buf, wb + 4, 0.0)     # steer
        struct.pack_into("<f", buf, wb + 8, 1500.0)  # vertical load
        struct.pack_into("<f", buf, wb + 12, 0.0)    # x force
        struct.pack_into("<f", buf, wb + 16, 200.0)  # y force
        struct.pack_into("<f", buf, wb + 20, 100.0)  # ang vel
        struct.pack_into("<f", buf, wb + 24, 0.0)    # lean
        buf[wb + 28] = 25      # air temp
        buf[wb + 29] = 50      # slip fraction
    return bytes(buf)


def test_parse_header_round_trip():
    blob = _build_header(num_blocks=3)
    head = parse_raf_header(blob)
    assert head.raf_version == 2
    assert head.update_interval_ms == 10
    assert head.header_size == _HEADER_SIZE
    assert head.block_size == _BLOCK_SIZE
    assert head.num_blocks == 3
    assert head.short_track_name == "BL1"
    assert head.track == "Blackwood GP"
    assert head.car == "FBM"
    assert head.player == "Tester"
    assert head.weather == "Sunny"
    assert head.lfs_version == "0.7E"
    assert head.num_wheels == 4
    assert head.num_splits == 4
    assert head.splits_ms == (30000, 60000, 90000, 120000)
    assert head.num_gears == 6
    assert len(head.gear_ratios) == 6
    assert head.gear_ratios[0] == pytest.approx(3.5)
    assert head.mass_kg == pytest.approx(600.0)
    assert head.track_ruler_length_m == pytest.approx(5000.0)


def test_rejects_non_raf():
    with pytest.raises(ValueError, match="LFSRAF"):
        parse_raf_header(b"\x00" * _HEADER_SIZE)


def test_rejects_future_version():
    blob = bytearray(_build_header(num_blocks=1))
    blob[8] = 99
    with pytest.raises(ValueError, match="version"):
        parse_raf_header(bytes(blob))


def test_rejects_undersized_block_size():
    # A malformed/truncated header reporting a block smaller than the
    # fixed layout must raise ValueError (not crash with struct.error)
    # so callers like raf-import can surface a friendly message.
    blob = bytearray(_build_header(num_blocks=1))
    struct.pack_into("<H", blob, 14, _BLOCK_SIZE - 8)
    with pytest.raises(ValueError, match="block_size"):
        parse_raf_header(bytes(blob))


def test_parse_block_g_and_position(tmp_path):
    blob = _build_header(num_blocks=2)
    body = _build_block(
        lat_g_q=20, fwd_g_q=-40, up_g_q=20,
        pos_x_m=12.5, pos_y_m=-3.25, pos_z_m=0.5,
        engine_rads=2.0 * math.pi * 100.0,  # → 6000 rpm
        heading_rad=0.0,
    ) + _build_block(index_distance_m=10.0)
    path = tmp_path / "test.raf"
    path.write_bytes(blob + body)
    head, rows = parse_raf(path)
    assert head.num_blocks == 2
    assert len(rows) == 2
    r = rows[0]
    g = 9.80665
    assert r["accel_y"] == pytest.approx(1.0 * g, rel=1e-3)
    assert r["accel_x"] == pytest.approx(-2.0 * g, rel=1e-3)
    assert r["accel_z"] == pytest.approx(1.0 * g, rel=1e-3)
    assert r["pos_x"] == pytest.approx(12.5, rel=1e-4)
    assert r["pos_y"] == pytest.approx(-3.25, rel=1e-4)
    assert r["pos_z"] == pytest.approx(0.5, rel=1e-4)
    assert r["rpm"] == pytest.approx(6000.0, rel=1e-3)
    assert r["heading"] == pytest.approx(0.0, abs=1e-3)
    assert r["car"] == "FBM"
    assert r["ctx_track"] == "Blackwood GP"
    assert r["time_ms"] == 0
    assert rows[1]["time_ms"] == 10
    # Per-wheel
    assert r["wheel_FL_vertical_load_n"] == pytest.approx(1500.0)
    assert r["wheel_RL_air_temp_c"] == 25
    assert r["wheel_RR_slip_fraction"] == 50


def test_split_into_laps_wraps_on_index_distance(tmp_path):
    """Two laps separated by a ruler wrap should split cleanly."""
    blob = _build_header(num_blocks=6, track_ruler_length_m=1000.0)
    body = b"".join([
        # lap 1
        _build_block(index_distance_m=100.0, car_distance_m=100.0),
        _build_block(index_distance_m=500.0, car_distance_m=500.0),
        _build_block(index_distance_m=950.0, car_distance_m=950.0),
        # wrap → lap 2
        _build_block(index_distance_m=20.0, car_distance_m=1020.0),
        _build_block(index_distance_m=600.0, car_distance_m=1600.0),
        _build_block(index_distance_m=980.0, car_distance_m=1980.0),
    ])
    path = tmp_path / "wrap.raf"
    path.write_bytes(blob + body)
    head, rows = parse_raf(path)
    laps = split_into_laps(head, rows)
    assert len(laps) == 2
    assert [len(lap) for lap in laps] == [3, 3]
    # Per-lap distance/time anchored
    assert laps[0][0]["current_lap_dist_m"] == pytest.approx(0.0)
    assert laps[1][0]["current_lap_dist_m"] == pytest.approx(0.0)
    assert laps[1][0]["time_ms"] == 0
    assert laps[1][-1]["time_ms"] == 20  # 2 samples × 10 ms


def test_raf_to_lap_csvs_writes_loadable_csvs(tmp_path):
    blob = _build_header(num_blocks=420, track_ruler_length_m=1000.0)
    blocks = []
    # 200 samples covering the full ruler (a real flying lap), then
    # wrap, then 220 samples covering only ~110 m (a partial in-lap
    # fragment that the replay was cut short on).
    for i in range(200):
        blocks.append(_build_block(
            index_distance_m=float(i) * 5.0,         # 0 → 995 m (≈ ruler)
            car_distance_m=float(i) * 5.0,
        ))
    for i in range(220):
        blocks.append(_build_block(
            index_distance_m=float(i) * 0.5,         # 0 → 109.5 m
            car_distance_m=1000.0 + float(i) * 0.5,
        ))
    path = tmp_path / "stint.raf"
    path.write_bytes(blob + b"".join(blocks))
    written = raf_to_lap_csvs(path, out_dir=tmp_path / "out")
    # In-lap fragment (≈ 11 % of ruler) dropped by default; full lap
    # kept regardless of being the first segment in the file.
    assert len(written) == 1
    csv_path = written[0]
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8").splitlines()
    # First line is the schema comment
    assert text[0].startswith("# lfs-telemetry telemetry schema=")
    # Header row should contain our standard fields
    assert "time_ms" in text[1]
    assert "wheel_FL_vertical_load_n" in text[1]
    # CSV must contain all 200 rows of the real flying lap
    # (header + schema = 2 lines, then 200 data rows)
    assert len(text) == 2 + 200


def test_raf_to_lap_csvs_keeps_partials_with_zero_frac(tmp_path):
    """``min_lap_distance_frac=0.0`` keeps every segment that meets the
    sample count threshold, mirroring the legacy un-filtered behaviour."""
    blob = _build_header(num_blocks=420, track_ruler_length_m=1000.0)
    blocks = []
    for i in range(200):
        blocks.append(_build_block(
            index_distance_m=float(i) * 5.0,
            car_distance_m=float(i) * 5.0,
        ))
    for i in range(220):
        blocks.append(_build_block(
            index_distance_m=float(i) * 0.5,
            car_distance_m=1000.0 + float(i) * 0.5,
        ))
    path = tmp_path / "stint.raf"
    path.write_bytes(blob + b"".join(blocks))
    written = raf_to_lap_csvs(
        path, out_dir=tmp_path / "out",
        min_lap_distance_frac=0.0,
    )
    assert len(written) == 2


def test_raf_to_lap_csvs_loads_into_lap_telemetry(tmp_path):
    from lfs_telemetry.telemetry import LapTelemetry

    blob = _build_header(num_blocks=300, track_ruler_length_m=1000.0)
    blocks = []
    for i in range(150):
        blocks.append(_build_block(
            index_distance_m=float(i) * 6.0,
            car_distance_m=float(i) * 6.0,
        ))
    for i in range(150):
        blocks.append(_build_block(
            index_distance_m=float(i) * 6.0,
            car_distance_m=900.0 + float(i) * 6.0,
        ))
    path = tmp_path / "stint.raf"
    path.write_bytes(blob + b"".join(blocks))
    written = raf_to_lap_csvs(path, out_dir=tmp_path / "laps")
    assert written, "expected at least one lap CSV"
    lap = LapTelemetry.from_csv(written[0])
    assert len(lap.raw) > 0
    assert "speed_ms" in lap.raw.columns
    # Enrichment should not crash on RAF-sourced data
    df = lap.enriched
    assert len(df) == len(lap.raw)
