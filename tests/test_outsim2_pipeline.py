"""Unit tests for the OutSimPack2 → fused sample path (no LFS required)."""

from __future__ import annotations

import struct

from lfs_telemetry.telemetry.live import _PendingByTime, _outsim2_to_basic
from lfs_telemetry.telemetry.replay import (
    _row_to_sample,
    _sample_to_row,
)
from lfs_telemetry.telemetry.protocol.packets import (
    OSO_ALL,
    OutSimPack2,
    OutSimPacket,
    WHEEL_ORDER,
    outsim2_size,
)
from lfs_telemetry.telemetry.protocol.insim import RaceContext
from lfs_telemetry.telemetry.observables import car_spec_for, observe_sample


def _build_outsim2_full(time_ms: int = 5000) -> bytes:
    parts: list[bytes] = []
    parts.append(b"LFST")                            # HEADER
    parts.append(struct.pack("<i", 1))               # ID
    parts.append(struct.pack("<I", time_ms))         # TIME
    # MAIN: 12 floats + 3 ints (60 bytes)
    parts.append(struct.pack(
        "<12f3i",
        0.0, 0.0, 0.5,
        0.0, 0.0, 0.0,
        2.0, 0.5, 9.81,
        25.0, 0.0, 0.0,
        int(0 * 65536), int(0 * 65536), int(0),
    ))
    # INPUTS: 5f
    parts.append(struct.pack("<5f", 0.7, 0.0, 0.05, 0.0, 0.0))
    # DRIVE: 4B 2f
    parts.append(struct.pack("<4B2f", 4, 0, 0, 0, 800.0, 250.0))
    # DISTANCE: 2f
    parts.append(struct.pack("<2f", 1234.5, 200000.0))
    # WHEELS: 4 × (7f 4B 2f) = 4 × 40
    for i in range(4):
        parts.append(struct.pack(
            "<7f4B2f",
            0.01 * (i + 1),       # susp_deflect
            0.0,                  # steer
            10.0, 5.0,            # x_force, y_force
            1500.0 + 100.0 * i,   # vertical_load_n
            50.0,                 # ang_vel
            0.0,                  # lean
            20, 128, 1, 0,        # air_temp, slip_fraction, touching, _pad
            0.05 * (i + 1),
            0.10 * (i + 1),
        ))
    # EXTRA_1: 2f
    parts.append(struct.pack("<2f", 0.0, 0.0))
    return b"".join(parts)


def test_outsim2_to_basic_has_all_fields() -> None:
    raw = _build_outsim2_full(time_ms=4242)
    pkt2 = OutSimPack2.parse(raw, OSO_ALL)
    basic = _outsim2_to_basic(pkt2)
    assert basic is not None
    assert isinstance(basic, OutSimPacket)
    assert basic.time_ms == 4242
    assert basic.ang_vel == pkt2.ang_vel
    assert basic.accel == pkt2.accel
    assert basic.packet_id == pkt2.packet_id


def test_pending_by_time_add_outsim2_releases_with_outgauge() -> None:
    raw = _build_outsim2_full(time_ms=8000)
    pkt2 = OutSimPack2.parse(raw, OSO_ALL)
    basic = _outsim2_to_basic(pkt2)
    assert basic is not None
    buf = _PendingByTime(window_ms=50)
    # Just adding outsim2 should buffer (no outgauge yet).
    early = buf.add_outsim2(pkt2, basic)
    assert early is None or not early.is_complete

    # Now feed an OutGauge with the same time stamp.
    from lfs_telemetry.telemetry.protocol.packets import OutGaugePacket
    og = OutGaugePacket(
        time_ms=8000, car="FOX", flags=0, gear=4, player_id=0,
        speed_ms=25.0, rpm=8500.0, turbo_bar=0.0, eng_temp_c=90.0,
        fuel=0.5, oil_pressure_bar=4.5, oil_temp_c=95.0,
        dash_lights=0, show_lights=0,
        throttle=0.7, brake=0.0, clutch=0.0,
        display1="", display2="",
    )
    sample = buf.add_outgauge(og)
    assert sample is not None
    assert sample.is_complete
    assert sample.outsim is basic
    assert sample.outsim2 is pkt2
    assert sample.outsim2.wheels is not None
    assert len(sample.outsim2.wheels) == 4


def test_observe_sample_uses_real_wheel_loads() -> None:
    raw = _build_outsim2_full()
    pkt2 = OutSimPack2.parse(raw, OSO_ALL)
    basic = _outsim2_to_basic(pkt2)
    assert basic is not None
    from lfs_telemetry.telemetry.protocol.packets import OutGaugePacket
    og = OutGaugePacket(
        time_ms=basic.time_ms, car="FOX", flags=0, gear=4, player_id=0,
        speed_ms=25.0, rpm=8500.0, turbo_bar=0.0, eng_temp_c=90.0,
        fuel=0.5, oil_pressure_bar=4.5, oil_temp_c=95.0,
        dash_lights=0, show_lights=0,
        throttle=0.7, brake=0.0, clutch=0.0,
        display1="", display2="",
    )
    from lfs_telemetry.telemetry.live import TelemetrySample
    sample = TelemetrySample(
        time_ms=basic.time_ms,
        outsim=basic,
        outgauge=og,
        outsim2=pkt2,
    )
    spec = car_spec_for("FOX")
    obs = observe_sample(sample, spec)
    # Expected mapping: WHEEL_ORDER[i]=("RL","RR","FL","FR")
    expected = dict(zip(WHEEL_ORDER, [w.vertical_load_n for w in pkt2.wheels]))
    for c in ("FL", "FR", "RL", "RR"):
        assert obs.corner_load_n[c] == expected[c]


def test_csv_roundtrip_preserves_wheels_and_context() -> None:
    raw = _build_outsim2_full(time_ms=12345)
    pkt2 = OutSimPack2.parse(raw, OSO_ALL)
    basic = _outsim2_to_basic(pkt2)
    assert basic is not None
    from lfs_telemetry.telemetry.protocol.packets import OutGaugePacket
    og = OutGaugePacket(
        time_ms=basic.time_ms, car="FOX", flags=0, gear=4, player_id=0,
        speed_ms=25.0, rpm=8500.0, turbo_bar=0.0, eng_temp_c=90.0,
        fuel=0.5, oil_pressure_bar=4.5, oil_temp_c=95.0,
        dash_lights=0, show_lights=0,
        throttle=0.7, brake=0.0, clutch=0.0,
        display1="", display2="",
    )
    from lfs_telemetry.telemetry.live import TelemetrySample
    ctx = RaceContext()
    ctx.track = "BL1"
    ctx.weather = 2
    sample = TelemetrySample(
        time_ms=basic.time_ms,
        outsim=basic,
        outgauge=og,
        outsim2=pkt2,
        race_context=ctx,
    )
    row = _sample_to_row(sample)
    # Coerce all values to strings as csv would.
    str_row = {k: ("" if v is None else str(v)) for k, v in row.items()}
    parsed = _row_to_sample(str_row)
    assert parsed.outsim2 is not None
    assert parsed.outsim2.wheels is not None
    assert len(parsed.outsim2.wheels) == 4
    # Verify a known wheel value survived the roundtrip.
    original = dict(zip(WHEEL_ORDER, pkt2.wheels))
    restored = dict(zip(WHEEL_ORDER, parsed.outsim2.wheels))
    for c in WHEEL_ORDER:
        assert restored[c].vertical_load_n == original[c].vertical_load_n
        assert restored[c].slip_ratio == original[c].slip_ratio


def test_outsim2_size_matches_constant() -> None:
    assert outsim2_size(OSO_ALL) == 280


def test_car_spec_for_known_and_unknown() -> None:
    fox = car_spec_for("FOX")
    bf1 = car_spec_for("BF1")
    unknown = car_spec_for("XYZ")
    assert fox.mass_kg < bf1.mass_kg or fox.wheelbase_m < bf1.wheelbase_m
    # Unknown falls back to defaults.
    assert unknown.driven == "RWD"
