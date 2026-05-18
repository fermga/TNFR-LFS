import struct

import pytest

from lfs_telemetry.telemetry.protocol.packets import (
    OUTGAUGE_SIZE,
    OUTSIM_SIZE,
    OutGaugePacket,
    OutSimPacket,
    build_isi_packet,
)


def _make_outsim(time_ms: int = 1234) -> bytes:
    return struct.pack(
        "<I" + "f" * 3 + "f" * 3 + "f" * 3 + "f" * 3 + "i" * 3,
        time_ms,
        0.0, 0.0, 0.5,                # ang_vel
        0.1, 0.02, -0.03,             # heading, pitch, roll
        2.0, 1.5, 9.81,               # accel (long, lat, vert)
        25.0, 0.0, 0.0,               # vel
        int(100 * 65536), int(50 * 65536), 0,  # pos
    )


def _make_outgauge(time_ms: int = 1234) -> bytes:
    return struct.pack(
        "<I4sHBB" + "f" * 7 + "II" + "fff" + "16s16s",
        time_ms,
        b"FOX\x00",
        0,
        4,
        0,
        25.0, 8500.0, 0.0, 90.0, 0.6, 4.5, 95.0,
        0, 0,
        0.8, 0.0, 0.0,
        b"GEAR 3".ljust(16, b"\x00"),
        b"".ljust(16, b"\x00"),
    )


def test_outsim_roundtrip():
    raw = _make_outsim()
    assert len(raw) == OUTSIM_SIZE
    pkt = OutSimPacket.parse(raw)
    assert pkt.time_ms == 1234
    assert pkt.ang_vel == pytest.approx((0.0, 0.0, 0.5), abs=1e-5)
    assert pkt.accel == pytest.approx((2.0, 1.5, 9.81), abs=1e-4)
    assert pkt.pos[0] == pytest.approx(100.0, abs=1e-5)
    assert pkt.packet_id is None


def test_outgauge_roundtrip():
    raw = _make_outgauge()
    assert len(raw) == OUTGAUGE_SIZE
    pkt = OutGaugePacket.parse(raw)
    assert pkt.time_ms == 1234
    assert pkt.car == "FOX"
    assert pkt.gear == 4
    assert pkt.rpm == pytest.approx(8500.0)
    assert pkt.throttle == pytest.approx(0.8, abs=1e-6)
    assert pkt.display1 == "GEAR 3"


def test_isi_packet_size():
    pkt = build_isi_packet(udp_port=30000, iname="tnfr")
    assert len(pkt) == 44
    # InSim v9+: size byte is bytes/4.
    assert pkt[0] == 44 // 4
    assert pkt[1] == 1  # ISP_ISI


# ---------------------------------------------------------------------------
# OutSimPack2 + InSim packet tests
# ---------------------------------------------------------------------------

from lfs_telemetry.telemetry.protocol.packets import (  # noqa: E402
    ISP_LAP,
    ISP_NPL,
    ISP_STA,
    OSO_ALL,
    OUTSIMPACK2_FULL_SIZE,
    WHEEL_ORDER,
    InSimLap,
    InSimNewPlayer,
    InSimState,
    OutSimPack2,
    outsim2_size,
)


def _build_outsimpack2_all() -> bytes:
    """Build a synthetic 280-byte OutSimPack2 with OSO_ALL."""
    parts = []
    # HEADER (4)
    parts.append(b"LFST")
    # ID (4)
    parts.append(struct.pack("<i", 7))
    # TIME (4)
    parts.append(struct.pack("<I", 12345))
    # MAIN (60): 12 floats + 3 ints
    parts.append(struct.pack(
        "<12f3i",
        0.0, 0.0, 0.5,
        0.1, 0.02, -0.03,
        2.0, 1.5, 9.81,
        25.0, 0.0, 0.0,
        int(100 * 65536), int(50 * 65536), 0,
    ))
    # INPUTS (20): 5 floats
    parts.append(struct.pack("<5f", 0.8, 0.1, 0.05, 0.0, 0.0))
    # DRIVE (12): 4 bytes + 2 floats
    parts.append(struct.pack("<4B2f", 4, 0, 0, 0, 800.0, 320.0))
    # DISTANCE (8): 2 floats
    parts.append(struct.pack("<2f", 1234.5, 9876.0))
    # WHEELS (160): 4 × 40
    for i in range(4):
        parts.append(struct.pack(
            "<7f4B2f",
            0.01 * (i + 1),  # susp_deflect
            0.0,             # steer
            100.0 * i,       # x_force
            -50.0 * i,       # y_force
            1500.0 + i * 100.0,  # vertical_load
            45.0,            # ang_vel
            0.0,             # lean_rel_road
            65 + i,          # air_temp
            128,             # slip_fraction byte
            1,               # touching
            0,               # spare
            0.05 * (i + 1),  # slip_ratio
            0.10 * (i + 1),  # tan_slip_angle
        ))
    # EXTRA_1 (8): 2 floats
    parts.append(struct.pack("<2f", 12.5, 0.0))
    return b"".join(parts)


def test_outsim2_full_size():
    assert outsim2_size(OSO_ALL) == OUTSIMPACK2_FULL_SIZE == 280


def test_outsimpack2_full_roundtrip():
    raw = _build_outsimpack2_all()
    assert len(raw) == 280
    pkt = OutSimPack2.parse(raw, OSO_ALL)
    assert pkt.header == "LFST"
    assert pkt.packet_id == 7
    assert pkt.time_ms == 12345
    assert pkt.gear == 4
    assert pkt.throttle == pytest.approx(0.8)
    assert pkt.brake == pytest.approx(0.1)
    assert pkt.current_lap_dist_m == pytest.approx(1234.5)
    assert pkt.steer_torque_nm == pytest.approx(12.5)
    assert pkt.pos == pytest.approx((100.0, 50.0, 0.0), abs=1e-5)
    assert pkt.wheels is not None and len(pkt.wheels) == 4
    # Wheels are in WHEEL_ORDER (RL, RR, FL, FR).
    assert WHEEL_ORDER == ("RL", "RR", "FL", "FR")
    assert pkt.wheels[0].susp_deflect_m == pytest.approx(0.01)
    assert pkt.wheels[3].susp_deflect_m == pytest.approx(0.04)
    assert pkt.wheels[2].vertical_load_n == pytest.approx(1700.0)
    assert pkt.wheels[1].slip_ratio == pytest.approx(0.10)
    assert pkt.wheels[0].slip_fraction == pytest.approx(128 / 255.0)
    assert pkt.wheels[0].touching == 1


def test_outsimpack2_partial_opts():
    # Header + ID + TIME only (12 bytes).
    OSO_BASIC = 0x001 | 0x002 | 0x004
    assert outsim2_size(OSO_BASIC) == 12
    raw = b"OSP2" + struct.pack("<i", 42) + struct.pack("<I", 9999)
    pkt = OutSimPack2.parse(raw, OSO_BASIC)
    assert pkt.header == "OSP2"
    assert pkt.packet_id == 42
    assert pkt.time_ms == 9999
    assert pkt.wheels is None
    assert pkt.gear is None


def test_insim_state_parse():
    # Size=28, Type=ISP_STA, ReqI=0, Zero=0; payload = 24 bytes.
    payload = struct.pack(
        "<fHBBBBBBBB BB 6s BB",
        1.0,        # replay speed
        0x0008,     # flags
        0,          # in_game_cam
        2,          # view_plid
        4,          # num_players
        4,          # num_connections
        0,          # num_finished
        1,          # race_in_progress
        0,          # qual_minutes
        10,         # race_laps
        0, 0,
        b"BL1\x00\x00\x00",
        2,          # weather
        1,          # wind
    )
    packet = struct.pack("<BBBB", 28, ISP_STA, 0, 0) + payload
    sta = InSimState.parse(packet)
    assert sta.race_in_progress == 1
    assert sta.race_laps == 10
    assert sta.track == "BL1"
    assert sta.weather == 2
    assert sta.view_plid == 2
    assert sta.num_players == 4


def test_insim_npl_parse():
    payload = struct.pack(
        "<BBH 24s 8s 4s 16s 4B 4B 4B 4B",
        1,          # UCID
        1,          # PType
        0x0040,     # Flags (PIF_AUTOGEARS)
        b"Driver".ljust(24, b"\x00"),
        b"AB12CDE".ljust(8, b"\x00"),
        b"FOX\x00",
        b"default".ljust(16, b"\x00"),
        2, 2, 2, 2,         # tyres
        0, 0, 0, 0,         # H_Mass, H_TRes, Model, Pass
        0, 0, 0, 0,         # RWAdj, FWAdj, sp[2]
        0x01, 1, 0, 50,     # SetF (SETF_SYMM_WHEELS), NumP, Config, Fuel%
    )
    plid = 7
    packet = struct.pack("<BBBB", 76, ISP_NPL, 0, plid) + payload
    npl = InSimNewPlayer.parse(packet)
    assert npl.player_id == 7
    assert npl.connection_id == 1
    assert npl.car_name == "FOX"
    assert npl.player_name == "Driver"
    assert npl.tyres == (2, 2, 2, 2)
    assert npl.set_flags == 0x01
    assert npl.fuel_pct == 50


def test_insim_lap_parse():
    payload = struct.pack(
        "<IIHH BBBB",
        93450,          # lap time ms
        279310,         # elapsed time ms
        3,              # laps_done
        0,              # flags
        0, 0, 1, 100,   # sp0, penalty, num_stops, fuel200
    )
    plid = 5
    packet = struct.pack("<BBBB", 20, ISP_LAP, 0, plid) + payload
    lap = InSimLap.parse(packet)
    assert lap.player_id == 5
    assert lap.lap_time_ms == 93450
    assert lap.elapsed_time_ms == 279310
    assert lap.laps_done == 3
    assert lap.num_stops == 1
    assert lap.fuel200 == 100


# ---------------------------------------------------------------------------
# decode_car_id + mod / new-IS_* parser coverage
# ---------------------------------------------------------------------------

from lfs_telemetry.telemetry.protocol.packets import (  # noqa: E402
    ISP_CNL,
    ISP_MAL,
    ISP_NCN,
    ISP_PLA,
    ISP_SLC,
    ISP_SMALL,
    PITLANE_ENTER,
    PITLANE_EXIT,
    SMALL_VTA,
    VOTE_RESTART,
    InSimConnectionLeft,
    InSimModsAllowed,
    InSimNewConnection,
    InSimPitLane,
    InSimSelectedCar,
    InSimSmall,
    InSimVoteAction,
    decode_car_id,
)


def test_decode_car_id_stock():
    # 3-char ASCII + NUL pad -> upper-case short name.
    assert decode_car_id(b"FOX\x00") == "FOX"
    assert decode_car_id(b"FBM\x00") == "FBM"
    assert decode_car_id(b"XRG\x00") == "XRG"
    assert decode_car_id(b"BF1\x00") == "BF1"


def test_decode_car_id_mod_little_endian():
    # SkinIDs from Detect&Monitor cars/mod_sizes.car keys.
    # Stored as little-endian uint32 in LFS packets, serialised as 6-hex.
    assert decode_car_id(bytes([0xFE, 0xF1, 0x56, 0x00])) == "56f1fe"
    assert decode_car_id(bytes([0x0B, 0x35, 0x7C, 0x00])) == "7c350b"
    assert decode_car_id(bytes([0xC6, 0xFE, 0xCC, 0x00])) == "ccfec6"
    assert decode_car_id(bytes([0xA4, 0xC2, 0xF3, 0x00])) == "f3c2a4"


def test_decode_car_id_empty():
    assert decode_car_id(b"") == ""
    assert decode_car_id(b"\x00\x00\x00\x00") == ""


def test_decode_car_id_unknown_ascii_falls_to_hex():
    # 3 ASCII letters but not in stock list -> treat as mod SkinID.
    # b"ZZZ\x00" little-endian uint32 = 0x005A5A5A -> "5a5a5a".
    assert decode_car_id(b"ZZZ\x00") == "5a5a5a"


def test_outgauge_with_mod_car():
    """OutGauge.Car[4] holding a binary SkinID must round-trip as 6-hex."""
    raw = struct.pack(
        "<I4sHBB" + "f" * 7 + "II" + "fff" + "16s16s",
        2000,
        bytes([0xFE, 0xF1, 0x56, 0x00]),  # mod 56f1fe
        0, 4, 0,
        25.0, 8500.0, 0.0, 90.0, 0.6, 4.5, 95.0,
        0, 0,
        0.8, 0.0, 0.0,
        b"".ljust(16, b"\x00"),
        b"".ljust(16, b"\x00"),
    )
    pkt = OutGaugePacket.parse(raw)
    assert pkt.car == "56f1fe"


def test_insim_npl_with_mod():
    payload = struct.pack(
        "<BBH 24s 8s 4s 16s 4B 4B 4B 4B",
        2, 1, 0,
        b"Driver".ljust(24, b"\x00"),
        b"".ljust(8, b"\x00"),
        bytes([0x0B, 0x35, 0x7C, 0x00]),  # mod 7c350b
        b"default".ljust(16, b"\x00"),
        2, 2, 2, 2,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 1, 0, 50,
    )
    packet = struct.pack("<BBBB", 76, ISP_NPL, 0, 3) + payload
    npl = InSimNewPlayer.parse(packet)
    assert npl.car_name == "7c350b"


def test_insim_ncn_parse():
    payload = struct.pack(
        "<24s24sBBBB",
        b"alice".ljust(24, b"\x00"),
        b"Alice".ljust(24, b"\x00"),
        1,    # admin
        3,    # total
        4,    # flags (remote)
        0,
    )
    packet = struct.pack("<BBBB", 14, ISP_NCN, 0, 7) + payload
    ncn = InSimNewConnection.parse(packet)
    assert ncn.connection_id == 7
    assert ncn.user_name == "alice"
    assert ncn.player_name == "Alice"
    assert ncn.admin == 1
    assert ncn.total == 3
    assert ncn.flags == 4


def test_insim_cnl_parse():
    packet = struct.pack("<BBBB BBBB", 2, ISP_CNL, 0, 5, 1, 2, 0, 0)
    cnl = InSimConnectionLeft.parse(packet)
    assert cnl.connection_id == 5
    assert cnl.reason == 1
    assert cnl.total == 2


def test_insim_slc_stock_and_mod():
    # Stock
    pkt_stock = struct.pack(
        "<BBBB 4s", 2, ISP_SLC, 0, 4, b"FBM\x00")
    slc = InSimSelectedCar.parse(pkt_stock)
    assert slc.connection_id == 4
    assert slc.car_name == "FBM"
    # Mod
    pkt_mod = struct.pack(
        "<BBBB 4s", 2, ISP_SLC, 0, 9,
        bytes([0xC6, 0xFE, 0xCC, 0x00]))
    slc = InSimSelectedCar.parse(pkt_mod)
    assert slc.connection_id == 9
    assert slc.car_name == "ccfec6"
    # Empty
    pkt_empty = struct.pack(
        "<BBBB 4s", 2, ISP_SLC, 0, 0, b"\x00\x00\x00\x00")
    slc = InSimSelectedCar.parse(pkt_empty)
    assert slc.car_name == ""


def test_insim_pla_parse():
    pkt = struct.pack(
        "<BBBB BBBB", 2, ISP_PLA, 0, 6, PITLANE_ENTER, 0, 0, 0)
    pla = InSimPitLane.parse(pkt)
    assert pla.player_id == 6
    assert pla.fact == PITLANE_ENTER
    pkt = struct.pack(
        "<BBBB BBBB", 2, ISP_PLA, 0, 6, PITLANE_EXIT, 0, 0, 0)
    pla = InSimPitLane.parse(pkt)
    assert pla.fact == PITLANE_EXIT


def test_insim_mal_parse_mixed_ids():
    ids = [0x56F1FE, 0x7C350B, 0xCCFEC6]
    payload_header = struct.pack("<BBBB", 2, 0xFF, 0, 0)  # UCID, Flags, Sp2, Sp3
    payload_mods = b"".join(struct.pack("<I", i) for i in ids)
    size_bytes = 8 + 4 * len(ids)  # 20
    packet = struct.pack("<BBBB", size_bytes // 4, ISP_MAL, 0, len(ids)) \
        + payload_header + payload_mods
    mal = InSimModsAllowed.parse(packet)
    assert mal.connection_id == 2
    assert mal.flags == 0xFF
    assert mal.mod_ids == ("56f1fe", "7c350b", "ccfec6")


def test_insim_mal_unrestricted():
    packet = struct.pack("<BBBB BBBB", 2, ISP_MAL, 0, 0, 0, 0, 0, 0)
    mal = InSimModsAllowed.parse(packet)
    assert mal.mod_ids == ()


def test_insim_small_vote_action():
    packet = struct.pack("<BBBB I", 2, ISP_SMALL, 0, SMALL_VTA, VOTE_RESTART)
    sm = InSimSmall.parse(packet)
    assert isinstance(sm, InSimVoteAction)
    assert sm.action == VOTE_RESTART
    assert sm.sub_t == SMALL_VTA
    assert sm.u_val == VOTE_RESTART


def test_insim_small_generic():
    packet = struct.pack("<BBBB I", 2, ISP_SMALL, 0, 1, 12345)  # SMALL_SSP
    sm = InSimSmall.parse(packet)
    assert not isinstance(sm, InSimVoteAction)
    assert sm.sub_t == 1
    assert sm.u_val == 12345
