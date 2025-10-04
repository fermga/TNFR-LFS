import math
import socket
import struct
import threading
from io import StringIO

import pytest

from tnfr_lfs.acquisition import (
    ButtonLayout,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    InSimClient,
    OverlayManager,
    OutGaugePacket,
    OutGaugeUDPClient,
    OutSimClient,
    OutSimPacket,
    OutSimUDPClient,
    TelemetryFormatError,
    TelemetryFusion,
)
from tnfr_lfs.core.epi import EPIBundle


@pytest.fixture
def outsim_payload():
    packer = struct.Struct("<I15fI")
    return packer.pack(
        1000,
        0.1,
        0.2,
        0.3,
        1.0,
        0.1,
        0.05,
        0.4,
        1.2,
        -3.81,
        21.0,
        0.5,
        0.0,
        1.0,
        2.0,
        0.5,
        7,
    )


@pytest.fixture
def outgauge_payload():
    packer = struct.Struct("<I4s16s8s6s6sHBBfffffffIIfff16s16sI")
    return packer.pack(
        1000,
        b"XRT\x00",
        b"Driver\x00" + b"\x00" * 10,
        b"ABC123\x00" + b"\x00",
        b"BL1\x00\x00",
        b"\x00" * 6,
        0,
        3,
        1,
        20.0,
        5200.0,
        0.0,
        90.0,
        0.5,
        3.0,
        100.0,
        1,
        0,
        0.82,
        0.1,
        0.0,
        b"GEAR3\x00" + b"\x00" * 10,
        b"SPD20\x00" + b"\x00" * 10,
        123,
    )


def test_ingest_validates_header():
    client = OutSimClient()
    data = StringIO(
        "timestamp,vertical_load,slip_ratio,lateral_accel,longitudinal_accel,yaw,pitch,roll,brake_pressure,locking,nfr,si,"
        "speed,yaw_rate,slip_angle,steer,throttle,gear,vertical_load_front,vertical_load_rear,mu_eff_front,mu_eff_rear,"
        "suspension_travel_front,suspension_travel_rear,suspension_velocity_front,suspension_velocity_rear\n"
        "0.0,6000,0.05,1.2,0.4,0.1,0.01,0.02,0.5,1,520,0.82,21.0,0.15,0.05,0.2,0.7,3,3200,2800,1.1,1.0,0.52,0.48,0.0,0.0\n"
    )
    records = client.ingest(data)
    assert len(records) == 1


def test_ingest_rejects_invalid_header():
    client = OutSimClient()
    data = StringIO("bad,columns\n1,2\n")
    with pytest.raises(TelemetryFormatError):
        client.ingest(data)


def test_outsim_packet_deserialization(outsim_payload):
    packet = OutSimPacket.from_bytes(outsim_payload)
    assert packet.player_id == 7
    assert packet.vel_x == pytest.approx(21.0)


def test_outgauge_packet_deserialization(outgauge_payload):
    packet = OutGaugePacket.from_bytes(outgauge_payload)
    assert packet.car == "XRT"
    assert packet.throttle == pytest.approx(0.82)


def test_outsim_udp_client_reads_payload(outsim_payload):
    with OutSimUDPClient(timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES, port=0) as client:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # No data yet should return None
            assert client.recv() is None
            sock.sendto(outsim_payload, client.address)
            packet = client.recv()
        finally:
            sock.close()
    assert isinstance(packet, OutSimPacket)


def test_outgauge_udp_client_reads_payload(outgauge_payload):
    with OutGaugeUDPClient(timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES, port=0) as client:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            assert client.recv() is None
            sock.sendto(outgauge_payload, client.address)
            packet = client.recv()
        finally:
            sock.close()
    assert isinstance(packet, OutGaugePacket)


def test_fusion_generates_record_and_bundle(outsim_payload, outgauge_payload):
    outsim = OutSimPacket.from_bytes(outsim_payload)
    outgauge = OutGaugePacket.from_bytes(outgauge_payload)
    fusion = TelemetryFusion()
    record = fusion.fuse(outsim, outgauge)
    assert record.vertical_load == pytest.approx(6000.0)
    assert record.slip_ratio == pytest.approx(0.05)
    assert record.speed == pytest.approx(math.hypot(outsim.vel_x, outsim.vel_y))
    assert record.yaw_rate == pytest.approx(outsim.ang_vel_z, rel=1e-6)
    assert record.vertical_load_front + record.vertical_load_rear == pytest.approx(record.vertical_load)
    assert 0.25 <= record.suspension_travel_front <= 0.75
    assert record.suspension_velocity_front == pytest.approx(0.0)
    assert record.mu_eff_front >= 0.0
    bundle = fusion.fuse_to_bundle(outsim, outgauge)
    assert isinstance(bundle, EPIBundle)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        data.extend(chunk)
    return bytes(data)


def test_insim_client_handshake_and_keepalive():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    captured: dict[str, bytes] = {}

    def _server() -> None:
        conn, _ = server.accept()
        with conn:
            header = _recv_exact(conn, 1)
            payload = _recv_exact(conn, header[0] - 1)
            captured["handshake"] = header + payload
            version_packet = InSimClient.VER_STRUCT.pack(
                InSimClient.VER_STRUCT.size,
                InSimClient.ISP_VER,
                7,
                0,
                InSimClient.INSIM_VERSION,
            )
            conn.sendall(version_packet)
            captured["keepalive"] = _recv_exact(conn, InSimClient.TINY_STRUCT.size)

    thread = threading.Thread(target=_server, daemon=True)
    thread.start()

    client = InSimClient(host="127.0.0.1", port=port, keepalive_interval=1.0, request_id=7)
    try:
        client.connect()
        client.send_keepalive()
    finally:
        client.close()

    thread.join(timeout=1.0)
    server.close()

    handshake = InSimClient.ISI_STRUCT.unpack(captured["handshake"])
    assert handshake[0] == InSimClient.ISI_STRUCT.size
    assert handshake[1] == InSimClient.ISP_ISI
    assert handshake[2] == 7
    assert handshake[5] == InSimClient.INSIM_VERSION
    assert handshake[8] == InSimClient.ISF_LOCAL
    assert handshake[9] == pytest.approx(1000)
    keepalive = InSimClient.TINY_STRUCT.unpack(captured["keepalive"])
    assert keepalive[1] == InSimClient.ISP_TINY
    assert keepalive[2] == 7
    assert keepalive[3] == InSimClient.TINY_ALIVE


def test_insim_client_button_serialisation():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    captured: dict[str, bytes] = {}

    def _server() -> None:
        conn, _ = server.accept()
        with conn:
            header = _recv_exact(conn, 1)
            payload = _recv_exact(conn, header[0] - 1)
            captured["handshake"] = header + payload
            version_packet = InSimClient.VER_STRUCT.pack(
                InSimClient.VER_STRUCT.size,
                InSimClient.ISP_VER,
                3,
                0,
                InSimClient.INSIM_VERSION,
            )
            conn.sendall(version_packet)
            captured["subscribe"] = _recv_exact(conn, InSimClient.TINY_STRUCT.size)
            btn_header = _recv_exact(conn, 1)
            btn_payload = _recv_exact(conn, btn_header[0] - 1)
            captured["button"] = btn_header + btn_payload
            clr_header = _recv_exact(conn, 1)
            clr_payload = _recv_exact(conn, clr_header[0] - 1)
            captured["clear"] = clr_header + clr_payload

    thread = threading.Thread(target=_server, daemon=True)
    thread.start()

    layout = ButtonLayout(left=15, top=12, width=90, height=25, click_id=5)
    with InSimClient(host="127.0.0.1", port=port, request_id=3) as client:
        manager = OverlayManager(client, layout=layout)
        manager.connect()
        manager.show(["Linea 1", "Linea 2"])
        client.clear_button(layout)

    thread.join(timeout=1.0)
    server.close()

    subscribe = InSimClient.TINY_STRUCT.unpack(captured["subscribe"])
    assert subscribe[3] == InSimClient.TINY_SUBT_BTC

    button_header = InSimClient.BTN_HEADER_STRUCT.unpack(
        captured["button"][: InSimClient.BTN_HEADER_STRUCT.size]
    )
    assert button_header[0] == len(captured["button"])
    assert button_header[2] == 3
    assert button_header[4] == 5
    assert button_header[8] == 15
    assert button_header[10] == 90
    text_payload = captured["button"][InSimClient.BTN_HEADER_STRUCT.size : -1]
    assert text_payload.decode("utf8") == "TNFR × LFS\nLinea 1\nLinea 2"
    assert captured["button"][-1] == 0

    clear_header = InSimClient.BTN_HEADER_STRUCT.unpack(
        captured["clear"][: InSimClient.BTN_HEADER_STRUCT.size]
    )
    assert clear_header[6] == InSimClient.BTN_STYLE_CLEAR
    assert captured["clear"][-1] == 0


def test_insim_client_poll_button_click():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def _server() -> None:
        conn, _ = server.accept()
        with conn:
            header = _recv_exact(conn, 1)
            _recv_exact(conn, header[0] - 1)
            version_packet = InSimClient.VER_STRUCT.pack(
                InSimClient.VER_STRUCT.size,
                InSimClient.ISP_VER,
                9,
                0,
                InSimClient.INSIM_VERSION,
            )
            conn.sendall(version_packet)
            _recv_exact(conn, InSimClient.TINY_STRUCT.size)
            btc_packet = InSimClient.BTC_STRUCT.pack(
                InSimClient.BTC_STRUCT.size,
                InSimClient.ISP_BTC,
                4,
                2,
                7,
                1,
                0,
                0,
                0,
                0x0200,
            )
            conn.sendall(btc_packet)

    thread = threading.Thread(target=_server, daemon=True)
    thread.start()

    client = InSimClient(host="127.0.0.1", port=port, request_id=4)
    try:
        client.connect()
        client.subscribe_controls()
        event = client.poll_button(timeout=0.2)
    finally:
        client.close()

    thread.join(timeout=1.0)
    server.close()

    assert event is not None
    assert event.ucid == 2
    assert event.click_id == 7
    assert event.inst == 1
    assert event.type_in == 0
    assert event.typed_char is None
    assert event.flags == 0x0200
