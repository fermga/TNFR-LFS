import socket
import struct
from io import StringIO

import pytest

from tnfr_lfs.acquisition import (
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
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
        "timestamp,vertical_load,slip_ratio,lateral_accel,longitudinal_accel,yaw,pitch,roll,brake_pressure,locking,nfr,si\n"
        "0.0,6000,0.05,1.2,0.4,0.1,0.01,0.02,0.5,1,520,0.82\n"
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
    bundle = fusion.fuse_to_bundle(outsim, outgauge)
    assert isinstance(bundle, EPIBundle)
