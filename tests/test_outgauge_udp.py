"""Focused tests for the OutGauge UDP decoder."""

from __future__ import annotations

import struct
import time
from types import SimpleNamespace

import pytest

from tnfr_lfs.ingestion.live import OutGaugePacket
from tnfr_lfs.ingestion.outgauge_udp import OutGaugeUDPClient


def _pad(value: str, size: int) -> bytes:
    return value.encode("latin-1").ljust(size, b"\x00")


def test_outgauge_packet_parses_extended_datagram_identifiers() -> None:
    base_payload = struct.pack(
        "<I4s16s8s6s6sHBBfffffffIIfff16s16sI",
        1234,
        _pad("CAR", 4),
        _pad("Driver", 16),
        _pad("PLATE", 8),
        _pad("BL1", 6),
        _pad("", 6),
        0,
        3,
        0,
        80.5,
        7150.0,
        0.4,
        90.0,
        65.0,
        4.0,
        95.0,
        0,
        0,
        0.8,
        0.2,
        0.05,
        _pad("HUD1", 16),
        _pad("HUD2", 16),
        42,
    )

    inner = (95.0, 94.0, 93.0, 92.0)
    middle = (90.0, 89.0, 88.0, 87.0)
    outer = (85.0, 84.0, 83.0, 82.0)
    pressures = (1.6, 1.5, 1.4, 1.3)
    brakes = (420.0, 410.0, 400.0, 390.0)

    extras = struct.pack("<20f", *(inner + middle + outer + pressures + brakes))

    packet = OutGaugePacket.from_bytes(base_payload + extras)

    # The OutGauge decoder is responsible for trimming padded strings and
    # keeping identifier fields intact even when extended tyre data is present.
    assert packet.car == "CAR"
    assert packet.player_name == "Driver"
    assert packet.track == "BL1"
    assert packet.layout == ""
    assert packet.display1 == "HUD1"
    assert packet.display2 == "HUD2"
    assert packet.packet_id == 42


def test_outgauge_recv_returns_quickly_when_socket_idle(monkeypatch) -> None:
    client = OutGaugeUDPClient(timeout=0.05, retries=5)
    call_args: list[float | None] = []

    def fake_select(read: list[object], write: list[object], err: list[object], timeout: float | None = None):
        call_args.append(timeout)
        return ([], [], [])

    fake_socket = SimpleNamespace(
        recvfrom=lambda _: (_ for _ in ()).throw(BlockingIOError()),
        close=lambda: None,
    )
    original_socket = client._socket
    monkeypatch.setattr(client, "_socket", fake_socket)
    original_socket.close()
    monkeypatch.setattr("tnfr_lfs.ingestion._socket_poll.select.select", fake_select)

    start = time.perf_counter()
    try:
        packet = client.recv()
    finally:
        client.close()
    elapsed_ms = (time.perf_counter() - start) * 1_000

    assert packet is None
    assert elapsed_ms < 10.0
    assert call_args, "select.select should be invoked"
    assert call_args[0] == pytest.approx(client._timeout, rel=0.1)
