"""Focused tests for the OutGauge UDP decoder."""

from __future__ import annotations

import asyncio
from collections import deque
import socket
import struct
import time
from typing import Deque
from types import SimpleNamespace

import pytest

from tnfr_lfs.ingestion import outgauge_udp as outgauge_module
from tnfr_lfs.ingestion.live import OutGaugePacket
from tnfr_lfs.ingestion.outgauge_udp import AsyncOutGaugeUDPClient, OutGaugeUDPClient


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

    packet = OutGaugePacket.from_bytes(base_payload + extras, freeze=True)

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


def _outgauge_payload(packet_id: int, time_value: int) -> bytes:
    return outgauge_module._PACK_STRUCT.pack(
        time_value,
        b"XFG\x00",
        b"Driver\x00" + b"\x00" * 9,
        b"\x00" * 8,
        b"BL1\x00\x00",
        b"LYT\x00\x00",
        0,
        3,
        0,
        50.0,
        4000.0,
        0.0,
        80.0,
        30.0,
        0.0,
        90.0,
        0,
        0,
        0.5,
        0.1,
        0.0,
        b"\x00" * 16,
        b"\x00" * 16,
        packet_id,
    )


def test_outgauge_host_resolution_failure_disables_filtering(monkeypatch) -> None:
    def raise_gaierror(*_args: object, **_kwargs: object) -> list[object]:
        raise socket.gaierror()

    monkeypatch.setattr(outgauge_module.socket, "getaddrinfo", raise_gaierror)

    client = OutGaugeUDPClient(host="unresolvable", port=0, timeout=0.05, retries=1)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packet = None
    try:
        sender.sendto(_outgauge_payload(7, 70), client.address)
        packet = client.recv()
        assert packet is not None
        assert packet.packet_id == 7
        assert client.ignored_hosts == 0
    finally:
        if packet is not None:
            packet.release()
        sender.close()
        client.close()


def test_outgauge_recv_drains_batch_after_wait(monkeypatch) -> None:
    client = OutGaugeUDPClient(timeout=0.05, retries=5)

    class FakeSocket:
        def __init__(self) -> None:
            self.queue: Deque[bytes] = deque()

        def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
            if not self.queue:
                raise BlockingIOError()
            payload = self.queue.popleft()
            return payload, ("127.0.0.1", 3000)

        def close(self) -> None:
            self.queue.clear()

    fake_socket = FakeSocket()
    original_socket = client._socket
    monkeypatch.setattr(client, "_socket", fake_socket)
    original_socket.close()

    wait_calls: list[float | None] = []

    def fake_wait(sock: object, *, timeout: float, deadline: float | None) -> bool:
        wait_calls.append(timeout)
        if not fake_socket.queue:
            fake_socket.queue.extend(
                _outgauge_payload(packet_id, time_value)
                for packet_id, time_value in ((5, 50), (6, 60), (7, 70))
            )
        return True

    monkeypatch.setattr(outgauge_module, "wait_for_read_ready", fake_wait)

    try:
        packets = [client.recv() for _ in range(3)]
    finally:
        client.close()

    assert [packet.packet_id for packet in packets if packet] == [5, 6, 7]
    for packet in packets:
        if packet is not None:
            packet.release()
    assert wait_calls
    assert wait_calls[0] == pytest.approx(client._timeout, rel=0.1)


def test_outgauge_pending_packet_flushes_when_successor_arrives(monkeypatch) -> None:
    client = OutGaugeUDPClient(timeout=0.2, retries=2, reorder_grace=0.5)

    class FakeSocket:
        def __init__(self) -> None:
            self.queue: Deque[bytes] = deque()

        def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
            if not self.queue:
                raise BlockingIOError()
            payload = self.queue.popleft()
            return payload, ("127.0.0.1", 3000)

        def close(self) -> None:
            self.queue.clear()

    fake_socket = FakeSocket()
    original_socket = client._socket
    monkeypatch.setattr(client, "_socket", fake_socket)
    original_socket.close()

    fake_socket.queue.append(_outgauge_payload(5, 50))
    packet = client.recv()
    assert packet is not None
    assert packet.packet_id == 5
    packet.release()

    fake_socket.queue.append(_outgauge_payload(6, 60))
    wait_calls: list[float] = []
    appended = False

    def fake_wait(sock: object, *, timeout: float, deadline: float | None) -> bool:
        nonlocal appended
        wait_calls.append(timeout)
        if not appended:
            fake_socket.queue.append(_outgauge_payload(7, 70))
            appended = True
        return True

    monkeypatch.setattr(outgauge_module, "wait_for_read_ready", fake_wait)

    start = time.perf_counter()
    next_packet = client.recv()
    elapsed = time.perf_counter() - start

    try:
        assert next_packet is not None
        assert next_packet.packet_id == 6
        assert elapsed < 0.2
        assert wait_calls
    finally:
        if next_packet is not None:
            next_packet.release()
        client.close()


def test_outgauge_isolated_packet_flushes_under_10ms_by_default(monkeypatch) -> None:
    client = OutGaugeUDPClient(timeout=0.2, retries=1)

    class FakeSocket:
        def __init__(self) -> None:
            self.queue: Deque[bytes] = deque()

        def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
            if not self.queue:
                raise BlockingIOError()
            payload = self.queue.popleft()
            return payload, ("127.0.0.1", 3000)

        def close(self) -> None:
            self.queue.clear()

    fake_socket = FakeSocket()
    original_socket = client._socket
    monkeypatch.setattr(client, "_socket", fake_socket)
    original_socket.close()

    wait_calls: list[float] = []

    def fake_wait(sock: object, *, timeout: float, deadline: float | None) -> bool:
        wait_calls.append(timeout)
        return False

    monkeypatch.setattr(outgauge_module, "wait_for_read_ready", fake_wait)

    fake_socket.queue.append(_outgauge_payload(7, 70))
    first = client.recv()
    assert first is not None
    assert first.packet_id == 7
    first.release()

    fake_socket.queue.append(_outgauge_payload(8, 90))
    start = time.perf_counter()
    second = client.recv()
    elapsed_ms = (time.perf_counter() - start) * 1_000

    try:
        assert second is not None
        assert second.packet_id == 8
        assert elapsed_ms < 10.0
        assert wait_calls, "wait_for_read_ready should be consulted for pending packet"
        assert wait_calls[0] <= 0.012
    finally:
        if second is not None:
            second.release()
        client.close()


def test_async_outgauge_client_recovers_out_of_order_packets() -> None:
    async def runner() -> None:
        client = await AsyncOutGaugeUDPClient.create(port=0, reorder_grace=0.1, timeout=0.5)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.bind(("127.0.0.1", 0))
        try:
            _, port = client.address
            target = ("127.0.0.1", port)
            await asyncio.sleep(0)
            sender.sendto(_outgauge_payload(5, 50), target)
            sender.sendto(_outgauge_payload(7, 70), target)
            sender.sendto(_outgauge_payload(6, 60), target)
            results = []
            for _ in range(3):
                packet = await client.recv()
                if packet is not None:
                    results.append(packet)
            assert [packet.packet_id for packet in results] == [5, 6, 7]
            stats = client.statistics
            assert stats["delivered"] == 3
            assert stats["reordered"] >= 1
            for packet in results:
                packet.release()
        finally:
            sender.close()
            await client.close()

    asyncio.run(runner())


def test_async_outgauge_client_wakes_waiters_on_close() -> None:
    async def runner() -> None:
        client = await AsyncOutGaugeUDPClient.create(port=0, timeout=0.2)
        try:
            recv_task = asyncio.create_task(client.recv())
            await asyncio.sleep(0)
            await client.close()
            with pytest.raises(RuntimeError):
                await recv_task
        finally:
            await client.close()

    asyncio.run(runner())
