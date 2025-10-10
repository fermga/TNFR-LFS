"""Focused tests for the OutSim UDP client."""

from __future__ import annotations

import asyncio
import socket
import time

import pytest

from tnfr_lfs.ingestion import outsim_udp as outsim_module
from tnfr_lfs.ingestion.outsim_udp import AsyncOutSimUDPClient, OutSimUDPClient
from tests.helpers import QueueUDPSocket, make_select_stub, make_wait_stub


def test_outsim_recv_returns_quickly_when_socket_idle(monkeypatch) -> None:
    client = OutSimUDPClient(timeout=0.05, retries=5)
    fake_select, call_args = make_select_stub()
    fake_socket = QueueUDPSocket()
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


def _outsim_payload(time_ms: int) -> bytes:
    return outsim_module._BASE_STRUCT.pack(time_ms, *([0.0] * 15))


def test_outsim_host_resolution_failure_disables_filtering(monkeypatch) -> None:
    def raise_gaierror(*_args: object, **_kwargs: object) -> list[object]:
        raise socket.gaierror()

    monkeypatch.setattr(outsim_module.socket, "getaddrinfo", raise_gaierror)

    client = OutSimUDPClient(host="unresolvable", port=0, timeout=0.05, retries=1)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packet = None
    try:
        sender.sendto(_outsim_payload(200), client.address)
        packet = client.recv()
        assert packet is not None
        assert packet.time == 200
        assert client.ignored_hosts == 0
    finally:
        if packet is not None:
            packet.release()
        sender.close()
        client.close()


def test_outsim_recv_drains_batch_after_wait(monkeypatch) -> None:
    client = OutSimUDPClient(timeout=0.05, retries=5)

    fake_socket = QueueUDPSocket(address=("127.0.0.1", 4123))
    original_socket = client._socket
    monkeypatch.setattr(client, "_socket", fake_socket)
    original_socket.close()

    def on_wait(_sock: object, _timeout: float, _deadline: float | None) -> None:
        if not fake_socket.queue:
            fake_socket.extend(_outsim_payload(time_value) for time_value in (100, 120, 140))

    fake_wait, wait_calls = make_wait_stub(hook=on_wait)
    monkeypatch.setattr(outsim_module, "wait_for_read_ready", fake_wait)

    try:
        packets = [client.recv() for _ in range(3)]
    finally:
        client.close()

    assert [packet.time for packet in packets if packet] == [100, 120, 140]
    assert wait_calls
    assert wait_calls[0] == pytest.approx(client._timeout, rel=0.1)


def test_outsim_pending_packet_flushes_when_successor_arrives(monkeypatch) -> None:
    client = OutSimUDPClient(timeout=0.2, retries=2, reorder_grace=0.5)

    fake_socket = QueueUDPSocket(address=("127.0.0.1", 4123))
    original_socket = client._socket
    monkeypatch.setattr(client, "_socket", fake_socket)
    original_socket.close()

    fake_socket.queue.append(_outsim_payload(100))
    packet = client.recv()
    assert packet is not None
    assert packet.time == 100
    packet.release()

    fake_socket.queue.append(_outsim_payload(120))
    wait_calls: list[float] = []
    appended = False

    def on_wait(_sock: object, _timeout: float, _deadline: float | None) -> None:
        nonlocal appended
        if not appended:
            fake_socket.queue.append(_outsim_payload(140))
            appended = True

    fake_wait, wait_calls = make_wait_stub(hook=on_wait)
    monkeypatch.setattr(outsim_module, "wait_for_read_ready", fake_wait)

    start = time.perf_counter()
    next_packet = client.recv()
    elapsed = time.perf_counter() - start

    try:
        assert next_packet is not None
        assert next_packet.time == 120
        assert elapsed < 0.2
        assert wait_calls, "wait_for_read_ready should be invoked during pending flush"
    finally:
        if next_packet is not None:
            next_packet.release()
        client.close()


def test_outsim_isolated_packet_flushes_under_10ms_by_default(monkeypatch) -> None:
    client = OutSimUDPClient(timeout=0.2, retries=1)

    fake_socket = QueueUDPSocket(address=("127.0.0.1", 4123))
    original_socket = client._socket
    monkeypatch.setattr(client, "_socket", fake_socket)
    original_socket.close()

    wait_calls: list[float] = []

    fake_wait, wait_calls = make_wait_stub(return_value=False)
    monkeypatch.setattr(outsim_module, "wait_for_read_ready", fake_wait)

    fake_socket.queue.append(_outsim_payload(100))
    first = client.recv()
    assert first is not None
    assert first.time == 100
    first.release()

    fake_socket.queue.append(_outsim_payload(120))
    start = time.perf_counter()
    second = client.recv()
    elapsed_ms = (time.perf_counter() - start) * 1_000

    try:
        assert second is not None
        assert second.time == 120
        assert elapsed_ms < 10.0
        assert wait_calls, "wait_for_read_ready should be consulted for pending packet"
        assert wait_calls[0] <= 0.012
    finally:
        if second is not None:
            second.release()
        client.close()


def test_async_outsim_client_handles_concurrent_receivers() -> None:
    async def runner() -> None:
        client = await AsyncOutSimUDPClient.create(port=0, reorder_grace=0.1, timeout=0.5)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.bind(("127.0.0.1", 0))
        try:
            _, port = client.address
            target = ("127.0.0.1", port)
            await asyncio.sleep(0)
            sender.sendto(_outsim_payload(100), target)
            sender.sendto(_outsim_payload(140), target)
            sender.sendto(_outsim_payload(120), target)
            results = []
            for _ in range(3):
                packet = await client.recv()
                if packet is not None:
                    results.append(packet)
            assert [packet.time for packet in results] == [100, 120, 140]
            stats = client.statistics
            assert stats["delivered"] == 3
            assert stats["reordered"] >= 1
        finally:
            sender.close()
            await client.close()

    asyncio.run(runner())


def test_async_outsim_client_wakes_waiters_on_close() -> None:
    async def runner() -> None:
        client = await AsyncOutSimUDPClient.create(port=0, timeout=0.2)
        try:
            recv_task = asyncio.create_task(client.recv())
            await asyncio.sleep(0)
            await client.close()
            with pytest.raises(RuntimeError):
                await recv_task
        finally:
            await client.close()

    asyncio.run(runner())
