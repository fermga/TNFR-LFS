"""Focused tests for the OutSim UDP client."""

from __future__ import annotations

import asyncio
from collections import deque
import socket
import time
from typing import Deque
from types import SimpleNamespace

import pytest

from tnfr_lfs.ingestion import outsim_udp as outsim_module
from tnfr_lfs.ingestion.outsim_udp import AsyncOutSimUDPClient, OutSimUDPClient


def test_outsim_recv_returns_quickly_when_socket_idle(monkeypatch) -> None:
    client = OutSimUDPClient(timeout=0.05, retries=5)
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


def _outsim_payload(time_ms: int) -> bytes:
    return outsim_module._BASE_STRUCT.pack(time_ms, *([0.0] * 15))


def test_outsim_recv_drains_batch_after_wait(monkeypatch) -> None:
    client = OutSimUDPClient(timeout=0.05, retries=5)

    class FakeSocket:
        def __init__(self) -> None:
            self.queue: Deque[bytes] = deque()

        def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
            if not self.queue:
                raise BlockingIOError()
            payload = self.queue.popleft()
            return payload, ("127.0.0.1", 4123)

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
                _outsim_payload(time_value) for time_value in (100, 120, 140)
            )
        return True

    monkeypatch.setattr(outsim_module, "wait_for_read_ready", fake_wait)

    try:
        packets = [client.recv() for _ in range(3)]
    finally:
        client.close()

    assert [packet.time for packet in packets if packet] == [100, 120, 140]
    assert len(wait_calls) == 1


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
