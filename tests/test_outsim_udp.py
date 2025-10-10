"""Focused tests for the OutSim UDP client."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from tnfr_lfs.ingestion.outsim_udp import OutSimUDPClient


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
