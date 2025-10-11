"""Focused tests for the OutSim UDP client."""

from __future__ import annotations

import asyncio
import socket
import time

import pytest

from tnfr_lfs.ingestion import outsim_udp as outsim_module
from tnfr_lfs.ingestion.outsim_udp import AsyncOutSimUDPClient, OutSimUDPClient
from tests.helpers import (
    assert_async_client_close_wakes,
    assert_async_udp_reordering,
    assert_udp_batch_drained,
    assert_udp_isolated_flush,
    assert_udp_pending_flush,
    build_outsim_payload,
    make_select_stub,
    patch_udp_client_socket,
    raise_gaierror,
)


def test_outsim_recv_returns_quickly_when_socket_idle(monkeypatch) -> None:
    client = OutSimUDPClient(timeout=0.05, retries=5)
    fake_select, call_args = make_select_stub()
    patch_udp_client_socket(monkeypatch, client)
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


def test_outsim_host_resolution_failure_disables_filtering(monkeypatch) -> None:
    monkeypatch.setattr(outsim_module.socket, "getaddrinfo", raise_gaierror)

    client = OutSimUDPClient(host="unresolvable", port=0, timeout=0.05, retries=1)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packet = None
    try:
        sender.sendto(build_outsim_payload(200), client.address)
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
    assert_udp_batch_drained(
        monkeypatch,
        client_factory=lambda: OutSimUDPClient(timeout=0.05, retries=5),
        module=outsim_module,
        payload_factory=build_outsim_payload,
        batch_specs=[100, 120, 140],
        expected_values=[100, 120, 140],
        value_extractor=lambda packet: packet.time,
        address=("127.0.0.1", 4123),
    )


def test_outsim_pending_packet_flushes_when_successor_arrives(monkeypatch) -> None:
    assert_udp_pending_flush(
        monkeypatch,
        client_factory=lambda: OutSimUDPClient(timeout=0.2, retries=2, reorder_grace=0.5),
        module=outsim_module,
        payload_factory=build_outsim_payload,
        first_spec=100,
        successor_spec=120,
        appended_spec=140,
        expected_first=100,
        expected_second=120,
        value_extractor=lambda packet: packet.time,
        max_elapsed=0.2,
        address=("127.0.0.1", 4123),
    )


def test_outsim_isolated_packet_flushes_under_10ms_by_default(monkeypatch) -> None:
    assert_udp_isolated_flush(
        monkeypatch,
        client_factory=lambda: OutSimUDPClient(timeout=0.2, retries=1),
        module=outsim_module,
        payload_factory=build_outsim_payload,
        first_spec=100,
        second_spec=120,
        expected_first=100,
        expected_second=120,
        value_extractor=lambda packet: packet.time,
        address=("127.0.0.1", 4123),
    )


def test_async_outsim_client_handles_concurrent_receivers() -> None:
    async def runner() -> None:
        await assert_async_udp_reordering(
            client_factory=lambda: AsyncOutSimUDPClient.create(
                port=0, reorder_grace=0.1, timeout=0.5
            ),
            payload_factory=build_outsim_payload,
            send_specs=[100, 140, 120],
            expected_values=[100, 120, 140],
            value_extractor=lambda packet: packet.time,
        )

    asyncio.run(runner())


def test_async_outsim_client_wakes_waiters_on_close() -> None:
    async def runner() -> None:
        await assert_async_client_close_wakes(
            client_factory=lambda: AsyncOutSimUDPClient.create(port=0, timeout=0.2)
        )

    asyncio.run(runner())
