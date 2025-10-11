"""Focused tests for the OutGauge UDP decoder."""

from __future__ import annotations

import asyncio
import socket
import struct
import time

import pytest

from tnfr_lfs.ingestion import outgauge_udp as outgauge_module
from tnfr_lfs.ingestion.live import OutGaugePacket
from tnfr_lfs.ingestion.outgauge_udp import AsyncOutGaugeUDPClient, OutGaugeUDPClient
from tests.helpers import (
    assert_async_client_close_wakes,
    assert_async_udp_reordering,
    assert_udp_batch_drained,
    assert_udp_isolated_flush,
    assert_udp_pending_flush,
    build_outgauge_payload,
    make_select_stub,
    pad_outgauge_field,
    patch_udp_client_socket,
    raise_gaierror,
)


def test_outgauge_packet_parses_extended_datagram_identifiers() -> None:
    base_payload = struct.pack(
        "<I4s16s8s6s6sHBBfffffffIIfff16s16sI",
        1234,
        pad_outgauge_field("CAR", 4),
        pad_outgauge_field("Driver", 16),
        pad_outgauge_field("PLATE", 8),
        pad_outgauge_field("BL1", 6),
        pad_outgauge_field("", 6),
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
        pad_outgauge_field("HUD1", 16),
        pad_outgauge_field("HUD2", 16),
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


def test_outgauge_host_resolution_failure_disables_filtering(monkeypatch) -> None:
    monkeypatch.setattr(outgauge_module.socket, "getaddrinfo", raise_gaierror)

    client = OutGaugeUDPClient(host="unresolvable", port=0, timeout=0.05, retries=1)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packet = None
    try:
        sender.sendto(
            build_outgauge_payload(7, 70, layout="LYT"),
            client.address,
        )
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
    assert_udp_batch_drained(
        monkeypatch,
        client_factory=lambda: OutGaugeUDPClient(timeout=0.05, retries=5),
        module=outgauge_module,
        payload_factory=build_outgauge_payload,
        batch_specs=[
            {"packet_id": 5, "time_value": 50, "layout": "LYT"},
            {"packet_id": 6, "time_value": 60, "layout": "LYT"},
            {"packet_id": 7, "time_value": 70, "layout": "LYT"},
        ],
        expected_values=[5, 6, 7],
        value_extractor=lambda packet: packet.packet_id,
        address=("127.0.0.1", 3000),
        release_packets=True,
    )


def test_outgauge_pending_packet_flushes_when_successor_arrives(monkeypatch) -> None:
    assert_udp_pending_flush(
        monkeypatch,
        client_factory=lambda: OutGaugeUDPClient(
            port=0, timeout=0.2, retries=2, reorder_grace=0.5
        ),
        module=outgauge_module,
        payload_factory=build_outgauge_payload,
        first_spec={"packet_id": 5, "time_value": 50, "layout": "LYT"},
        successor_spec={"packet_id": 6, "time_value": 60, "layout": "LYT"},
        appended_spec={"packet_id": 7, "time_value": 70, "layout": "LYT"},
        expected_first=5,
        expected_second=6,
        value_extractor=lambda packet: packet.packet_id,
        max_elapsed=0.2,
        address=("127.0.0.1", 3000),
        release_packets=True,
    )


def test_outgauge_isolated_packet_flushes_under_10ms_by_default(monkeypatch) -> None:
    assert_udp_isolated_flush(
        monkeypatch,
        client_factory=lambda: OutGaugeUDPClient(port=0, timeout=0.2, retries=1),
        module=outgauge_module,
        payload_factory=build_outgauge_payload,
        first_spec={"packet_id": 7, "time_value": 70, "layout": "LYT"},
        second_spec={"packet_id": 8, "time_value": 90, "layout": "LYT"},
        expected_first=7,
        expected_second=8,
        value_extractor=lambda packet: packet.packet_id,
        address=("127.0.0.1", 3000),
        release_packets=True,
    )


def test_async_outgauge_client_recovers_out_of_order_packets() -> None:
    async def runner() -> None:
        await assert_async_udp_reordering(
            client_factory=lambda: AsyncOutGaugeUDPClient.create(
                port=0, reorder_grace=0.1, timeout=0.5
            ),
            payload_factory=build_outgauge_payload,
            send_specs=[
                {"packet_id": 5, "time_value": 50, "layout": "LYT"},
                {"packet_id": 7, "time_value": 70, "layout": "LYT"},
                {"packet_id": 6, "time_value": 60, "layout": "LYT"},
            ],
            expected_values=[5, 6, 7],
            value_extractor=lambda packet: packet.packet_id,
            release_packets=True,
        )

    asyncio.run(runner())


def test_async_outgauge_client_wakes_waiters_on_close() -> None:
    async def runner() -> None:
        await assert_async_client_close_wakes(
            client_factory=lambda: AsyncOutGaugeUDPClient.create(port=0, timeout=0.2)
        )

    asyncio.run(runner())
