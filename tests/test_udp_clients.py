"""Shared behavioral tests for UDP telemetry clients."""

from __future__ import annotations

import asyncio
import socket
import struct
import time
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from tnfr_lfs.telemetry.live import OutGaugePacket

from tests.helpers import (
    assert_async_client_close_wakes,
    assert_async_udp_reordering,
    assert_udp_batch_drained,
    assert_udp_isolated_flush,
    assert_udp_pending_flush,
    make_select_stub,
    pad_outgauge_field,
    patch_udp_client_socket,
    raise_gaierror,
)


def _build_payload(payload_factory: Callable[..., bytes], spec: Any) -> bytes:
    """Construct a payload from ``spec`` using ``payload_factory``."""

    if isinstance(spec, dict):
        return payload_factory(**spec)
    if isinstance(spec, Sequence) and not isinstance(spec, (bytes, bytearray, str)):
        return payload_factory(*spec)
    return payload_factory(spec)


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


def test_udp_recv_returns_quickly_when_socket_idle(
    monkeypatch: pytest.MonkeyPatch, udp_client_spec: dict[str, Any]
) -> None:
    client = udp_client_spec["sync_constructor"](timeout=0.05, retries=5)
    fake_select, call_args = make_select_stub()
    patch_udp_client_socket(
        monkeypatch, client, address=udp_client_spec["default_address"]
    )
    monkeypatch.setattr("tnfr_lfs.telemetry._socket_poll.select.select", fake_select)

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


def test_udp_host_resolution_failure_disables_filtering(
    monkeypatch: pytest.MonkeyPatch, udp_client_spec: dict[str, Any]
) -> None:
    module = udp_client_spec["module"]
    monkeypatch.setattr(module.socket, "getaddrinfo", raise_gaierror)

    client = udp_client_spec["sync_constructor"](
        host="unresolvable", port=0, timeout=0.05, retries=1
    )
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packet = None
    try:
        payload = _build_payload(
            udp_client_spec["payload_factory"], udp_client_spec["host_payload_spec"]
        )
        sender.sendto(payload, client.address)
        packet = client.recv()
        assert packet is not None
        expected_value = udp_client_spec["host_expected_value"]
        value = udp_client_spec["value_extractor"](packet)
        assert value == expected_value
        assert client.ignored_hosts == 0
    finally:
        if packet is not None:
            packet.release()
        sender.close()
        client.close()


def test_udp_recv_drains_batch_after_wait(
    monkeypatch: pytest.MonkeyPatch, udp_client_spec: dict[str, Any]
) -> None:
    assert_udp_batch_drained(
        monkeypatch,
        client_factory=lambda: udp_client_spec["sync_constructor"](
            timeout=0.05, retries=5
        ),
        module=udp_client_spec["module"],
        payload_factory=udp_client_spec["payload_factory"],
        batch_specs=udp_client_spec["batch_specs"],
        expected_values=udp_client_spec["batch_expected_values"],
        value_extractor=udp_client_spec["value_extractor"],
        address=udp_client_spec["default_address"],
        release_packets=udp_client_spec["release_packets"],
    )


def test_udp_pending_packet_flushes_when_successor_arrives(
    monkeypatch: pytest.MonkeyPatch, udp_client_spec: dict[str, Any]
) -> None:
    expected_first, expected_second = udp_client_spec["pending_expected_values"]
    assert_udp_pending_flush(
        monkeypatch,
        client_factory=lambda: udp_client_spec["sync_constructor"](
            port=0, timeout=0.2, retries=2, reorder_grace=0.5
        ),
        module=udp_client_spec["module"],
        payload_factory=udp_client_spec["payload_factory"],
        first_spec=udp_client_spec["pending_first"],
        successor_spec=udp_client_spec["pending_successor"],
        appended_spec=udp_client_spec["pending_appended"],
        expected_first=expected_first,
        expected_second=expected_second,
        value_extractor=udp_client_spec["value_extractor"],
        max_elapsed=0.2,
        address=udp_client_spec["default_address"],
        release_packets=udp_client_spec["release_packets"],
    )


def test_udp_isolated_packet_flushes_under_10ms_by_default(
    monkeypatch: pytest.MonkeyPatch, udp_client_spec: dict[str, Any]
) -> None:
    expected_first, expected_second = udp_client_spec["isolated_expected_values"]
    assert_udp_isolated_flush(
        monkeypatch,
        client_factory=lambda: udp_client_spec["sync_constructor"](
            port=0, timeout=0.2, retries=1
        ),
        module=udp_client_spec["module"],
        payload_factory=udp_client_spec["payload_factory"],
        first_spec=udp_client_spec["isolated_first"],
        second_spec=udp_client_spec["isolated_second"],
        expected_first=expected_first,
        expected_second=expected_second,
        value_extractor=udp_client_spec["value_extractor"],
        address=udp_client_spec["default_address"],
        release_packets=udp_client_spec["release_packets"],
    )


def test_async_udp_client_recovers_out_of_order_packets(
    udp_client_spec: dict[str, Any]
) -> None:
    async def runner() -> None:
        await assert_async_udp_reordering(
            client_factory=lambda: udp_client_spec["async_constructor"](
                port=0, reorder_grace=0.1, timeout=0.5
            ),
            payload_factory=udp_client_spec["payload_factory"],
            send_specs=udp_client_spec["async_send_specs"],
            expected_values=udp_client_spec["async_expected_values"],
            value_extractor=udp_client_spec["value_extractor"],
            release_packets=udp_client_spec["release_packets"],
        )

    asyncio.run(runner())


def test_async_udp_client_wakes_waiters_on_close(
    udp_client_spec: dict[str, Any]
) -> None:
    async def runner() -> None:
        await assert_async_client_close_wakes(
            client_factory=lambda: udp_client_spec["async_constructor"](
                port=0, timeout=0.2
            )
        )

    asyncio.run(runner())

