"""Tests for the OutGauge UDP decoder."""

from __future__ import annotations

import struct

import pytest

from tnfr_lfs.acquisition.outgauge_udp import OutGaugePacket


def _pad(value: str, size: int) -> bytes:
    return value.encode("latin-1").ljust(size, b"\x00")


def test_outgauge_packet_parses_extended_datagram() -> None:
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

    expected_average = tuple(
        (inner[idx] + middle[idx] + outer[idx]) / 3.0 for idx in range(4)
    )

    assert packet.tyre_temps == pytest.approx(expected_average)
    assert packet.brake_temps == pytest.approx(brakes)
