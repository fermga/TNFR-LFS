"""Benchmark pooled packet decoding versus frozen snapshots."""

from __future__ import annotations

import argparse
import statistics
import struct
import time
from typing import Callable

from tnfr_lfs.ingestion.outgauge_udp import OutGaugePacket
from tnfr_lfs.ingestion.outsim_udp import OutSimPacket


def _format(label: str, durations: list[float], iterations: int) -> str:
    mean = statistics.fmean(durations)
    throughput = iterations / mean if mean else float("nan")
    deviation = statistics.pstdev(durations) if len(durations) > 1 else 0.0
    return (
        f"{label}: {mean * 1_000:.3f} ms ± {deviation * 1_000:.3f} ms "
        f"({throughput:,.0f} packets/s)"
    )


def _measure(func: Callable[[bool], None], *, iterations: int, repeats: int) -> tuple[list[float], list[float]]:
    pooled: list[float] = []
    frozen: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            func(False)
        pooled.append(time.perf_counter() - start)

        start = time.perf_counter()
        for _ in range(iterations):
            func(True)
        frozen.append(time.perf_counter() - start)
    return pooled, frozen


def _outsim_payload() -> bytes:
    base = [
        0.0,
        0.0,
        0.0,
        0.1,
        0.02,
        -0.01,
        0.5,
        0.0,
        -9.81,
        15.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    inputs = [0.5, 0.1, 0.0, 0.0, -0.2]
    wheels = [
        0.01,
        0.02,
        110.0,
        95.0,
        350.0,
        0.05,
    ] * 4
    return struct.pack("<I15fI5f24f", 12345, *base, 7, *inputs, *wheels)


def _outgauge_payload() -> bytes:
    def _pad(value: str, size: int) -> bytes:
        return value.encode("latin-1").ljust(size, b"\x00")

    base = struct.pack(
        "<I4s16s8s6s6sHBBfffffffIIfff16s16sI",
        67890,
        _pad("XFG", 4),
        _pad("Driver", 16),
        _pad("01", 8),
        _pad("BL1", 6),
        _pad("", 6),
        0,
        3,
        0,
        75.0,
        6200.0,
        0.3,
        95.0,
        45.0,
        4.5,
        92.0,
        0,
        0,
        0.6,
        0.1,
        0.0,
        _pad("HUD1", 16),
        _pad("HUD2", 16),
        321,
    )
    extras = struct.pack(
        "<20f",
        *(
            (95.0, 94.0, 93.5, 92.0)
            + (91.0, 90.5, 88.0, 87.5)
            + (86.0, 85.5, 84.0, 83.5)
            + (1.6, 1.55, 1.5, 1.45)
            + (420.0, 410.0, 400.0, 390.0)
        )
    )
    return base + extras


def _decode_outsim(payload: bytes) -> Callable[[bool], None]:
    def _decode(freeze: bool) -> None:
        packet = OutSimPacket.from_bytes(payload, freeze=freeze)
        if not freeze:
            packet.release()

    return _decode


def _decode_outgauge(payload: bytes) -> Callable[[bool], None]:
    def _decode(freeze: bool) -> None:
        packet = OutGaugePacket.from_bytes(payload, freeze=freeze)
        if not freeze:
            packet.release()

    return _decode


def run(iterations: int, repeats: int) -> None:
    outsim_payload = _outsim_payload()
    outgauge_payload = _outgauge_payload()

    outsim_pooled, outsim_frozen = _measure(
        _decode_outsim(outsim_payload), iterations=iterations, repeats=repeats
    )
    outgauge_pooled, outgauge_frozen = _measure(
        _decode_outgauge(outgauge_payload), iterations=iterations, repeats=repeats
    )

    print("OutSim decoding")
    print("  " + _format("pooled", outsim_pooled, iterations))
    print("  " + _format("freeze=True", outsim_frozen, iterations))
    print("OutGauge decoding")
    print("  " + _format("pooled", outgauge_pooled, iterations))
    print("  " + _format("freeze=True", outgauge_frozen, iterations))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10000, help="Packets decoded per run")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timing repeats")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(iterations=max(args.iterations, 1), repeats=max(args.repeats, 1))
