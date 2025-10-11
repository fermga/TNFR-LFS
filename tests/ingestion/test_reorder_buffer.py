from __future__ import annotations

import logging
from itertools import count

import pytest

from tnfr_lfs.ingestion import outgauge_udp as outgauge_module
from tnfr_lfs.ingestion import outsim_udp as outsim_module
from tnfr_lfs.ingestion._reorder_buffer import CircularReorderBuffer

from tests.helpers import build_outgauge_packet, build_outsim_packet


def test_circular_buffer_orders_out_of_sequence_packets() -> None:
    buffer = CircularReorderBuffer[str](3)
    buffer.insert(1.0, "later", 20)
    buffer.insert(0.5, "earlier", 10)

    entries = list(buffer)
    assert [key for _, _, key in entries] == [10, 20]

    popped = buffer.pop_oldest()
    assert popped == (0.5, "earlier")


def test_circular_buffer_discards_oldest_when_full() -> None:
    buffer = CircularReorderBuffer[str](2)
    buffer.insert(0.0, "first", 1)
    buffer.insert(0.1, "second", 2)
    index, evicted = buffer.insert(0.2, "middle", 15)

    assert index == 1
    assert evicted is not None
    assert evicted.key == 1
    assert evicted.packet == "first"
    assert [key for _, _, key in buffer] == [2, 15]


def test_outsim_reorders_packets_and_drains_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = count(start=1)

    def fake_monotonic() -> float:
        return float(next(clock))

    monkeypatch.setattr(outsim_module.time, "monotonic", fake_monotonic)

    client = outsim_module.OutSimUDPClient(port=0, timeout=0.0, retries=1, buffer_size=3)
    client._buffer_grace = 0.0

    try:
        client._record_packet(build_outsim_packet(time=200))
        client._record_packet(build_outsim_packet(time=100))

        drained = client.drain_ready()
        assert [packet.time for packet in drained] == [100, 200]
        stats = client.statistics
        assert stats["reordered"] == 1
        assert stats["late_recovered"] == 1
    finally:
        client.close()


def test_outgauge_capacity_discards_oldest(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = count(start=1)

    def fake_monotonic() -> float:
        return float(next(clock))

    monkeypatch.setattr(outgauge_module.time, "monotonic", fake_monotonic)

    client = outgauge_module.OutGaugeUDPClient(port=0, timeout=0.0, retries=1, buffer_size=2)
    client._buffer_grace = 0.0

    try:
        client._record_packet(build_outgauge_packet(packet_id=1, time=10))
        client._record_packet(build_outgauge_packet(packet_id=2, time=20))
        client._record_packet(build_outgauge_packet(packet_id=3, time=30))

        first = client._pop_ready_packet(now=100.0, allow_grace=True)
        second = client._pop_ready_packet(now=100.0, allow_grace=True)

        assert first is not None and first.packet_id == 2
        assert second is not None and second.packet_id == 3
    finally:
        client.close()


def test_outgauge_reorders_packets(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = count(start=1)

    def fake_monotonic() -> float:
        return float(next(clock))

    monkeypatch.setattr(outgauge_module.time, "monotonic", fake_monotonic)

    client = outgauge_module.OutGaugeUDPClient(port=0, timeout=0.0, retries=1, buffer_size=4)
    client._buffer_grace = 0.0

    try:
        client._record_packet(build_outgauge_packet(packet_id=2, time=20))
        client._record_packet(build_outgauge_packet(packet_id=1, time=10))

        drained = client.drain_ready()
        assert [packet.packet_id for packet in drained] == [1, 2]
        assert client.statistics["reordered"] == 1
    finally:
        client.close()


def test_outsim_buffer_overflow_records_loss_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=outsim_module.__name__)

    processor = outsim_module._OutSimPacketProcessor(
        remote_host="127.0.0.1",
        port=0,
        buffer_capacity=1,
        buffer_grace=0.0,
        jump_tolerance=0,
    )

    class DummyOutSimPacket:
        def __init__(self, time_value: int) -> None:
            self.time = time_value
            self.release_count = 0

        def release(self) -> None:
            self.release_count += 1

    first = DummyOutSimPacket(1)
    second = DummyOutSimPacket(2)

    processor.record_packet(first, arrival=0.0)
    processor.record_packet(second, arrival=0.1)

    assert first.release_count == 1
    assert processor.statistics["loss_events"] == 1
    overflow_records = [record for record in caplog.records if "overflow" in record.message]
    assert overflow_records
    assert any(getattr(record, "evicted_time", None) == 1 for record in overflow_records)

    async_client = outsim_module.AsyncOutSimUDPClient(buffer_size=1)
    try:
        async_client._processor = processor
        assert async_client.statistics["loss_events"] == 1
    finally:
        async_client._processor = None

    processor.flush()
    assert second.release_count == 1


def test_outgauge_buffer_overflow_records_loss_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=outgauge_module.__name__)

    processor = outgauge_module._OutGaugePacketProcessor(
        remote_host="127.0.0.1",
        port=0,
        buffer_capacity=1,
        buffer_grace=0.0,
        jump_tolerance=0,
    )

    class DummyOutGaugePacket:
        def __init__(self, packet_id: int) -> None:
            self.packet_id = packet_id
            self.time = packet_id
            self.release_count = 0

        def release(self) -> None:
            self.release_count += 1

    first = DummyOutGaugePacket(1)
    second = DummyOutGaugePacket(2)

    processor.record_packet(first, arrival=0.0)
    processor.record_packet(second, arrival=0.1)

    assert first.release_count == 1
    assert processor.statistics["loss_events"] == 1
    overflow_records = [record for record in caplog.records if "overflow" in record.message]
    assert overflow_records
    assert any(getattr(record, "evicted_packet_id", None) == 1 for record in overflow_records)

    async_client = outgauge_module.AsyncOutGaugeUDPClient(buffer_size=1)
    try:
        async_client._processor = processor
        assert async_client.statistics["loss_events"] == 1
    finally:
        async_client._processor = None

    processor.flush()
    assert second.release_count == 1
