from __future__ import annotations

import logging
from itertools import count

import pytest

from tnfr_lfs.telemetry import outgauge_udp as outgauge_module
from tnfr_lfs.telemetry import outsim_udp as outsim_module
from tnfr_lfs.telemetry._reorder_buffer import CircularReorderBuffer

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


@pytest.mark.parametrize(
    (
        "module",
        "client_attr",
        "packet_builder",
        "first_packet_kwargs",
        "second_packet_kwargs",
        "order_attr",
        "stat_keys",
        "buffer_size",
    ),
    [
        pytest.param(
            outsim_module,
            "OutSimUDPClient",
            build_outsim_packet,
            {"time": 200},
            {"time": 100},
            "time",
            ("reordered", "late_recovered"),
            3,
            id="outsim",
        ),
        pytest.param(
            outgauge_module,
            "OutGaugeUDPClient",
            build_outgauge_packet,
            {"packet_id": 2, "time": 20},
            {"packet_id": 1, "time": 10},
            "packet_id",
            ("reordered",),
            4,
            id="outgauge",
        ),
    ],
)
def test_udp_clients_reorder_packets(
    module,
    client_attr: str,
    packet_builder,
    first_packet_kwargs: dict[str, int],
    second_packet_kwargs: dict[str, int],
    order_attr: str,
    stat_keys: tuple[str, ...],
    buffer_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = count(start=1)

    def fake_monotonic() -> float:
        return float(next(clock))

    monkeypatch.setattr(module.time, "monotonic", fake_monotonic)

    client_cls = getattr(module, client_attr)
    client = client_cls(port=0, timeout=0.0, retries=1, buffer_size=buffer_size)
    client._buffer_grace = 0.0

    try:
        client._record_packet(packet_builder(**first_packet_kwargs))
        client._record_packet(packet_builder(**second_packet_kwargs))

        drained = client.drain_ready()
        expected_order = [second_packet_kwargs[order_attr], first_packet_kwargs[order_attr]]
        assert [getattr(packet, order_attr) for packet in drained] == expected_order
        stats = client.statistics
        for key in stat_keys:
            assert stats[key] == 1
    finally:
        client.close()


@pytest.mark.parametrize(
    (
        "module",
        "processor_attr",
        "async_client_attr",
        "identifier_attr",
        "overflow_attr",
    ),
    [
        pytest.param(
            outsim_module,
            "_OutSimPacketProcessor",
            "AsyncOutSimUDPClient",
            "time",
            "evicted_time",
            id="outsim",
        ),
        pytest.param(
            outgauge_module,
            "_OutGaugePacketProcessor",
            "AsyncOutGaugeUDPClient",
            "packet_id",
            "evicted_packet_id",
            id="outgauge",
        ),
    ],
)
def test_udp_buffer_overflow_records_loss_and_logs(
    module,
    processor_attr: str,
    async_client_attr: str,
    identifier_attr: str,
    overflow_attr: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=module.__name__)

    processor_cls = getattr(module, processor_attr)
    processor = processor_cls(
        remote_host="127.0.0.1",
        port=0,
        buffer_capacity=1,
        buffer_grace=0.0,
        jump_tolerance=0,
    )

    class DummyPacket:
        def __init__(self, value: int) -> None:
            setattr(self, identifier_attr, value)
            self.time = value
            self.release_count = 0

        def release(self) -> None:
            self.release_count += 1

    first = DummyPacket(1)
    second = DummyPacket(2)

    processor.record_packet(first, arrival=0.0)
    processor.record_packet(second, arrival=0.1)

    assert first.release_count == 1
    assert processor.statistics["loss_events"] == 1
    overflow_records = [record for record in caplog.records if "overflow" in record.message]
    assert overflow_records
    expected_identifier = getattr(first, identifier_attr)
    assert any(
        getattr(record, overflow_attr, None) == expected_identifier
        for record in overflow_records
    )

    async_client_cls = getattr(module, async_client_attr)
    async_client = async_client_cls(buffer_size=1)
    try:
        async_client._processor = processor
        assert async_client.statistics["loss_events"] == 1
    finally:
        async_client._processor = None

    processor.flush()
    assert second.release_count == 1
