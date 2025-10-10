from __future__ import annotations

from itertools import count

import pytest

from tnfr_lfs.ingestion import outgauge_udp as outgauge_module
from tnfr_lfs.ingestion import outsim_udp as outsim_module
from tnfr_lfs.ingestion._reorder_buffer import CircularReorderBuffer


def _make_outsim_packet(time_value: int) -> outsim_module.OutSimPacket:
    return outsim_module.OutSimPacket(
        time=time_value,
        ang_vel_x=0.0,
        ang_vel_y=0.0,
        ang_vel_z=0.0,
        heading=0.0,
        pitch=0.0,
        roll=0.0,
        accel_x=0.0,
        accel_y=0.0,
        accel_z=0.0,
        vel_x=0.0,
        vel_y=0.0,
        vel_z=0.0,
        pos_x=0.0,
        pos_y=0.0,
        pos_z=0.0,
    )


def _make_outgauge_packet(packet_id: int, time_value: int) -> outgauge_module.OutGaugePacket:
    return outgauge_module.OutGaugePacket(
        time=time_value,
        car="XFG",
        player_name="Driver",
        plate="",
        track="BL1",
        layout="",
        flags=0,
        gear=0,
        plid=0,
        speed=0.0,
        rpm=0.0,
        turbo=0.0,
        eng_temp=0.0,
        fuel=0.0,
        oil_pressure=0.0,
        oil_temp=0.0,
        dash_lights=0,
        show_lights=0,
        throttle=0.0,
        brake=0.0,
        clutch=0.0,
        display1="",
        display2="",
        packet_id=packet_id,
    )


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
        client._record_packet(_make_outsim_packet(200))
        client._record_packet(_make_outsim_packet(100))

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
        client._record_packet(_make_outgauge_packet(1, 10))
        client._record_packet(_make_outgauge_packet(2, 20))
        client._record_packet(_make_outgauge_packet(3, 30))

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
        client._record_packet(_make_outgauge_packet(2, 20))
        client._record_packet(_make_outgauge_packet(1, 10))

        drained = client.drain_ready()
        assert [packet.packet_id for packet in drained] == [1, 2]
        assert client.statistics["reordered"] == 1
    finally:
        client.close()
