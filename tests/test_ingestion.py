"""Ingestion pipeline regression tests."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from io import StringIO
import logging
import math
import socket
import struct
import time

import pytest

from tnfr_lfs.ingestion import outgauge_udp as outgauge_module
from tnfr_lfs.ingestion import outsim_udp as outsim_module
from tnfr_lfs.ingestion.live import (
    DEFAULT_SCHEMA,
    LEGACY_COLUMNS,
    OPTIONAL_SCHEMA_COLUMNS,
    OutGaugeUDPClient,
    OutSimClient,
    OutSimUDPClient,
    TelemetryFusion,
)
from tnfr_lfs.ingestion.outgauge_udp import FrozenOutGaugePacket, OutGaugePacket
from tnfr_lfs.ingestion.outsim_udp import FrozenOutSimPacket, OutSimPacket
from tests.helpers import (
    QueueUDPSocket,
    append_once_on_wait,
    build_extended_outsim_packet,
    build_extended_outsim_payload,
    build_outgauge_payload,
    build_outsim_payload,
    build_sample_outgauge_packet,
    build_synthetic_packet_pair,
    make_wait_stub,
)
@pytest.fixture
def extended_outsim_payload() -> bytes:
    return build_extended_outsim_payload()


@pytest.fixture
def extended_outsim_packet(extended_outsim_payload: bytes) -> FrozenOutSimPacket:
    return build_extended_outsim_packet()


@pytest.fixture
def zero_deflection_outsim_packet(
    extended_outsim_packet: FrozenOutSimPacket,
) -> FrozenOutSimPacket:
    zero_wheels = tuple(
        replace(wheel, suspension_deflection=0.0) for wheel in extended_outsim_packet.wheels
    )
    return replace(extended_outsim_packet, wheels=zero_wheels)


@pytest.fixture
def sample_outgauge_packet() -> FrozenOutGaugePacket:
    return build_sample_outgauge_packet()


def test_outsim_ingest_captures_per_wheel_slip_and_radius() -> None:
    schema = DEFAULT_SCHEMA
    header = ",".join(schema.columns)
    values = {
        "timestamp": "0.1",
        "structural_timestamp": "0.1",
        "vertical_load": "5200",
        "slip_ratio": "0.02",
        "slip_ratio_fl": "0.03",
        "slip_ratio_fr": "0.01",
        "slip_ratio_rl": "0.02",
        "slip_ratio_rr": "0.02",
        "lateral_accel": "1.5",
        "longitudinal_accel": "-0.2",
        "yaw": "0.05",
        "pitch": "0.01",
        "roll": "0.0",
        "brake_pressure": "12.5",
        "locking": "0.0",
        "nfr": "510.0",
        "si": "0.82",
        "speed": "45.0",
        "yaw_rate": "0.25",
        "slip_angle": "0.08",
        "slip_angle_fl": "0.11",
        "slip_angle_fr": "0.06",
        "slip_angle_rl": "0.05",
        "slip_angle_rr": "0.04",
        "steer": "0.12",
        "throttle": "0.3",
        "gear": "3",
        "vertical_load_front": "2600",
        "vertical_load_rear": "2600",
        "mu_eff_front": "1.2",
        "mu_eff_rear": "1.1",
        "mu_eff_front_lateral": "1.15",
        "mu_eff_front_longitudinal": "1.05",
        "mu_eff_rear_lateral": "1.08",
        "mu_eff_rear_longitudinal": "1.0",
        "suspension_travel_front": "0.03",
        "suspension_travel_rear": "0.04",
        "suspension_velocity_front": "0.5",
        "suspension_velocity_rear": "0.4",
        "tyre_temp_fl": "88.0",
        "tyre_temp_fr": "87.5",
        "tyre_temp_rl": "85.0",
        "tyre_temp_rr": "84.5",
        "tyre_pressure_fl": "1.45",
        "tyre_pressure_fr": "1.43",
        "tyre_pressure_rl": "1.42",
        "tyre_pressure_rr": "1.41",
        "instantaneous_radius": "14.0",
        "front_track_width": "1.46",
        "wheelbase": "2.74",
    }
    payload = ",".join(values[column] for column in schema.columns)
    buffer = StringIO(f"{header}\n{payload}\n")
    client = OutSimClient(schema=schema)
    records = client.ingest(buffer)
    assert len(records) == 1
    record = records[0]
    assert record.slip_angle_fl == pytest.approx(0.11)
    assert record.slip_angle_fr == pytest.approx(0.06)
    assert record.instantaneous_radius == pytest.approx(14.0)
    assert record.front_track_width == pytest.approx(1.46)
    assert record.wheelbase == pytest.approx(2.74)


def test_fusion_incremental_scaling() -> None:
    fusion = TelemetryFusion()
    total_samples = 1200
    half = total_samples // 2
    start = time.perf_counter()
    midpoint: float | None = None
    for index in range(total_samples):
        outsim, outgauge = build_synthetic_packet_pair(index)
        bundle = fusion.fuse_to_bundle(outsim, outgauge)
        assert bundle is not None
        if index == half - 1:
            midpoint = time.perf_counter()
    end = time.perf_counter()
    assert midpoint is not None
    first_span = midpoint - start
    second_span = end - midpoint
    assert second_span <= first_span * 1.8
    assert len(fusion.extractor._nu_f_analyzer._history) <= 150
    assert len(fusion._vertical_history) <= 64
    assert len(fusion._line_history) <= 60
    assert fusion.extractor._baseline.count == total_samples
    assert not hasattr(fusion, "_records")


def test_fusion_preserves_zero_suspension_deflection(
    zero_deflection_outsim_packet: FrozenOutSimPacket,
    sample_outgauge_packet: FrozenOutGaugePacket,
) -> None:
    fusion = TelemetryFusion()
    record = fusion.fuse(zero_deflection_outsim_packet, sample_outgauge_packet)

    assert record.suspension_travel_front == pytest.approx(0.0)
    assert record.suspension_travel_rear == pytest.approx(0.0)

    front_velocity = record.suspension_velocity_front
    rear_velocity = record.suspension_velocity_rear
    if math.isnan(front_velocity):
        assert True
    else:
        assert front_velocity == pytest.approx(0.0)
    if math.isnan(rear_velocity):
        assert True
    else:
        assert rear_velocity == pytest.approx(0.0)


def test_outsim_ingest_sets_nan_for_missing_optional_columns() -> None:
    schema = DEFAULT_SCHEMA
    required_columns = [
        column for column in schema.columns if column not in OPTIONAL_SCHEMA_COLUMNS
    ]
    header = ",".join(required_columns)
    values = {
        "timestamp": "0.2",
        "vertical_load": "4800",
        "slip_ratio": "0.015",
        "lateral_accel": "1.2",
        "longitudinal_accel": "-0.1",
        "yaw": "0.04",
        "pitch": "0.005",
        "roll": "0.0",
        "brake_pressure": "10.5",
        "locking": "0.0",
        "nfr": "500.0",
        "si": "0.8",
        "speed": "50.0",
        "yaw_rate": "0.2",
        "slip_angle": "0.07",
        "steer": "0.1",
        "throttle": "0.25",
        "gear": "4",
        "vertical_load_front": "2400",
        "vertical_load_rear": "2400",
        "mu_eff_front": "1.1",
        "mu_eff_rear": "1.05",
        "mu_eff_front_lateral": "1.12",
        "mu_eff_front_longitudinal": "1.03",
        "mu_eff_rear_lateral": "1.07",
        "mu_eff_rear_longitudinal": "0.98",
        "suspension_travel_front": "0.025",
        "suspension_travel_rear": "0.03",
        "suspension_velocity_front": "0.45",
        "suspension_velocity_rear": "0.35",
    }
    payload = ",".join(values[column] for column in required_columns)
    buffer = StringIO(f"{header}\n{payload}\n")
    client = OutSimClient(schema=schema)
    record = client.ingest(buffer)[0]

    assert math.isnan(record.slip_ratio_fl)
    assert math.isnan(record.slip_ratio_rr)
    assert math.isnan(record.slip_angle_fl)
    assert math.isnan(record.slip_angle_rr)
    assert math.isnan(record.tyre_temp_fl)
    assert math.isnan(record.tyre_temp_rr)
    assert math.isnan(record.tyre_pressure_fl)
    assert math.isnan(record.tyre_pressure_rr)
    assert math.isnan(record.instantaneous_radius)
    assert math.isnan(record.front_track_width)
    assert math.isnan(record.wheelbase)
    assert math.isnan(record.rpm)
    assert math.isnan(record.line_deviation)


def test_outsim_ingest_legacy_defaults_are_nan() -> None:
    header = ",".join(LEGACY_COLUMNS)
    payload_values = [
        "0.0",
        "5000.0",
        "0.01",
        "1.2",
        "0.3",
        "0.02",
        "0.01",
        "0.0",
        "15.0",
        "0.0",
        "450.0",
        "0.75",
    ]
    buffer = StringIO(f"{header}\n{','.join(payload_values)}\n")
    client = OutSimClient(schema=DEFAULT_SCHEMA)
    record = client.ingest(buffer)[0]

    assert math.isnan(record.speed)
    assert math.isnan(record.yaw_rate)
    assert math.isnan(record.slip_angle)
    assert math.isnan(record.slip_ratio_fl)
    assert math.isnan(record.slip_angle_fl)
    assert math.isnan(record.vertical_load_front)
    assert math.isnan(record.mu_eff_front)
    assert math.isnan(record.suspension_travel_front)
    assert math.isnan(record.tyre_temp_fl)
    assert math.isnan(record.tyre_pressure_fl)
    assert math.isnan(record.rpm)
    assert math.isnan(record.line_deviation)
    assert math.isnan(record.instantaneous_radius)
    assert math.isnan(record.front_track_width)
    assert math.isnan(record.wheelbase)

def test_outsim_packet_from_bytes_parses_extended_layout(
    extended_outsim_packet: FrozenOutSimPacket,
) -> None:
    packet = extended_outsim_packet
    assert packet.player_id == 7
    assert packet.inputs is not None
    assert packet.inputs.throttle == pytest.approx(0.72)
    assert packet.inputs.brake == pytest.approx(0.35)
    assert packet.inputs.clutch == pytest.approx(0.1)
    assert packet.inputs.handbrake == pytest.approx(0.0)
    assert packet.inputs.steer == pytest.approx(-0.15)
    first_wheel = packet.wheels[0]
    last_wheel = packet.wheels[3]
    assert first_wheel.slip_ratio == pytest.approx(0.02)
    assert first_wheel.lateral_force == pytest.approx(180.0)
    assert first_wheel.longitudinal_force == pytest.approx(120.0)
    assert first_wheel.load == pytest.approx(310.0)
    assert first_wheel.suspension_deflection == pytest.approx(0.06)
    assert last_wheel.slip_angle == pytest.approx(0.025)
    assert last_wheel.load == pytest.approx(285.0)


def test_fusion_consumes_extended_outsim_packet(
    extended_outsim_packet: FrozenOutSimPacket,
    sample_outgauge_packet: FrozenOutGaugePacket,
) -> None:
    fusion = TelemetryFusion()
    record = fusion.fuse(extended_outsim_packet, sample_outgauge_packet)
    wheel_slip_ratios = [
        wheel.slip_ratio for wheel in extended_outsim_packet.wheels[:4] if wheel.decoded
    ]
    expected_slip_ratio = sum(wheel_slip_ratios) / len(wheel_slip_ratios)
    wheel_angles_and_loads = [
        (wheel.slip_angle, wheel.load)
        for wheel in extended_outsim_packet.wheels[:4]
        if wheel.decoded
    ]
    weighted_sum = 0.0
    weight_total = 0.0
    for angle, load in wheel_angles_and_loads:
        weight = load if math.isfinite(load) and load > 1e-3 else 1.0
        weighted_sum += angle * weight
        weight_total += weight
    expected_slip_angle = weighted_sum / weight_total
    assert record.slip_ratio == pytest.approx(expected_slip_ratio)
    assert record.slip_angle == pytest.approx(expected_slip_angle)
    assert record.throttle == pytest.approx(0.72)
    assert record.brake_pressure == pytest.approx(0.2)
    assert record.brake_input == pytest.approx(0.35)
    assert record.clutch_input == pytest.approx(0.1)
    assert record.handbrake_input == pytest.approx(0.0)
    assert record.steer_input == pytest.approx(-0.15)
    assert record.slip_ratio_fl == pytest.approx(0.02)
    assert record.slip_ratio_rr == pytest.approx(0.015)
    assert record.slip_angle_fr == pytest.approx(0.04)
    assert record.wheel_load_fl == pytest.approx(310.0)
    assert record.wheel_load_rr == pytest.approx(285.0)
    assert record.suspension_deflection_fl == pytest.approx(0.06)
    assert record.suspension_deflection_rr == pytest.approx(0.05)
    assert record.wheel_lateral_force_fr == pytest.approx(175.0)
    assert record.wheel_longitudinal_force_rl == pytest.approx(130.0)
    assert record.vertical_load == pytest.approx(1190.0)
    assert record.vertical_load_front == pytest.approx(615.0)
    assert record.vertical_load_rear == pytest.approx(575.0)
    assert record.suspension_travel_front == pytest.approx((0.06 + 0.058) * 0.5)
    assert record.suspension_travel_rear == pytest.approx((0.052 + 0.05) * 0.5)
    assert math.isnan(record.tyre_temp_fl)
    assert math.isnan(record.tyre_temp_fr)
    assert math.isnan(record.tyre_temp_rl)
    assert math.isnan(record.tyre_temp_rr)
    assert math.isnan(record.tyre_pressure_fl)
    assert math.isnan(record.tyre_pressure_fr)
    assert math.isnan(record.tyre_pressure_rl)
    assert math.isnan(record.tyre_pressure_rr)


def test_udp_client_preserves_extended_payload_for_fusion(
    extended_outsim_payload: bytes, sample_outgauge_packet: FrozenOutGaugePacket
) -> None:
    client = OutSimUDPClient(host="127.0.0.1", port=0, timeout=0.01, retries=20)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(extended_outsim_payload, ("127.0.0.1", client.address[1]))
        packet = client.recv()
        assert packet is not None
        assert packet.inputs is not None

        fusion = TelemetryFusion()
        record = fusion.fuse(packet, sample_outgauge_packet)

        assert record.throttle == pytest.approx(0.72)
        assert record.brake_input == pytest.approx(0.35)
        assert record.clutch_input == pytest.approx(0.1)
        assert record.steer_input == pytest.approx(-0.15)
        assert record.wheel_load_fl == pytest.approx(310.0)
        assert record.wheel_load_rr == pytest.approx(285.0)
        assert record.suspension_deflection_fl == pytest.approx(0.06)
        assert record.suspension_deflection_rr == pytest.approx(0.05)
        packet.release()
    finally:
        sender.close()
        client.close()


@pytest.mark.parametrize(
    ("client_cls", "remote_host"),
    (
        (OutSimUDPClient, "203.0.113.1"),
        (OutGaugeUDPClient, "198.51.100.1"),
    ),
)
def test_udp_client_allows_remote_host_binding(
    client_cls: type[OutSimUDPClient | OutGaugeUDPClient], remote_host: str
) -> None:
    client = client_cls(host=remote_host, port=0)
    try:
        host, port = client.address
        assert host == "0.0.0.0"
        assert port > 0
    finally:
        client.close()


def test_outgauge_from_bytes_decodes_extended_tyre_payload(
    extended_outsim_packet: FrozenOutSimPacket,
) -> None:
    def _pad(value: str, size: int) -> bytes:
        encoded = value.encode("latin-1")
        if len(encoded) > size:
            raise AssertionError("value too long for OutGauge field")
        return encoded + b"\x00" * (size - len(encoded))

    base_payload = struct.pack(
        "<I4s16s8s6s6sHBBfffffffIIfff16s16sI",
        43210,
        _pad("XFG", 4),
        _pad("Driver", 16),
        _pad("123", 8),
        _pad("BL1", 6),
        _pad("", 6),
        0,
        3,
        0,
        22.5,
        5150.0,
        0.0,
        92.0,
        40.5,
        4.2,
        80.0,
        0,
        0,
        0.41,
        0.12,
        0.02,
        _pad("FUEL", 16),
        _pad("TEMP", 16),
        77,
    )
    inner = (95.0, 94.0, 88.5, 87.5)
    middle = (90.0, 89.0, 84.0, 83.0)
    outer = (85.0, 84.0, 79.0, 78.0)
    pressures = (1.58, 1.56, 1.48, 1.46)
    brakes = (420.0, 410.0, 400.0, 395.0)
    extras = struct.pack("<20f", *(inner + middle + outer + pressures + brakes))

    packet = OutGaugePacket.from_bytes(base_payload + extras, freeze=True)

    for value, expected in zip(packet.tyre_temps_inner, inner):
        assert value == pytest.approx(expected)
    for value, expected in zip(packet.tyre_temps_middle, middle):
        assert value == pytest.approx(expected)
    for value, expected in zip(packet.tyre_temps_outer, outer):
        assert value == pytest.approx(expected)
    for value, expected in zip(packet.tyre_pressures, pressures):
        assert value == pytest.approx(expected)
    for value, expected in zip(packet.brake_temps, brakes):
        assert value == pytest.approx(expected)

    expected_average = tuple(
        (inner[idx] + middle[idx] + outer[idx]) / 3.0 for idx in range(4)
    )
    for value, expected in zip(packet.tyre_temps, expected_average):
        assert value == pytest.approx(expected)

    fusion = TelemetryFusion()
    record = fusion.fuse(extended_outsim_packet, packet)

    assert record.tyre_pressure_fl == pytest.approx(pressures[0])
    assert record.tyre_pressure_fr == pytest.approx(pressures[1])
    assert record.tyre_pressure_rl == pytest.approx(pressures[2])
    assert record.tyre_pressure_rr == pytest.approx(pressures[3])


def test_fusion_uses_outgauge_tyre_temperatures(
    extended_outsim_packet: FrozenOutSimPacket,
    sample_outgauge_packet: FrozenOutGaugePacket,
) -> None:
    fusion = TelemetryFusion()
    tyre_temps = (88.3, 87.6, 84.2, 83.9)
    inner = (92.1, 91.4, 87.3, 86.5)
    middle = (88.0, 87.1, 84.0, 83.1)
    outer = (85.4, 84.8, 81.9, 81.0)
    enriched_packet = replace(
        sample_outgauge_packet,
        tyre_temps=tyre_temps,
        tyre_temps_inner=inner,
        tyre_temps_middle=middle,
        tyre_temps_outer=outer,
    )

    record = fusion.fuse(extended_outsim_packet, enriched_packet)

    assert record.tyre_temp_fl == pytest.approx(tyre_temps[0])
    assert record.tyre_temp_fr == pytest.approx(tyre_temps[1])
    assert record.tyre_temp_rl == pytest.approx(tyre_temps[2])
    assert record.tyre_temp_rr == pytest.approx(tyre_temps[3])
    assert record.tyre_temp_fl_inner == pytest.approx(inner[0])
    assert record.tyre_temp_fr_inner == pytest.approx(inner[1])
    assert record.tyre_temp_rl_inner == pytest.approx(inner[2])
    assert record.tyre_temp_rr_inner == pytest.approx(inner[3])
    assert record.tyre_temp_fl_middle == pytest.approx(middle[0])
    assert record.tyre_temp_fr_middle == pytest.approx(middle[1])
    assert record.tyre_temp_rl_middle == pytest.approx(middle[2])
    assert record.tyre_temp_rr_middle == pytest.approx(middle[3])
    assert record.tyre_temp_fl_outer == pytest.approx(outer[0])
    assert record.tyre_temp_fr_outer == pytest.approx(outer[1])
    assert record.tyre_temp_rl_outer == pytest.approx(outer[2])
    assert record.tyre_temp_rr_outer == pytest.approx(outer[3])


def test_fusion_marks_missing_wheel_block_as_nan(
    sample_outgauge_packet: FrozenOutGaugePacket,
) -> None:
    time_ms = 5678
    base_floats = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -9.81,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    payload = struct.pack("<I15f", time_ms, *base_floats)
    packet = OutSimPacket.from_bytes(payload)

    fusion = TelemetryFusion()
    record = fusion.fuse(packet, sample_outgauge_packet)

    wheel_attrs = [
        "slip_ratio_fl",
        "slip_ratio_fr",
        "slip_ratio_rl",
        "slip_ratio_rr",
        "slip_angle_fl",
        "slip_angle_fr",
        "slip_angle_rl",
        "slip_angle_rr",
        "wheel_load_fl",
        "wheel_load_fr",
        "wheel_load_rl",
        "wheel_load_rr",
        "wheel_lateral_force_fl",
        "wheel_lateral_force_fr",
        "wheel_lateral_force_rl",
        "wheel_lateral_force_rr",
        "wheel_longitudinal_force_fl",
        "wheel_longitudinal_force_fr",
        "wheel_longitudinal_force_rl",
        "wheel_longitudinal_force_rr",
        "suspension_deflection_fl",
        "suspension_deflection_fr",
        "suspension_deflection_rl",
        "suspension_deflection_rr",
    ]
    for attribute in wheel_attrs:
        assert math.isnan(getattr(record, attribute)), attribute

    assert math.isnan(record.suspension_travel_front)
    assert math.isnan(record.suspension_travel_rear)
    assert math.isnan(record.suspension_velocity_front)
    assert math.isnan(record.suspension_velocity_rear)
    packet.release()


def test_outsim_udp_client_reorders_and_tracks_losses(monkeypatch, caplog) -> None:
    payloads: deque[tuple[bytes, tuple[str, int]]] = deque(
        [
            (build_outsim_payload(100), ("127.0.0.1", 4123)),
            (build_outsim_payload(160), ("127.0.0.1", 4123)),
            (build_outsim_payload(130), ("127.0.0.1", 4123)),
            (build_outsim_payload(290), ("127.0.0.1", 4123)),
            (build_outsim_payload(290), ("127.0.0.1", 4123)),
        ]
    )
    dummy_socket = QueueUDPSocket(queue=payloads)
    monkeypatch.setattr(outsim_module.socket, "socket", lambda *_, **__: dummy_socket)
    monkeypatch.setattr(outsim_module, "wait_for_read_ready", lambda *_, **__: True)
    client = outsim_module.OutSimUDPClient(
        host="127.0.0.1",
        port=0,
        timeout=0.0,
        retries=2,
        reorder_grace=0.01,
        jump_tolerance=100,
    )
    caplog.set_level(logging.WARNING, logger=outsim_module.__name__)

    delivered: list[int] = []
    for _ in range(4):
        packet = client.recv()
        assert packet is not None
        delivered.append(packet.time)
        packet.release()

    assert delivered == [100, 130, 160, 290]
    stats = client.statistics
    assert stats["duplicates"] == 1
    assert stats["reordered"] >= 1
    assert stats["late_recovered"] >= 1
    assert stats["loss_events"] == 1
    assert any("out-of-order" in record.message for record in caplog.records)
    assert any("time jump" in record.message for record in caplog.records)
    client.close()


def test_outsim_udp_client_flushes_pending_when_successor_arrives(monkeypatch) -> None:
    payloads: deque[tuple[bytes, tuple[str, int]]] = deque(
        [
            (build_outsim_payload(100), ("127.0.0.1", 4123)),
            (build_outsim_payload(120), ("127.0.0.1", 4123)),
        ]
    )
    dummy_socket = QueueUDPSocket(queue=payloads)
    monkeypatch.setattr(outsim_module.socket, "socket", lambda *_, **__: dummy_socket)
    client = outsim_module.OutSimUDPClient(
        host="127.0.0.1",
        port=0,
        timeout=0.2,
        retries=2,
        reorder_grace=0.5,
    )

    packet = client.recv()
    assert packet is not None
    packet.release()

    fake_wait, wait_calls = make_wait_stub(
        hook=append_once_on_wait(
            payloads,
            lambda: (build_outsim_payload(140), ("127.0.0.1", 4123)),
        )
    )
    monkeypatch.setattr(outsim_module, "wait_for_read_ready", fake_wait)

    start = time.perf_counter()
    next_packet = client.recv()
    elapsed = time.perf_counter() - start

    try:
        assert next_packet is not None
        assert next_packet.time == 120
        assert elapsed < 0.2
        assert wait_calls
    finally:
        if next_packet is not None:
            next_packet.release()
        client.close()


def test_outgauge_udp_client_recovers_late_packets(monkeypatch, caplog) -> None:
    payloads: deque[tuple[bytes, tuple[str, int]]] = deque(
        [
            (
                build_outgauge_payload(0, 0, layout="GP"),
                ("127.0.0.1", 3000),
            ),
            (
                build_outgauge_payload(1, 10, layout="GP"),
                ("127.0.0.1", 3000),
            ),
            (
                build_outgauge_payload(3, 30, layout="GP"),
                ("127.0.0.1", 3000),
            ),
            (
                build_outgauge_payload(2, 20, layout="GP"),
                ("127.0.0.1", 3000),
            ),
        ]
    )
    dummy_socket = QueueUDPSocket(queue=payloads)
    monkeypatch.setattr(outgauge_module.socket, "socket", lambda *_, **__: dummy_socket)
    monkeypatch.setattr(outgauge_module, "wait_for_read_ready", lambda *_, **__: True)
    client = outgauge_module.OutGaugeUDPClient(
        host="127.0.0.1",
        port=0,
        timeout=0.0,
        retries=2,
        reorder_grace=0.01,
        jump_tolerance=50,
    )
    caplog.set_level(logging.WARNING, logger=outgauge_module.__name__)

    delivered_ids: list[int] = []
    packet = client.recv()
    assert packet is not None
    delivered_ids.append(packet.packet_id)
    packet.release()
    packet = client.recv()
    assert packet is not None
    delivered_ids.append(packet.packet_id)
    packet.release()
    packet = client.recv()
    assert packet is not None
    delivered_ids.append(packet.packet_id)
    packet.release()
    time.sleep(0.02)
    packet = client.recv()
    assert packet is not None
    delivered_ids.append(packet.packet_id)
    packet.release()

    assert delivered_ids == [0, 1, 2, 3]
    stats = client.statistics
    assert stats["loss_events"] == 1
    assert stats["recovered"] == 1
    assert stats["reordered"] >= 1
    assert any("packet gap" in record.message for record in caplog.records)
    assert any("out-of-order" in record.message for record in caplog.records)
    client.close()


def test_outgauge_udp_client_flushes_pending_when_successor_arrives(monkeypatch) -> None:
    payloads: deque[tuple[bytes, tuple[str, int]]] = deque(
        [
            (build_outgauge_payload(5, 50, layout="GP"), ("127.0.0.1", 3000)),
            (build_outgauge_payload(6, 60, layout="GP"), ("127.0.0.1", 3000)),
        ]
    )
    dummy_socket = QueueUDPSocket(queue=payloads)
    monkeypatch.setattr(outgauge_module.socket, "socket", lambda *_, **__: dummy_socket)
    client = outgauge_module.OutGaugeUDPClient(
        host="127.0.0.1",
        port=0,
        timeout=0.2,
        retries=2,
        reorder_grace=0.5,
    )

    packet = client.recv()
    assert packet is not None
    packet.release()

    fake_wait, wait_calls = make_wait_stub(
        hook=append_once_on_wait(
            payloads,
            lambda: (build_outgauge_payload(7, 70, layout="GP"), ("127.0.0.1", 3000)),
        )
    )
    monkeypatch.setattr(outgauge_module, "wait_for_read_ready", fake_wait)

    start = time.perf_counter()
    next_packet = client.recv()
    elapsed = time.perf_counter() - start

    try:
        assert next_packet is not None
        assert next_packet.packet_id == 6
        assert elapsed < 0.2
        assert wait_calls
    finally:
        if next_packet is not None:
            next_packet.release()
        client.close()
