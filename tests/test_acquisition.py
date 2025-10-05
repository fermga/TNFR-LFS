"""Acquisition pipeline regression tests."""

from __future__ import annotations

from io import StringIO
import struct

import pytest

from tnfr_lfs.acquisition.fusion import TelemetryFusion
from tnfr_lfs.acquisition.outgauge_udp import OutGaugePacket
from tnfr_lfs.acquisition.outsim_client import DEFAULT_SCHEMA, OutSimClient
from tnfr_lfs.acquisition.outsim_udp import OutSimPacket


@pytest.fixture
def extended_outsim_packet() -> OutSimPacket:
    time_ms = 1234
    base_floats = [
        0.1,
        0.2,
        0.3,
        0.45,
        0.05,
        -0.02,
        0.5,
        0.6,
        -9.2,
        15.0,
        1.5,
        0.0,
        102.0,
        205.0,
        0.4,
    ]
    player_id = 7
    driver_inputs = [0.72, 0.35, 0.1, 0.0, -0.15]
    wheel_values = [
        0.02,
        0.05,
        120.0,
        180.0,
        310.0,
        0.06,
        0.03,
        0.04,
        115.0,
        175.0,
        305.0,
        0.058,
        0.01,
        0.03,
        130.0,
        165.0,
        290.0,
        0.052,
        0.015,
        0.025,
        125.0,
        160.0,
        285.0,
        0.05,
    ]
    payload = struct.pack(
        "<I15fI5f24f",
        time_ms,
        *base_floats,
        player_id,
        *driver_inputs,
        *wheel_values,
    )
    return OutSimPacket.from_bytes(payload)


@pytest.fixture
def sample_outgauge_packet() -> OutGaugePacket:
    return OutGaugePacket(
        time=0,
        car="XFG",
        player_name="Driver",
        plate="",
        track="BL1",
        layout="",
        flags=0,
        gear=3,
        plid=0,
        speed=15.0,
        rpm=5200.0,
        turbo=0.0,
        eng_temp=0.0,
        fuel=40.0,
        oil_pressure=0.0,
        oil_temp=0.0,
        dash_lights=0,
        show_lights=0,
        throttle=0.3,
        brake=0.2,
        clutch=0.1,
        display1="",
        display2="",
        packet_id=0,
    )


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


def test_outsim_packet_from_bytes_parses_extended_layout(extended_outsim_packet: OutSimPacket) -> None:
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
    extended_outsim_packet: OutSimPacket, sample_outgauge_packet: OutGaugePacket
) -> None:
    fusion = TelemetryFusion()
    record = fusion.fuse(extended_outsim_packet, sample_outgauge_packet)
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
