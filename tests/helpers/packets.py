from __future__ import annotations

import struct
from types import SimpleNamespace
from typing import Any, Dict

from tnfr_lfs.telemetry.live import OutSimWheelState
from tnfr_lfs.telemetry.outgauge_udp import FrozenOutGaugePacket, OutGaugePacket
from tnfr_lfs.telemetry.outsim_udp import (
    FrozenOutSimPacket,
    FrozenOutSimWheelState,
    OutSimPacket,
)


_DEFAULT_OUTSIM_KWARGS: Dict[str, Any] = {
    "time": 0,
    "ang_vel_x": 0.0,
    "ang_vel_y": 0.0,
    "ang_vel_z": 0.0,
    "heading": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
    "accel_x": 0.0,
    "accel_y": 0.0,
    "accel_z": 0.0,
    "vel_x": 0.0,
    "vel_y": 0.0,
    "vel_z": 0.0,
    "pos_x": 0.0,
    "pos_y": 0.0,
    "pos_z": 0.0,
    "inputs": None,
    "wheels": None,
    "player_id": 0,
}


_DEFAULT_OUTGAUGE_KWARGS: Dict[str, Any] = {
    "time": 0,
    "car": "XFG",
    "player_name": "Driver",
    "plate": "",
    "track": "BL1",
    "layout": "",
    "flags": 0,
    "gear": 0,
    "plid": 0,
    "speed": 0.0,
    "rpm": 0.0,
    "turbo": 0.0,
    "eng_temp": 0.0,
    "fuel": 0.0,
    "oil_pressure": 0.0,
    "oil_temp": 0.0,
    "dash_lights": 0,
    "show_lights": 0,
    "throttle": 0.0,
    "brake": 0.0,
    "clutch": 0.0,
    "display1": "",
    "display2": "",
    "packet_id": 0,
    "brake_temps": (0.0, 0.0, 0.0, 0.0),
}


def build_outsim_packet(**overrides: Any) -> OutSimPacket:
    kwargs = _DEFAULT_OUTSIM_KWARGS.copy()
    kwargs.update(overrides)
    return OutSimPacket(**kwargs)


def build_outgauge_packet(**overrides: Any) -> OutGaugePacket:
    kwargs = _DEFAULT_OUTGAUGE_KWARGS.copy()
    kwargs.update(overrides)
    return OutGaugePacket(**kwargs)


def build_extended_outsim_payload() -> bytes:
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
    return struct.pack(
        "<I15fI5f24f",
        time_ms,
        *base_floats,
        player_id,
        *driver_inputs,
        *wheel_values,
    )


def build_extended_outsim_packet() -> FrozenOutSimPacket:
    payload = build_extended_outsim_payload()
    return OutSimPacket.from_bytes(payload, freeze=True)


def build_sample_outgauge_packet() -> FrozenOutGaugePacket:
    return FrozenOutGaugePacket(
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


def build_synthetic_packet_pair(
    index: int,
) -> tuple[FrozenOutSimPacket, FrozenOutGaugePacket]:
    time_ms = index * 50
    base_speed = 20.0 + 0.02 * index
    wheels = tuple(
        FrozenOutSimWheelState(
            slip_ratio=0.01 + 0.0001 * index + offset,
            slip_angle=0.02 + (offset * 5.0),
            longitudinal_force=110.0 + index + offset * 10.0,
            lateral_force=95.0 + index + offset * 8.0,
            load=900.0 + index + offset * 50.0,
            suspension_deflection=0.03 + offset,
            decoded=True,
        )
        for offset in (0.0, 0.0005, -0.0004, 0.0003)
    )
    outsim = FrozenOutSimPacket(
        time=time_ms,
        ang_vel_x=0.01,
        ang_vel_y=0.02,
        ang_vel_z=0.03,
        heading=0.1,
        pitch=0.05,
        roll=0.02,
        accel_x=0.5,
        accel_y=0.3,
        accel_z=-9.0,
        vel_x=base_speed,
        vel_y=0.5,
        vel_z=0.0,
        pos_x=float(index) * 0.1,
        pos_y=float(index) * 0.2,
        pos_z=0.0,
        player_id=1,
        inputs=None,
        wheels=wheels[:4],
    )
    outgauge = FrozenOutGaugePacket(
        time=index,
        car="XFG",
        player_name="Driver",
        plate="",
        track="BL1",
        layout="GP",
        flags=0,
        gear=3,
        plid=0,
        speed=base_speed,
        rpm=4000.0 + index,
        turbo=0.0,
        eng_temp=80.0,
        fuel=40.0,
        oil_pressure=0.0,
        oil_temp=90.0,
        dash_lights=0,
        show_lights=0,
        throttle=0.4,
        brake=0.2,
        clutch=0.1,
        display1="",
        display2="",
        packet_id=index,
    )
    return outsim, outgauge


def build_outsim_sample(
    timestamp: float,
    speed: float,
    slip: float,
    *,
    lateral: float = 6.0,
) -> OutSimPacket:
    reference_speed = max(speed, 1.0)
    vel_x = reference_speed * (1.0 + slip)
    return build_outsim_packet(
        time=int(timestamp * 1000),
        ang_vel_z=0.12,
        heading=0.01,
        pitch=0.02,
        roll=0.01,
        accel_x=0.3,
        accel_y=lateral,
        vel_x=vel_x,
        vel_y=reference_speed * 0.05,
        player_id=1,
    )


def build_outgauge_sample(
    car: str,
    track: str,
    speed: float,
    *,
    rpm: float = 5200.0,
) -> OutGaugePacket:
    return build_outgauge_packet(
        time=0,
        car=car,
        player_name="Test",
        track=track,
        gear=4,
        speed=speed,
        rpm=rpm,
        fuel=50.0,
        throttle=0.65,
        brake=0.1,
    )


def simple_outsim_namespace(
    *wheels: OutSimWheelState | FrozenOutSimWheelState,
) -> SimpleNamespace:
    return SimpleNamespace(wheels=wheels)
