from __future__ import annotations

from typing import Any, Dict

from tnfr_lfs.ingestion.outgauge_udp import OutGaugePacket
from tnfr_lfs.ingestion.outsim_udp import OutSimPacket


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
