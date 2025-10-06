"""Telemetry acquisition backends."""

from .insim import ButtonEvent, ButtonLayout, InSimClient, MacroQueue, OverlayManager
from .outsim_client import OutSimClient, TelemetryFormatError
from .outsim_udp import OUTSIM_MAX_PACKET_SIZE, OutSimPacket, OutSimUDPClient
from .outgauge_udp import OutGaugePacket, OutGaugeUDPClient
from .fusion import TelemetryFusion

DEFAULT_TIMEOUT = 0.05
DEFAULT_RETRIES = 5

__all__ = [
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT",
    "ButtonEvent",
    "ButtonLayout",
    "InSimClient",
    "MacroQueue",
    "OverlayManager",
    "OutGaugePacket",
    "OutGaugeUDPClient",
    "OUTSIM_MAX_PACKET_SIZE",
    "OutSimClient",
    "OutSimPacket",
    "OutSimUDPClient",
    "TelemetryFormatError",
    "TelemetryFusion",
]
