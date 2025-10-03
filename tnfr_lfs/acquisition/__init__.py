"""Telemetry acquisition backends."""

from .outsim_client import OutSimClient, TelemetryFormatError
from .outsim_udp import OutSimPacket, OutSimUDPClient
from .outgauge_udp import OutGaugePacket, OutGaugeUDPClient
from .fusion import TelemetryFusion

DEFAULT_TIMEOUT = 0.05
DEFAULT_RETRIES = 5

__all__ = [
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT",
    "OutGaugePacket",
    "OutGaugeUDPClient",
    "OutSimClient",
    "OutSimPacket",
    "OutSimUDPClient",
    "TelemetryFormatError",
    "TelemetryFusion",
]
