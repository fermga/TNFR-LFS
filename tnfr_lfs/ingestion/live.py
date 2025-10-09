"""Interfaces for live telemetry ingestion from Live for Speed."""

from __future__ import annotations

from tnfr_lfs.ingestion.fusion import (
    FusionCalibration,
    TelemetryFusion,
    _WheelTelemetry,
)
from tnfr_lfs.ingestion.insim import (
    ButtonEvent,
    ButtonLayout,
    InSimClient,
    MacroQueue,
    OverlayManager,
)
from tnfr_lfs.ingestion.outgauge_udp import OutGaugePacket, OutGaugeUDPClient
from tnfr_lfs.ingestion.outsim_client import (
    DEFAULT_SCHEMA,
    LEGACY_COLUMNS,
    OPTIONAL_SCHEMA_COLUMNS,
    OutSimClient,
    TelemetryFormatError,
)
from tnfr_lfs.ingestion.outsim_udp import (
    OUTSIM_MAX_PACKET_SIZE,
    OutSimDriverInputs,
    OutSimPacket,
    OutSimUDPClient,
    OutSimWheelState,
)

DEFAULT_TIMEOUT = 0.05
DEFAULT_RETRIES = 5

__all__ = [
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT",
    "ButtonEvent",
    "ButtonLayout",
    "DEFAULT_SCHEMA",
    "FusionCalibration",
    "LEGACY_COLUMNS",
    "InSimClient",
    "MacroQueue",
    "OPTIONAL_SCHEMA_COLUMNS",
    "OverlayManager",
    "OutGaugePacket",
    "OutGaugeUDPClient",
    "OUTSIM_MAX_PACKET_SIZE",
    "OutSimClient",
    "OutSimDriverInputs",
    "OutSimPacket",
    "OutSimUDPClient",
    "OutSimWheelState",
    "TelemetryFormatError",
    "TelemetryFusion",
    "_WheelTelemetry",
]
