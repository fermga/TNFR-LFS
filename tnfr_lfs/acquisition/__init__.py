"""Telemetry acquisition backends (deprecated; use :mod:`tnfr_lfs.ingestion.live`)."""

from __future__ import annotations

import warnings
from typing import Any

__all__ = [
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT",
    "ButtonEvent",
    "ButtonLayout",
    "FusionCalibration",
    "InSimClient",
    "MacroQueue",
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

_WARNED_NAMES: set[str] = set()


def __getattr__(name: str) -> Any:
    from tnfr_lfs.ingestion import live as _live

    if hasattr(_live, name):
        if name not in _WARNED_NAMES:
            warnings.warn(
                "'tnfr_lfs.acquisition' is deprecated and will be removed in a future release; "
                "import from 'tnfr_lfs.ingestion.live' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            _WARNED_NAMES.add(name)
        value = getattr(_live, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'tnfr_lfs.acquisition' has no attribute {name!r}")


def __dir__() -> list[str]:
    from tnfr_lfs.ingestion import live as _live

    return sorted(set(__all__) | set(dir(_live)))
