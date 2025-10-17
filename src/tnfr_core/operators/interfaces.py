"""Compatibility shim for structural typing protocols."""

from __future__ import annotations

import warnings

from tnfr_core.runtime.shared import (
    SupportsBrakesNode,
    SupportsChassisNode,
    SupportsContextBundle,
    SupportsContextChassis,
    SupportsContextRecord,
    SupportsContextTransmission,
    SupportsContextTyres,
    SupportsDriverNode,
    SupportsEPIBundle,
    SupportsEPINode,
    SupportsGoal,
    SupportsMicrosector,
    SupportsSuspensionNode,
    SupportsTelemetrySample,
    SupportsTrackNode,
    SupportsTransmissionNode,
    SupportsTyresNode,
)

warnings.warn(
    (
        "'tnfr_core.operators.interfaces' is deprecated and will be removed in a "
        "future release; import from 'tnfr_core.runtime.shared' instead."
    ),
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "SupportsTelemetrySample",
    "SupportsEPINode",
    "SupportsTyresNode",
    "SupportsSuspensionNode",
    "SupportsChassisNode",
    "SupportsBrakesNode",
    "SupportsTransmissionNode",
    "SupportsTrackNode",
    "SupportsDriverNode",
    "SupportsEPIBundle",
    "SupportsContextRecord",
    "SupportsContextBundle",
    "SupportsContextTyres",
    "SupportsContextChassis",
    "SupportsContextTransmission",
    "SupportsGoal",
    "SupportsMicrosector",
]
