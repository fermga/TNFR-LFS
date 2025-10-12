"""Structural typing interfaces for contextual helpers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = [
    "SupportsContextRecord",
    "SupportsContextBundle",
    "SupportsContextTyres",
    "SupportsContextChassis",
    "SupportsContextTransmission",
]


@runtime_checkable
class SupportsContextRecord(Protocol):
    """Telemetry-like payload exposing the fields required for context factors."""

    lateral_accel: float
    vertical_load: float
    longitudinal_accel: float


@runtime_checkable
class SupportsContextTyres(Protocol):
    """Tyre subsystem metrics required to derive contextual surface ratios."""

    load: float


@runtime_checkable
class SupportsContextChassis(Protocol):
    """Chassis subsystem metrics required to derive curvature and traffic cues."""

    lateral_accel: float
    longitudinal_accel: float


@runtime_checkable
class SupportsContextTransmission(Protocol):
    """Transmission subsystem metrics used as a fallback for traffic cues."""

    longitudinal_accel: float


@runtime_checkable
class SupportsContextBundle(Protocol):
    """Aggregate bundle exposing the nodes required by contextual helpers."""

    tyres: SupportsContextTyres | None
    chassis: SupportsContextChassis | None
    transmission: SupportsContextTransmission | None
