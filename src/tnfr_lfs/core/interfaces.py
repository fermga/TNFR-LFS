"""Structural typing interfaces for telemetry and contextual helpers.

This module groups :class:`typing.Protocol` definitions that describe the
attributes consumed by the analytical helpers in :mod:`tnfr_lfs.core`.  The
protocols enable structural typing across the codebase so that third party
implementations can interoperate with the analytics layer without relying on
the concrete dataclasses used by the default ingestion pipeline.
"""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

__all__ = [
    "SupportsTelemetrySample",
    "SupportsEPINode",
    "SupportsEPIBundle",
    "SupportsContextRecord",
    "SupportsContextBundle",
    "SupportsContextTyres",
    "SupportsContextChassis",
    "SupportsContextTransmission",
]


@runtime_checkable
class SupportsTelemetrySample(Protocol):
    """Telemetry payload exposing the fields required by core analytics."""

    timestamp: float
    structural_timestamp: float | None

    lateral_accel: float
    longitudinal_accel: float
    vertical_load: float
    vertical_load_front: float
    vertical_load_rear: float

    speed: float
    yaw: float
    pitch: float
    roll: float
    yaw_rate: float
    steer: float

    suspension_velocity_front: float
    suspension_velocity_rear: float
    suspension_travel_front: float
    suspension_travel_rear: float

    brake_pressure: float
    locking: float
    throttle: float
    gear: int

    nfr: float
    si: float
    slip_ratio: float
    slip_angle: float

    mu_eff_front: float
    mu_eff_rear: float
    mu_eff_front_lateral: float
    mu_eff_front_longitudinal: float
    mu_eff_rear_lateral: float
    mu_eff_rear_longitudinal: float

    slip_ratio_fl: float
    slip_ratio_fr: float
    slip_ratio_rl: float
    slip_ratio_rr: float

    slip_angle_fl: float
    slip_angle_fr: float
    slip_angle_rl: float
    slip_angle_rr: float

    wheel_load_fl: float
    wheel_load_fr: float
    wheel_load_rl: float
    wheel_load_rr: float

    wheel_lateral_force_fl: float
    wheel_lateral_force_fr: float
    wheel_lateral_force_rl: float
    wheel_lateral_force_rr: float

    wheel_longitudinal_force_fl: float
    wheel_longitudinal_force_fr: float
    wheel_longitudinal_force_rl: float
    wheel_longitudinal_force_rr: float

    tyre_temp_fl: float
    tyre_temp_fr: float
    tyre_temp_rl: float
    tyre_temp_rr: float
    tyre_temp_fl_inner: float
    tyre_temp_fr_inner: float
    tyre_temp_rl_inner: float
    tyre_temp_rr_inner: float
    tyre_temp_fl_middle: float
    tyre_temp_fr_middle: float
    tyre_temp_rl_middle: float
    tyre_temp_rr_middle: float
    tyre_temp_fl_outer: float
    tyre_temp_fr_outer: float
    tyre_temp_rl_outer: float
    tyre_temp_rr_outer: float

    tyre_pressure_fl: float
    tyre_pressure_fr: float
    tyre_pressure_rl: float
    tyre_pressure_rr: float

    brake_temp_fl: float
    brake_temp_fr: float
    brake_temp_rl: float
    brake_temp_rr: float

    rpm: float
    line_deviation: float

    reference: "SupportsTelemetrySample" | None


@runtime_checkable
class SupportsEPINode(Protocol):
    """Subsystem node payload used by EPI analytics."""

    delta_nfr: float
    sense_index: float
    nu_f: float
    dEPI_dt: float
    integrated_epi: float


@runtime_checkable
class SupportsEPIBundle(Protocol):
    """Aggregated telemetry insights required by EPI consumers."""

    timestamp: float
    epi: float
    delta_nfr: float
    sense_index: float
    delta_breakdown: Mapping[str, Mapping[str, float]]
    node_evolution: Mapping[str, tuple[float, float]]
    structural_timestamp: float | None
    dEPI_dt: float
    integrated_epi: float
    delta_nfr_proj_longitudinal: float
    delta_nfr_proj_lateral: float
    nu_f_classification: str
    nu_f_category: str
    nu_f_label: str
    nu_f_dominant: float
    coherence_index: float
    ackermann_parallel_index: float

    tyres: SupportsEPINode
    suspension: SupportsEPINode
    chassis: SupportsEPINode
    brakes: SupportsEPINode
    transmission: SupportsEPINode
    track: SupportsEPINode
    driver: SupportsEPINode


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
