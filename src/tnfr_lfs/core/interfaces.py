"""Structural typing interfaces for telemetry and contextual helpers.

This module groups :class:`typing.Protocol` definitions that describe the
attributes consumed by the analytical helpers in :mod:`tnfr_lfs.core`.  The
protocols enable structural typing across the codebase so that third party
implementations can interoperate with the analytics layer without relying on
the concrete dataclasses used by the default ingestion pipeline.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, Tuple, runtime_checkable

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
class SupportsTyresNode(SupportsEPINode, Protocol):
    """Tyre subsystem payload consumed by analytics and operators."""

    load: float
    slip_ratio: float
    mu_eff_front: float
    mu_eff_rear: float
    mu_eff_front_lateral: float
    mu_eff_front_longitudinal: float
    mu_eff_rear_lateral: float
    mu_eff_rear_longitudinal: float
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


@runtime_checkable
class SupportsSuspensionNode(SupportsEPINode, Protocol):
    """Suspension subsystem payload consumed by analytics and operators."""

    travel_front: float
    travel_rear: float
    velocity_front: float
    velocity_rear: float


@runtime_checkable
class SupportsChassisNode(SupportsEPINode, Protocol):
    """Chassis subsystem payload consumed by analytics and operators."""

    yaw: float
    pitch: float
    roll: float
    yaw_rate: float
    lateral_accel: float
    longitudinal_accel: float


@runtime_checkable
class SupportsBrakesNode(SupportsEPINode, Protocol):
    """Brake subsystem payload consumed by analytics and operators."""

    brake_pressure: float
    locking: float
    brake_temp_fl: float
    brake_temp_fr: float
    brake_temp_rl: float
    brake_temp_rr: float
    brake_temp_peak: float
    brake_temp_mean: float


@runtime_checkable
class SupportsTransmissionNode(SupportsEPINode, Protocol):
    """Transmission subsystem payload consumed by analytics and operators."""

    throttle: float
    gear: int
    speed: float
    longitudinal_accel: float
    rpm: float
    line_deviation: float


@runtime_checkable
class SupportsTrackNode(SupportsEPINode, Protocol):
    """Track condition payload consumed by analytics and operators."""

    axle_load_balance: float
    axle_velocity_balance: float
    yaw: float
    lateral_accel: float
    gradient: float


@runtime_checkable
class SupportsDriverNode(SupportsEPINode, Protocol):
    """Driver payload consumed by analytics and operators."""

    steer: float
    throttle: float
    style_index: float


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

    tyres: SupportsTyresNode
    suspension: SupportsSuspensionNode
    chassis: SupportsChassisNode
    brakes: SupportsBrakesNode
    transmission: SupportsTransmissionNode
    track: SupportsTrackNode
    driver: SupportsDriverNode


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


@runtime_checkable
class SupportsGoal(Protocol):
    """Goal specification produced by the segmentation heuristics."""

    phase: str
    archetype: str
    description: str
    target_delta_nfr: float
    target_sense_index: float
    nu_f_target: float
    nu_exc_target: float
    rho_target: float
    target_phase_lag: float
    target_phase_alignment: float
    measured_phase_lag: float
    measured_phase_alignment: float
    target_phase_synchrony: float
    measured_phase_synchrony: float
    slip_lat_window: Tuple[float, float]
    slip_long_window: Tuple[float, float]
    yaw_rate_window: Tuple[float, float]
    dominant_nodes: Tuple[str, ...]
    target_delta_nfr_long: float
    target_delta_nfr_lat: float
    delta_axis_weights: Mapping[str, float]
    archetype_delta_nfr_long_target: float
    archetype_delta_nfr_lat_target: float
    archetype_nu_f_target: float
    archetype_si_phi_target: float
    detune_ratio_weights: Mapping[str, float]
    track_gradient: float


@runtime_checkable
class SupportsMicrosector(Protocol):
    """Microsector abstraction consumed by operator orchestration."""

    index: int
    start_time: float
    end_time: float
    curvature: float
    brake_event: bool
    support_event: bool
    delta_nfr_signature: float
    goals: Sequence[SupportsGoal]
    phase_boundaries: Mapping[str, Tuple[int, int]]
    phase_samples: Mapping[str, Tuple[int, ...]]
    active_phase: str
    dominant_nodes: Mapping[str, Tuple[str, ...]]
    phase_weights: Mapping[str, Mapping[str, float] | float]
    grip_rel: float
    phase_lag: Mapping[str, float]
    phase_alignment: Mapping[str, float]
    phase_synchrony: Mapping[str, float]
    phase_motor_latency: Mapping[str, float]
    motor_latency_ms: float
    filtered_measures: Mapping[str, object]
    recursivity_trace: Sequence[Mapping[str, float | str | None]]
    last_mutation: Mapping[str, object] | None
    window_occupancy: Mapping[str, Mapping[str, float]]
    delta_nfr_std: float
    nodal_delta_nfr_std: float
    phase_delta_nfr_std: Mapping[str, float]
    phase_nodal_delta_nfr_std: Mapping[str, float]
    delta_nfr_entropy: float
    node_entropy: float
    phase_delta_nfr_entropy: Mapping[str, float]
    phase_node_entropy: Mapping[str, float]
    phase_axis_targets: Mapping[str, Mapping[str, float]]
    phase_axis_weights: Mapping[str, Mapping[str, float]]
    context_factors: Mapping[str, float]
    sample_context_factors: Mapping[int, Mapping[str, float]]
    operator_events: Mapping[str, Sequence[Mapping[str, object]]]

    def phase_indices(self, phase: str) -> range:
        """Return the range of telemetry samples associated with ``phase``."""

