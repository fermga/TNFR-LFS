"""Structured models for Event Performance Indicator computations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class TyresNode:
    """Metrics associated with the tyre (neumáticos) subsystem."""

    delta_nfr: float
    sense_index: float
    nu_f: float = 0.0
    dEPI_dt: float = 0.0
    integrated_epi: float = 0.0
    load: float = 0.0
    slip_ratio: float = 0.0
    mu_eff_front: float = 0.0
    mu_eff_rear: float = 0.0
    mu_eff_front_lateral: float = 0.0
    mu_eff_front_longitudinal: float = 0.0
    mu_eff_rear_lateral: float = 0.0
    mu_eff_rear_longitudinal: float = 0.0
    tyre_temp_fl: float = 0.0
    tyre_temp_fr: float = 0.0
    tyre_temp_rl: float = 0.0
    tyre_temp_rr: float = 0.0
    tyre_pressure_fl: float = 0.0
    tyre_pressure_fr: float = 0.0
    tyre_pressure_rl: float = 0.0
    tyre_pressure_rr: float = 0.0


@dataclass(frozen=True)
class SuspensionNode:
    """Metrics associated with the suspension subsystem."""

    delta_nfr: float
    sense_index: float
    nu_f: float = 0.0
    dEPI_dt: float = 0.0
    integrated_epi: float = 0.0
    travel_front: float = 0.0
    travel_rear: float = 0.0
    velocity_front: float = 0.0
    velocity_rear: float = 0.0


@dataclass(frozen=True)
class ChassisNode:
    """Metrics associated with the chassis subsystem."""

    delta_nfr: float
    sense_index: float
    nu_f: float = 0.0
    dEPI_dt: float = 0.0
    integrated_epi: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    yaw_rate: float = 0.0
    lateral_accel: float = 0.0
    longitudinal_accel: float = 0.0


@dataclass(frozen=True)
class BrakesNode:
    """Metrics associated with the brake subsystem."""

    delta_nfr: float
    sense_index: float
    nu_f: float = 0.0
    dEPI_dt: float = 0.0
    integrated_epi: float = 0.0
    brake_pressure: float = 0.0
    locking: float = 0.0


@dataclass(frozen=True)
class TransmissionNode:
    """Metrics associated with the transmission subsystem."""

    delta_nfr: float
    sense_index: float
    nu_f: float = 0.0
    dEPI_dt: float = 0.0
    integrated_epi: float = 0.0
    throttle: float = 0.0
    gear: int = 0
    speed: float = 0.0
    longitudinal_accel: float = 0.0
    rpm: float = 0.0
    line_deviation: float = 0.0


@dataclass(frozen=True)
class TrackNode:
    """Metrics associated with the track (pista) conditions."""

    delta_nfr: float
    sense_index: float
    nu_f: float = 0.0
    dEPI_dt: float = 0.0
    integrated_epi: float = 0.0
    axle_load_balance: float = 0.0
    axle_velocity_balance: float = 0.0
    yaw: float = 0.0
    lateral_accel: float = 0.0


@dataclass(frozen=True)
class DriverNode:
    """Metrics associated with the driver (piloto)."""

    delta_nfr: float
    sense_index: float
    nu_f: float = 0.0
    dEPI_dt: float = 0.0
    integrated_epi: float = 0.0
    steer: float = 0.0
    throttle: float = 0.0
    style_index: float = 0.0


@dataclass(frozen=True)
class EPIBundle:
    """Aggregated telemetry insights for a single sample."""

    timestamp: float
    epi: float
    delta_nfr: float
    sense_index: float
    tyres: TyresNode
    suspension: SuspensionNode
    chassis: ChassisNode
    brakes: BrakesNode
    transmission: TransmissionNode
    track: TrackNode
    driver: DriverNode
    structural_timestamp: float | None = None
    delta_breakdown: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    dEPI_dt: float = 0.0
    integrated_epi: float = 0.0
    node_evolution: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    delta_nfr_longitudinal: float = 0.0
    delta_nfr_lateral: float = 0.0

