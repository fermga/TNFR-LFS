"""Structured models for Event Performance Indicator computations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TyresNode:
    """Metrics associated with the tyre (neumáticos) subsystem."""

    delta_nfr: float
    sense_index: float
    nu_f: float = 0.0


@dataclass(frozen=True)
class SuspensionNode:
    """Metrics associated with the suspension subsystem."""

    delta_nfr: float
    sense_index: float
    nu_f: float = 0.0


@dataclass(frozen=True)
class ChassisNode:
    """Metrics associated with the chassis subsystem."""

    delta_nfr: float
    sense_index: float
    nu_f: float = 0.0


@dataclass(frozen=True)
class BrakesNode:
    """Metrics associated with the brake subsystem."""

    delta_nfr: float
    sense_index: float
    nu_f: float = 0.0


@dataclass(frozen=True)
class TransmissionNode:
    """Metrics associated with the transmission subsystem."""

    delta_nfr: float
    sense_index: float
    nu_f: float = 0.0


@dataclass(frozen=True)
class TrackNode:
    """Metrics associated with the track (pista) conditions."""

    delta_nfr: float
    sense_index: float
    nu_f: float = 0.0


@dataclass(frozen=True)
class DriverNode:
    """Metrics associated with the driver (piloto)."""

    delta_nfr: float
    sense_index: float
    nu_f: float = 0.0


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
    dEPI_dt: float = 0.0
    integrated_epi: float = 0.0

