"""Archetype metadata shared across segmentation and recommendation layers."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


ARCHETYPE_HAIRPIN = "hairpin"
ARCHETYPE_MEDIUM = "medium"
ARCHETYPE_FAST = "fast"
ARCHETYPE_CHICANE = "chicane"
ARCHETYPE_DEFAULT = ARCHETYPE_MEDIUM


@dataclass(frozen=True)
class PhaseArchetypeTargets:
    """Targets describing the tactical intent of an archetype per phase family."""

    delta_nfr_long: float
    delta_nfr_lat: float
    nu_f: float
    si_phi: float
    lag: float
    detune_weights: Mapping[str, float]

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        mapping = {str(axis): float(value) for axis, value in self.detune_weights.items()}
        object.__setattr__(self, "detune_weights", MappingProxyType(mapping))


def _targets(
    delta_long: float,
    delta_lat: float,
    nu_f: float,
    si_phi: float,
    lag: float,
    detune_long: float,
    detune_lat: float,
) -> PhaseArchetypeTargets:
    total = abs(detune_long) + abs(detune_lat)
    if total <= 0.0:
        weights = {"longitudinal": 0.5, "lateral": 0.5}
    else:
        weights = {
            "longitudinal": detune_long / total,
            "lateral": detune_lat / total,
        }
    return PhaseArchetypeTargets(
        delta_nfr_long=float(delta_long),
        delta_nfr_lat=float(delta_lat),
        nu_f=float(nu_f),
        si_phi=float(si_phi),
        lag=float(lag),
        detune_weights=weights,
    )


DEFAULT_ARCHETYPE_PHASE_TARGETS: Mapping[str, Mapping[str, PhaseArchetypeTargets]] = {
    ARCHETYPE_HAIRPIN: MappingProxyType(
        {
            "entry": _targets(-0.42, 0.18, 0.42, 0.86, -0.12, 0.7, 0.3),
            "apex": _targets(-0.06, 0.48, 0.55, 0.9, -0.02, 0.45, 0.55),
            "exit": _targets(0.22, 0.26, 0.36, 0.88, 0.08, 0.6, 0.4),
        }
    ),
    ARCHETYPE_MEDIUM: MappingProxyType(
        {
            "entry": _targets(-0.28, 0.22, 0.48, 0.9, -0.08, 0.6, 0.4),
            "apex": _targets(-0.04, 0.36, 0.5, 0.92, -0.01, 0.45, 0.55),
            "exit": _targets(0.18, 0.24, 0.4, 0.88, 0.05, 0.55, 0.45),
        }
    ),
    ARCHETYPE_FAST: MappingProxyType(
        {
            "entry": _targets(-0.18, 0.14, 0.62, 0.94, -0.04, 0.56, 0.44),
            "apex": _targets(-0.02, 0.3, 0.64, 0.96, 0.0, 0.48, 0.52),
            "exit": _targets(0.14, 0.2, 0.58, 0.93, 0.03, 0.52, 0.48),
        }
    ),
    ARCHETYPE_CHICANE: MappingProxyType(
        {
            "entry": _targets(-0.24, 0.26, 0.46, 0.88, -0.1, 0.58, 0.42),
            "apex": _targets(-0.02, 0.42, 0.52, 0.9, 0.0, 0.44, 0.56),
            "exit": _targets(0.16, 0.3, 0.44, 0.86, 0.04, 0.5, 0.5),
        }
    ),
}


def archetype_phase_targets(
    archetype: str,
) -> Mapping[str, PhaseArchetypeTargets]:
    """Return the phase targets for ``archetype`` (falls back to medium)."""

    table = DEFAULT_ARCHETYPE_PHASE_TARGETS.get(archetype)
    if table is None:
        table = DEFAULT_ARCHETYPE_PHASE_TARGETS[ARCHETYPE_DEFAULT]
    return table


__all__ = [
    "ARCHETYPE_CHICANE",
    "ARCHETYPE_DEFAULT",
    "ARCHETYPE_FAST",
    "ARCHETYPE_HAIRPIN",
    "ARCHETYPE_MEDIUM",
    "PhaseArchetypeTargets",
    "DEFAULT_ARCHETYPE_PHASE_TARGETS",
    "archetype_phase_targets",
]

