"""Metadata helpers for structural EPI evolution."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Mapping

__all__ = [
    "PhaseContext",
    "extract_phase_context",
    "resolve_nu_targets",
    "NON_NODAL_METADATA_KEYS",
]


NON_NODAL_METADATA_KEYS = frozenset(
    {
        "__theta__",
        "__w_phase__",
        "nu_f_objectives",
        "__nu_f__",
        "nu_f_targets",
    }
)


@dataclass(frozen=True)
class PhaseContext:
    """Container describing the active EPI phase metadata."""

    identifier: Any | None
    weights: Mapping[str, Mapping[str, float] | float] | None


def _extract_metadata_value(delta_map: Any, keys: Sequence[str]) -> Any:
    """Return the first non-``None`` metadata value matched by ``keys``."""

    for key in keys:
        if hasattr(delta_map, key):
            value = getattr(delta_map, key)
            if value is not None:
                return value
    if isinstance(delta_map, MappingABC):
        for key in keys:
            if key in delta_map:
                value = delta_map[key]
                if value is not None:
                    return value
    return None


def extract_phase_context(delta_map: Any) -> PhaseContext:
    """Extract phase identifier and weights from ``delta_map`` metadata."""

    phase_identifier = _extract_metadata_value(delta_map, ("__theta__", "theta"))
    raw_phase_weights = _extract_metadata_value(
        delta_map, ("__w_phase__", "phase_weights")
    )
    phase_weights: Mapping[str, Mapping[str, float] | float] | None
    if isinstance(raw_phase_weights, MappingABC):
        phase_weights = raw_phase_weights  # type: ignore[assignment]
    else:
        phase_weights = None
    return PhaseContext(identifier=phase_identifier, weights=phase_weights)


def resolve_nu_targets(delta_map: Any) -> Dict[str, float] | None:
    """Resolve explicit ν_f objectives from ``delta_map`` metadata."""

    raw_nu_targets = _extract_metadata_value(
        delta_map, ("nu_f_objectives", "__nu_f__", "nu_f_targets")
    )
    if not isinstance(raw_nu_targets, MappingABC):
        return None

    targets: Dict[str, float] = {}
    for key, value in raw_nu_targets.items():
        try:
            targets[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return targets
