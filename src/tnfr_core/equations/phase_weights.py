"""Phase weighting helpers shared across EPI computations."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Mapping

from tnfr_core.equations.phases import expand_phase_alias, phase_family

__all__ = ["_phase_weight"]


def _phase_weight(
    node: str,
    phase: str | None,
    phase_weights: Mapping[str, Mapping[str, float] | float] | None,
) -> float:
    """Resolve the weight modifier for a node under the given phase profile."""

    if not phase or not phase_weights or not isinstance(phase_weights, MappingABC):
        return 1.0
    profile: Mapping[str, float] | float | None = None
    for candidate in (*expand_phase_alias(phase), phase_family(phase), phase):
        if candidate is None:
            continue
        profile = phase_weights.get(candidate)
        if profile is not None:
            break
    if profile is None:
        profile = phase_weights.get("__default__")
    if profile is None:
        return 1.0
    if isinstance(profile, MappingABC):
        if node in profile:
            return float(profile[node])
        if "__default__" in profile:
            return float(profile["__default__"])
        return 1.0
    if isinstance(profile, (int, float)):
        return float(profile)
    return 1.0
