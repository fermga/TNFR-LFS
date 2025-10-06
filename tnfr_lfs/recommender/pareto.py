"""Pareto utilities to analyse multi-objective optimisation sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """Container describing the outcome of a candidate evaluation."""

    decision_vector: Mapping[str, float]
    score: float
    breakdown: Mapping[str, float]
    objective_breakdown: Mapping[str, float]
    telemetry: Sequence[object] = ()

    def as_dict(self) -> Mapping[str, object]:
        """Return a serialisable representation of the point."""

        return MappingProxyType(
            {
                "decision_vector": dict(self.decision_vector),
                "score": float(self.score),
                "breakdown": dict(self.breakdown),
                "objective_breakdown": dict(self.objective_breakdown),
            }
        )


def _dominates(candidate: ParetoPoint, other: ParetoPoint) -> bool:
    """Return ``True`` if ``candidate`` dominates ``other``."""

    if candidate is other:
        return False
    dominated = False
    candidate_keys = set(candidate.breakdown)
    other_keys = set(other.breakdown)
    all_keys = candidate_keys | other_keys
    for key in all_keys:
        candidate_value = float(candidate.breakdown.get(key, float("inf")))
        other_value = float(other.breakdown.get(key, float("inf")))
        if candidate_value > other_value:
            return False
        if candidate_value < other_value:
            dominated = True
    return dominated


def pareto_front(points: Iterable[ParetoPoint]) -> list[ParetoPoint]:
    """Filter ``points`` keeping only Pareto optimal entries."""

    candidates = list(points)
    front: list[ParetoPoint] = []
    for candidate in candidates:
        if any(_dominates(other, candidate) for other in candidates):
            continue
        front.append(candidate)
    front.sort(key=lambda entry: entry.score, reverse=True)
    return front


__all__ = ["ParetoPoint", "pareto_front"]

