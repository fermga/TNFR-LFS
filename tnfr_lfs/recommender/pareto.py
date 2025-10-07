"""Pareto utilities to analyse multi-objective optimisation sweeps.

This module considers any non-finite metric value (``NaN``/``±inf``) as a
dominated outcome. Values are normalised during comparisons so that
invalid points are always outranked by finite counterparts.
"""

from __future__ import annotations

import math
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


def _normalise_metric(value: float) -> float:
    """Return ``value`` if it is finite, ``inf`` otherwise.

    Using ``inf`` for invalid values guarantees the candidate holding the
    metric will be considered worse in the domination check, preventing
    NaN/±inf points from appearing in the Pareto front.
    """

    return value if math.isfinite(value) else float("inf")


def _has_invalid_metrics(point: ParetoPoint) -> bool:
    """Return ``True`` when ``point`` contains non-finite metrics."""

    return any(not math.isfinite(float(value)) for value in point.breakdown.values())


def _dominates(candidate: ParetoPoint, other: ParetoPoint) -> bool:
    """Return ``True`` if ``candidate`` dominates ``other``.

    Each metric is normalised with :func:`math.isfinite` so that
    ``NaN``/``±inf`` values behave like ``inf`` and therefore break
    domination for the owning candidate.
    """

    if candidate is other:
        return False
    dominated = False
    candidate_keys = set(candidate.breakdown)
    other_keys = set(other.breakdown)
    all_keys = candidate_keys | other_keys
    for key in all_keys:
        candidate_value = _normalise_metric(
            float(candidate.breakdown.get(key, float("inf")))
        )
        other_value = _normalise_metric(float(other.breakdown.get(key, float("inf"))))
        if candidate_value > other_value:
            return False
        if candidate_value < other_value:
            dominated = True
    return dominated


def pareto_front(points: Iterable[ParetoPoint]) -> list[ParetoPoint]:
    """Filter ``points`` keeping only Pareto optimal entries.

    Candidates containing non-finite metrics are discarded before the
    dominance check to ensure the resulting front only contains valid
    evaluations.
    """

    candidates = [point for point in points if not _has_invalid_metrics(point)]
    front: list[ParetoPoint] = []
    for candidate in candidates:
        if any(_dominates(other, candidate) for other in candidates):
            continue
        front.append(candidate)
    front.sort(key=lambda entry: entry.score, reverse=True)
    return front


__all__ = ["ParetoPoint", "pareto_front"]

