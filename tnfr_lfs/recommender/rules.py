"""Rule-based recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, List, Protocol, Sequence

from ..core.epi_models import EPIBundle
from ..core.segmentation import Microsector


@dataclass
class Recommendation:
    """Represents an actionable recommendation."""

    category: str
    message: str
    rationale: str


class RecommendationRule(Protocol):
    """Interface implemented by recommendation rules."""

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
    ) -> Iterable[Recommendation]:
        ...


class LoadBalanceRule:
    """Suggests changes when ΔNFR deviates from the baseline."""

    threshold: float = 10.0

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
    ) -> Iterable[Recommendation]:
        for result in results:
            if abs(result.delta_nfr) > self.threshold:
                direction = "increase" if result.delta_nfr < 0 else "decrease"
                yield Recommendation(
                    category="suspension",
                    message=f"{direction.title()} Rear Ride height to rebalance load",
                    rationale=(
                        "ΔNFR deviated by "
                        f"{result.delta_nfr:.1f} units relative to baseline at t={result.timestamp:.2f}."
                    ),
                )


class StabilityIndexRule:
    """Issue recommendations when the sense index degrades."""

    threshold: float = 0.6

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
    ) -> Iterable[Recommendation]:
        for result in results:
            if result.sense_index < self.threshold:
                yield Recommendation(
                    category="aero",
                    message="Stabilise aero balance to recover sense index",
                    rationale=(
                        "Sense index dropped to "
                        f"{result.sense_index:.2f} at t={result.timestamp:.2f}, below the threshold of "
                        f"{self.threshold:.2f}."
                    ),
                )


class CoherenceRule:
    """High-level rule that considers the average sense index across a stint."""

    min_average_si: float = 0.75

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
    ) -> Iterable[Recommendation]:
        if not results:
            return []
        average_si = mean(result.sense_index for result in results)
        if average_si < self.min_average_si:
            return [
                Recommendation(
                    category="driver",
                    message="Review driving inputs for consistency",
                    rationale=(
                        "Average sense index across the analysed stint is "
                        f"{average_si:.2f}, below the expected threshold of {self.min_average_si:.2f}."
                    ),
                )
            ]
        return []


class RecommendationEngine:
    """Aggregate a list of rules and produce recommendations."""

    def __init__(self, rules: Sequence[RecommendationRule] | None = None) -> None:
        self.rules: List[RecommendationRule] = list(rules) if rules else [
            LoadBalanceRule(),
            StabilityIndexRule(),
            CoherenceRule(),
        ]

    def generate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
    ) -> List[Recommendation]:
        recommendations: List[Recommendation] = []
        for rule in self.rules:
            recommendations.extend(list(rule.evaluate(results, microsectors)))
        # Deduplicate identical recommendations while preserving order.
        unique: List[Recommendation] = []
        seen = set()
        for recommendation in recommendations:
            key = (recommendation.category, recommendation.message)
            if key in seen:
                continue
            seen.add(key)
            unique.append(recommendation)
        return unique
