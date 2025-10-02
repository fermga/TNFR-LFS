"""Rule-based recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol, Sequence

from ..core.epi import EPIResult, compute_coherence


@dataclass
class Recommendation:
    """Represents an actionable recommendation."""

    category: str
    message: str
    rationale: str


class RecommendationRule(Protocol):
    """Interface implemented by recommendation rules."""

    def evaluate(self, results: Sequence[EPIResult]) -> Iterable[Recommendation]:
        ...


class LoadBalanceRule:
    """Suggests changes when ΔNFR deviates from the baseline."""

    threshold: float = 10.0

    def evaluate(self, results: Sequence[EPIResult]) -> Iterable[Recommendation]:
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
    """Issue recommendations when stability index is unstable."""

    threshold: float = 0.08

    def evaluate(self, results: Sequence[EPIResult]) -> Iterable[Recommendation]:
        for result in results:
            if abs(result.delta_si) > self.threshold:
                direction = "increase" if result.delta_si < 0 else "reduce"
                yield Recommendation(
                    category="aero",
                    message=f"{direction.title()} Front Wing angle to stabilise turn-in",
                    rationale=(
                        "Stability index deviated by "
                        f"{result.delta_si:.3f} relative to baseline at t={result.timestamp:.2f}."
                    ),
                )


class CoherenceRule:
    """High-level rule that considers the coherence score across a stint."""

    min_coherence: float = 0.85

    def evaluate(self, results: Sequence[EPIResult]) -> Iterable[Recommendation]:
        coherence = compute_coherence(results)
        if coherence < self.min_coherence:
            yield Recommendation(
                category="driver",
                message="Review driving inputs for consistency",
                rationale=(
                    "Coherence score across the analysed stint is "
                    f"{coherence:.2f}, below the expected threshold of {self.min_coherence:.2f}."
                ),
            )


class RecommendationEngine:
    """Aggregate a list of rules and produce recommendations."""

    def __init__(self, rules: Sequence[RecommendationRule] | None = None) -> None:
        self.rules: List[RecommendationRule] = list(rules) if rules else [
            LoadBalanceRule(),
            StabilityIndexRule(),
            CoherenceRule(),
        ]

    def generate(self, results: Sequence[EPIResult]) -> List[Recommendation]:
        recommendations: List[Recommendation] = []
        for rule in self.rules:
            recommendations.extend(rule.evaluate(results))
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
