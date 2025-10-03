"""Rule-based recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, List, Mapping, Protocol, Sequence

from ..core.epi_models import EPIBundle
from ..core.segmentation import Goal, Microsector


@dataclass
class Recommendation:
    """Represents an actionable recommendation."""

    category: str
    message: str
    rationale: str
    priority: int = 0


@dataclass(frozen=True)
class ThresholdProfile:
    """Thresholds tuned per car model and circuit."""

    entry_delta_tolerance: float
    apex_delta_tolerance: float
    exit_delta_tolerance: float
    piano_delta_tolerance: float

    def tolerance_for_phase(self, phase: str) -> float:
        mapping = {
            "entry": self.entry_delta_tolerance,
            "apex": self.apex_delta_tolerance,
            "exit": self.exit_delta_tolerance,
        }
        return mapping[phase]


@dataclass(frozen=True)
class RuleContext:
    """Context shared with the rules to build rationales."""

    car_model: str
    track_name: str
    thresholds: ThresholdProfile

    @property
    def profile_label(self) -> str:
        return f"{self.car_model}/{self.track_name}"


class RecommendationRule(Protocol):
    """Interface implemented by recommendation rules."""

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        ...


class LoadBalanceRule:
    """Suggests changes when ΔNFR deviates from the baseline."""

    threshold: float = 10.0

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
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
                    priority=100,
                )


class StabilityIndexRule:
    """Issue recommendations when the sense index degrades."""

    threshold: float = 0.6

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
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
                    priority=110,
                )


class CoherenceRule:
    """High-level rule that considers the average sense index across a stint."""

    min_average_si: float = 0.75

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
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
                    priority=120,
                )
            ]
        return []


DEFAULT_THRESHOLD_PROFILE = ThresholdProfile(
    entry_delta_tolerance=1.5,
    apex_delta_tolerance=1.0,
    exit_delta_tolerance=2.0,
    piano_delta_tolerance=2.5,
)


DEFAULT_THRESHOLD_LIBRARY: Mapping[str, Mapping[str, ThresholdProfile]] = {
    "generic": {"generic": DEFAULT_THRESHOLD_PROFILE}
}


def _goal_for_phase(microsector: Microsector, phase: str) -> Goal | None:
    for goal in microsector.goals:
        if goal.phase == phase:
            return goal
    return None


def _phase_samples(results: Sequence[EPIBundle], indices: range) -> List[EPIBundle]:
    return [results[i] for i in indices if 0 <= i < len(results)]


class PhaseDeltaDeviationRule:
    """Detects ΔNFR mismatches for a given phase of the corner."""

    def __init__(
        self,
        phase: str,
        operator_label: str,
        category: str,
        phase_label: str,
        priority: int,
    ) -> None:
        self.phase = phase
        self.operator_label = operator_label
        self.category = category
        self.phase_label = phase_label
        self.priority = priority

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors or context is None:
            return []

        tolerance = context.thresholds.tolerance_for_phase(self.phase)
        recommendations: List[Recommendation] = []
        for microsector in microsectors:
            goal = _goal_for_phase(microsector, self.phase)
            if goal is None:
                continue
            samples = _phase_samples(results, microsector.phase_indices(self.phase))
            if not samples:
                continue
            actual_delta = mean(bundle.delta_nfr for bundle in samples)
            deviation = actual_delta - goal.target_delta_nfr
            if abs(deviation) <= tolerance:
                continue
            action = "incrementar" if deviation < 0 else "reducir"
            recommendations.append(
                Recommendation(
                    category=self.category,
                    message=(
                        f"{self.operator_label}: {action} ΔNFR en microsector {microsector.index}"
                    ),
                    rationale=(
                        f"El objetivo ΔNFR para la {self.phase_label} era {goal.target_delta_nfr:.2f}, "
                        f"pero la media registrada fue {actual_delta:.2f} ({deviation:+.2f}). "
                        f"La tolerancia definida para {context.profile_label} es ±{tolerance:.2f}."
                    ),
                    priority=self.priority,
                )
            )
        return recommendations


class CurbComplianceRule:
    """Analyses support events (pianos) against the ΔNFR target."""

    def __init__(self, priority: int) -> None:
        self.priority = priority

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors or context is None:
            return []

        tolerance = context.thresholds.piano_delta_tolerance
        recommendations: List[Recommendation] = []
        for microsector in microsectors:
            if not microsector.support_event:
                continue
            goal = _goal_for_phase(microsector, "apex")
            if goal is None:
                continue
            indices = microsector.phase_indices("apex")
            tyre_samples = [
                results[i].tyres.delta_nfr
                for i in indices
                if 0 <= i < len(results)
            ]
            if not tyre_samples:
                continue
            actual_delta = mean(tyre_samples)
            deviation = actual_delta - goal.target_delta_nfr
            if abs(deviation) <= tolerance:
                continue
            action = "liberar" if deviation > 0 else "cargar"
            recommendations.append(
                Recommendation(
                    category="pianos",
                    message=(
                        f"Operador de pianos: {action} apoyo en microsector {microsector.index}"
                    ),
                    rationale=(
                        "Los pianos registraron un ΔNFR medio de "
                        f"{actual_delta:.2f} frente al objetivo {goal.target_delta_nfr:.2f}. "
                        f"El desvío {deviation:+.2f} supera la tolerancia ±{tolerance:.2f} "
                        f"configurada para {context.profile_label}."
                    ),
                    priority=self.priority,
                )
            )
        return recommendations


class RecommendationEngine:
    """Aggregate a list of rules and produce recommendations."""

    def __init__(
        self,
        rules: Sequence[RecommendationRule] | None = None,
        *,
        car_model: str | None = None,
        track_name: str | None = None,
        threshold_library: Mapping[str, Mapping[str, ThresholdProfile]] | None = None,
    ) -> None:
        self.car_model = car_model or "generic"
        self.track_name = track_name or "generic"
        self.threshold_library = threshold_library or DEFAULT_THRESHOLD_LIBRARY
        if rules is None:
            self.rules = [
                PhaseDeltaDeviationRule(
                    phase="entry",
                    operator_label="Operador de frenado",
                    category="entry",
                    phase_label="entrada",
                    priority=10,
                ),
                PhaseDeltaDeviationRule(
                    phase="apex",
                    operator_label="Operador de vértice",
                    category="apex",
                    phase_label="vértice",
                    priority=20,
                ),
                CurbComplianceRule(priority=25),
                PhaseDeltaDeviationRule(
                    phase="exit",
                    operator_label="Operador de tracción",
                    category="exit",
                    phase_label="salida",
                    priority=30,
                ),
                LoadBalanceRule(),
                StabilityIndexRule(),
                CoherenceRule(),
            ]
        else:
            self.rules = list(rules)

    def _lookup_profile(self, car_model: str, track_name: str) -> ThresholdProfile:
        library = self.threshold_library
        for car_key in (car_model, "generic"):
            car_profiles = library.get(car_key)
            if not car_profiles:
                continue
            for track_key in (track_name, "generic"):
                profile = car_profiles.get(track_key)
                if profile is not None:
                    return profile
        return DEFAULT_THRESHOLD_PROFILE

    def _resolve_context(
        self,
        car_model: str | None,
        track_name: str | None,
    ) -> RuleContext:
        resolved_car = car_model or self.car_model
        resolved_track = track_name or self.track_name
        profile = self._lookup_profile(resolved_car, resolved_track)
        return RuleContext(
            car_model=resolved_car,
            track_name=resolved_track,
            thresholds=profile,
        )

    def generate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        *,
        car_model: str | None = None,
        track_name: str | None = None,
    ) -> List[Recommendation]:
        context = self._resolve_context(car_model, track_name)
        recommendations: List[Recommendation] = []
        for rule in self.rules:
            recommendations.extend(list(rule.evaluate(results, microsectors, context)))
        enumerated = list(enumerate(recommendations))
        enumerated.sort(key=lambda item: (item[1].priority, item[0]))
        unique: List[Recommendation] = []
        seen = set()
        for _, recommendation in enumerated:
            key = (recommendation.category, recommendation.message)
            if key in seen:
                continue
            seen.add(key)
            unique.append(recommendation)
        return unique
