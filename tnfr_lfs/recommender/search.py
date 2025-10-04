"""Search utilities to build optimisation-aware setup plans.

This module introduces a light-weight coordinate descent optimiser that works
on top of a decision vector whose bounds are defined per car model.  The
optimiser evaluates candidate vectors through a domain specific objective that
favours a higher Sense Index (Si) while penalising the integral of the absolute
ΔNFR within each microsector.  The resulting plan can be combined with the
rule-based recommendation engine to surface explainable guidance together with
the optimised setup deltas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Callable, Dict, Mapping, MutableMapping, Sequence

from ..core.epi_models import EPIBundle
from ..core.segmentation import Microsector
from .rules import Recommendation, RecommendationEngine


DecisionVector = Mapping[str, float]


@dataclass(frozen=True)
class DecisionVariable:
    """Represents an adjustable setup parameter bounded by the car model."""

    name: str
    lower: float
    upper: float
    step: float

    def clamp(self, value: float) -> float:
        return min(self.upper, max(self.lower, value))


@dataclass(frozen=True)
class DecisionSpace:
    """Collection of decision variables valid for a given car model."""

    car_model: str
    variables: Sequence[DecisionVariable]

    def initial_guess(self) -> Dict[str, float]:
        return {var.name: (var.lower + var.upper) / 2 for var in self.variables}

    def clamp(self, vector: Mapping[str, float]) -> Dict[str, float]:
        return {var.name: var.clamp(vector.get(var.name, 0.0)) for var in self.variables}


@dataclass
class SearchResult:
    """Outcome of the optimisation stage."""

    decision_vector: Dict[str, float]
    objective_value: float
    iterations: int
    evaluations: int
    telemetry: Sequence[EPIBundle]


@dataclass
class Plan:
    """Aggregated plan mixing optimisation deltas with explainable rules."""

    decision_vector: Dict[str, float]
    objective_value: float
    telemetry: Sequence[EPIBundle]
    recommendations: Sequence[Recommendation]
    sensitivities: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


class CoordinateDescentOptimizer:
    """Simple coordinate descent routine with bound-aware steps."""

    def __init__(self, tolerance: float = 1e-4, max_iterations: int = 50):
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def optimise(
        self,
        objective: Callable[[Mapping[str, float]], float],
        space: DecisionSpace,
        initial_vector: Mapping[str, float] | None = None,
    ) -> tuple[Dict[str, float], float, int, int]:
        """Return the best decision vector according to ``objective``."""

        vector = space.clamp(initial_vector or space.initial_guess())
        best_score = objective(vector)
        evaluations = 1
        for iteration in range(1, self.max_iterations + 1):
            improved = False
            for variable in space.variables:
                current_value = vector[variable.name]
                best_local_value = current_value
                local_best = best_score
                for direction in (-1.0, 1.0):
                    candidate_value = variable.clamp(current_value + direction * variable.step)
                    if candidate_value == current_value:
                        continue
                    candidate_vector = dict(vector)
                    candidate_vector[variable.name] = candidate_value
                    score = objective(candidate_vector)
                    evaluations += 1
                    if score > local_best + self.tolerance:
                        local_best = score
                        best_local_value = candidate_value
                if best_local_value != current_value:
                    vector[variable.name] = best_local_value
                    best_score = local_best
                    improved = True
            if not improved:
                return vector, best_score, iteration, evaluations
        return vector, best_score, self.max_iterations, evaluations


def _timestamp_delta(results: Sequence[EPIBundle], index: int) -> float:
    if not results:
        return 0.0
    if index + 1 < len(results):
        return max(1e-3, results[index + 1].timestamp - results[index].timestamp)
    if index > 0:
        return max(1e-3, results[index].timestamp - results[index - 1].timestamp)
    return 1e-3


def _microsector_integral(results: Sequence[EPIBundle], microsector: Microsector) -> float:
    integral = 0.0
    for start, stop in microsector.phase_boundaries.values():
        for idx in range(start, min(stop, len(results))):
            integral += abs(results[idx].delta_nfr) * _timestamp_delta(results, idx)
    return integral


def objective_score(results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None) -> float:
    """Compute the scalar objective combining Si and |ΔNFR| integrals."""

    if not results:
        return float("-inf")
    mean_si = fmean(bundle.sense_index for bundle in results)
    if microsectors:
        integral = sum(_microsector_integral(results, micro) for micro in microsectors)
        duration = max(results[-1].timestamp - results[0].timestamp, 1e-3)
        nfr_penalty = integral / duration
    else:
        nfr_penalty = fmean(abs(bundle.delta_nfr) for bundle in results)
    return mean_si - 0.05 * nfr_penalty


DEFAULT_DECISION_LIBRARY: Mapping[str, DecisionSpace] = {
    "generic_gt": DecisionSpace(
        car_model="generic_gt",
        variables=(
            DecisionVariable("brake_bias_pct", lower=-4.0, upper=4.0, step=0.5),
            DecisionVariable("front_rebound_clicks", lower=-6.0, upper=6.0, step=1.0),
            DecisionVariable("rear_rebound_clicks", lower=-6.0, upper=6.0, step=1.0),
            DecisionVariable("front_compression_clicks", lower=-6.0, upper=6.0, step=1.0),
            DecisionVariable("rear_compression_clicks", lower=-6.0, upper=6.0, step=1.0),
            DecisionVariable("front_arb_steps", lower=-5.0, upper=5.0, step=1.0),
            DecisionVariable("rear_arb_steps", lower=-5.0, upper=5.0, step=1.0),
            DecisionVariable("diff_power_lock", lower=-20.0, upper=20.0, step=5.0),
            DecisionVariable("diff_coast_lock", lower=-20.0, upper=20.0, step=5.0),
            DecisionVariable("front_tyre_pressure", lower=-0.6, upper=0.6, step=0.1),
            DecisionVariable("rear_tyre_pressure", lower=-0.6, upper=0.6, step=0.1),
            DecisionVariable("rear_ride_height", lower=-4.0, upper=4.0, step=0.5),
            DecisionVariable("front_ride_height", lower=-4.0, upper=4.0, step=0.5),
            DecisionVariable("rear_wing_angle", lower=-3.0, upper=3.0, step=0.5),
        ),
    ),
    "formula": DecisionSpace(
        car_model="formula",
        variables=(
            DecisionVariable("brake_bias_pct", lower=-3.0, upper=3.0, step=0.25),
            DecisionVariable("front_rebound_clicks", lower=-8.0, upper=8.0, step=1.0),
            DecisionVariable("rear_rebound_clicks", lower=-8.0, upper=8.0, step=1.0),
            DecisionVariable("front_compression_clicks", lower=-8.0, upper=8.0, step=1.0),
            DecisionVariable("rear_compression_clicks", lower=-8.0, upper=8.0, step=1.0),
            DecisionVariable("front_arb_steps", lower=-6.0, upper=6.0, step=1.0),
            DecisionVariable("rear_arb_steps", lower=-6.0, upper=6.0, step=1.0),
            DecisionVariable("diff_power_lock", lower=-25.0, upper=25.0, step=5.0),
            DecisionVariable("diff_coast_lock", lower=-25.0, upper=25.0, step=5.0),
            DecisionVariable("front_tyre_pressure", lower=-0.5, upper=0.5, step=0.05),
            DecisionVariable("rear_tyre_pressure", lower=-0.5, upper=0.5, step=0.05),
            DecisionVariable("rear_ride_height", lower=-8.0, upper=8.0, step=0.5),
            DecisionVariable("front_ride_height", lower=-8.0, upper=8.0, step=0.5),
            DecisionVariable("rear_wing_angle", lower=-5.0, upper=5.0, step=0.5),
        ),
    ),
}


class SetupPlanner:
    """High level API combining optimisation with explainable rules."""

    def __init__(
        self,
        recommendation_engine: RecommendationEngine | None = None,
        decision_library: Mapping[str, DecisionSpace] | None = None,
        optimiser: CoordinateDescentOptimizer | None = None,
    ) -> None:
        self.recommendation_engine = recommendation_engine or RecommendationEngine()
        self.decision_library = decision_library or DEFAULT_DECISION_LIBRARY
        self.optimiser = optimiser or CoordinateDescentOptimizer()

    def _space_for_car(self, car_model: str) -> DecisionSpace:
        try:
            return self.decision_library[car_model]
        except KeyError as exc:
            raise KeyError(f"No decision space registered for car model '{car_model}'.") from exc

    def plan(
        self,
        baseline: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        *,
        car_model: str = "generic_gt",
        simulator: Callable[[Mapping[str, float], Sequence[EPIBundle]], Sequence[EPIBundle]] | None = None,
    ) -> Plan:
        """Generate the final plan that blends search and rule-based guidance."""

        space = self._space_for_car(car_model)
        cache: MutableMapping[tuple[tuple[str, float], ...], tuple[float, Sequence[EPIBundle]]] = {}

        def _simulate_and_score(vector: Mapping[str, float]) -> tuple[float, Sequence[EPIBundle]]:
            clamped = space.clamp(vector)
            key = tuple(sorted(clamped.items()))
            if key not in cache:
                simulated = simulator(clamped, baseline) if simulator else baseline
                cache[key] = (objective_score(simulated, microsectors), simulated)
            return cache[key]

        def evaluate(vector: Mapping[str, float]) -> float:
            return _simulate_and_score(vector)[0]

        vector, score, iterations, evaluations = self.optimiser.optimise(evaluate, space)
        telemetry = _simulate_and_score(vector)[1]
        recommendations = self.recommendation_engine.generate(telemetry, microsectors)
        sensitivities = self._compute_sensitivities(
            vector=vector,
            telemetry=telemetry,
            baseline=baseline,
            microsectors=microsectors,
            simulator=simulator,
            space=space,
            cache=cache,
            score=score,
        )
        return Plan(
            decision_vector=vector,
            objective_value=score,
            telemetry=telemetry,
            recommendations=tuple(recommendations),
            sensitivities=sensitivities,
        )

    def _compute_sensitivities(
        self,
        *,
        vector: Mapping[str, float],
        telemetry: Sequence[EPIBundle],
        baseline: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None,
        simulator: Callable[[Mapping[str, float], Sequence[EPIBundle]], Sequence[EPIBundle]] | None,
        space: DecisionSpace,
        cache: MutableMapping[tuple[tuple[str, float], ...], tuple[float, Sequence[EPIBundle]]],
        score: float,
    ) -> Mapping[str, Mapping[str, float]]:
        if not telemetry:
            return {}

        base_mean_si = fmean(bundle.sense_index for bundle in telemetry)
        sensitivities: Dict[str, Dict[str, float]] = {
            "objective_score": {},
            "sense_index": {},
        }

        def _simulate(clamped_vector: Mapping[str, float]) -> tuple[float, Sequence[EPIBundle]]:
            key = tuple(sorted(clamped_vector.items()))
            if key not in cache:
                simulated = simulator(clamped_vector, baseline) if simulator else baseline
                cache[key] = (objective_score(simulated, microsectors), simulated)
            return cache[key]

        for variable in space.variables:
            base_value = vector[variable.name]
            raw_step = max(variable.step * 0.25, 1e-3)
            forward_room = variable.upper - base_value
            backward_room = base_value - variable.lower
            central_step = min(raw_step, forward_room, backward_room)

            if central_step > 1e-9:
                plus_value = base_value + central_step
                minus_value = base_value - central_step
                plus_vector = dict(vector)
                minus_vector = dict(vector)
                plus_vector[variable.name] = plus_value
                minus_vector[variable.name] = minus_value
                plus_clamped = space.clamp(plus_vector)
                minus_clamped = space.clamp(minus_vector)
                plus_score, plus_telemetry = _simulate(plus_clamped)
                minus_score, minus_telemetry = _simulate(minus_clamped)
                si_plus = fmean(bundle.sense_index for bundle in plus_telemetry)
                si_minus = fmean(bundle.sense_index for bundle in minus_telemetry)
                denom = 2.0 * central_step
                sensitivities["objective_score"][variable.name] = (plus_score - minus_score) / denom
                sensitivities["sense_index"][variable.name] = (si_plus - si_minus) / denom
                continue

            if forward_room > 1e-9:
                step = min(raw_step, forward_room)
                plus_vector = dict(vector)
                plus_vector[variable.name] = base_value + step
                plus_clamped = space.clamp(plus_vector)
                plus_score, plus_telemetry = _simulate(plus_clamped)
                si_plus = fmean(bundle.sense_index for bundle in plus_telemetry)
                sensitivities["objective_score"][variable.name] = (plus_score - score) / step
                sensitivities["sense_index"][variable.name] = (si_plus - base_mean_si) / step
            elif backward_room > 1e-9:
                step = min(raw_step, backward_room)
                minus_vector = dict(vector)
                minus_vector[variable.name] = base_value - step
                minus_clamped = space.clamp(minus_vector)
                minus_score, minus_telemetry = _simulate(minus_clamped)
                si_minus = fmean(bundle.sense_index for bundle in minus_telemetry)
                sensitivities["objective_score"][variable.name] = (score - minus_score) / step
                sensitivities["sense_index"][variable.name] = (base_mean_si - si_minus) / step
            else:
                sensitivities["objective_score"][variable.name] = 0.0
                sensitivities["sense_index"][variable.name] = 0.0

        return sensitivities

