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
from types import MappingProxyType
from statistics import fmean
from typing import Any, Callable, Dict, Mapping, MutableMapping, Sequence

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
    phase_sensitivities: Mapping[str, Mapping[str, Mapping[str, float]]] = field(
        default_factory=dict
    )


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


def _absolute_delta_integral(results: Sequence[EPIBundle]) -> float:
    total = 0.0
    for idx, bundle in enumerate(results):
        total += abs(bundle.delta_nfr) * _timestamp_delta(results, idx)
    return total


def _phase_integrals(
    results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None
) -> Dict[str, float]:
    if not results or not microsectors:
        return {}
    totals: Dict[str, float] = {}
    for microsector in microsectors:
        for phase, (start, stop) in microsector.phase_boundaries.items():
            subtotal = 0.0
            for idx in range(start, min(stop, len(results))):
                subtotal += abs(results[idx].delta_nfr) * _timestamp_delta(results, idx)
            if subtotal:
                totals[phase] = totals.get(phase, 0.0) + subtotal
    return totals


def objective_score(results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None) -> float:
    """Compute the scalar objective combining Si and |ΔNFR| integrals."""

    if not results:
        return float("-inf")
    mean_si = fmean(bundle.sense_index for bundle in results)
    coherence_mean = fmean(getattr(bundle, "coherence_index", 0.0) for bundle in results)
    geometry_penalty = 0.0
    geometry_samples = 0
    if microsectors:
        integral = sum(_microsector_integral(results, micro) for micro in microsectors)
        duration = max(results[-1].timestamp - results[0].timestamp, 1e-3)
        nfr_penalty = integral / duration
        for micro in microsectors:
            for goal in micro.goals:
                measured_alignment = micro.phase_alignment.get(
                    goal.phase, getattr(goal, "measured_phase_alignment", 1.0)
                )
                target_alignment = getattr(
                    goal, "target_phase_alignment", measured_alignment
                )
                measured_lag = micro.phase_lag.get(
                    goal.phase, getattr(goal, "measured_phase_lag", 0.0)
                )
                target_lag = getattr(goal, "target_phase_lag", measured_lag)
                geometry_penalty += abs(target_alignment - measured_alignment)
                geometry_penalty += 0.5 * abs(measured_lag - target_lag)
                geometry_samples += 1
    else:
        nfr_penalty = fmean(abs(bundle.delta_nfr) for bundle in results)
    if geometry_samples:
        geometry_penalty /= geometry_samples
    return mean_si + 0.05 * coherence_mean - 0.05 * nfr_penalty - 0.02 * geometry_penalty


_ROAD_ALIGNMENT = (
    ("front_camber_deg", -1.6, 1.6, 0.1),
    ("rear_camber_deg", -1.4, 1.4, 0.1),
    ("front_toe_deg", -0.6, 0.6, 0.05),
    ("rear_toe_deg", -0.6, 0.6, 0.05),
    ("caster_deg", -1.0, 1.0, 0.25),
    ("steering_lock_deg", -6.0, 6.0, 0.5),
    ("parallel_steer", -0.6, 0.6, 0.05),
)

_ROAD_SUSPENSION = (
    ("front_spring_stiffness", -25.0, 25.0, 1.0),
    ("rear_spring_stiffness", -25.0, 25.0, 1.0),
    ("front_rebound_clicks", -8.0, 8.0, 1.0),
    ("rear_rebound_clicks", -8.0, 8.0, 1.0),
    ("front_compression_clicks", -8.0, 8.0, 1.0),
    ("rear_compression_clicks", -8.0, 8.0, 1.0),
    ("front_arb_steps", -6.0, 6.0, 1.0),
    ("rear_arb_steps", -6.0, 6.0, 1.0),
    ("front_ride_height", -15.0, 15.0, 0.5),
    ("rear_ride_height", -15.0, 15.0, 0.5),
)

_TRANSMISSION_RATIOS = (
    ("final_drive_ratio", -0.5, 0.5, 0.01),
    ("gear_1_ratio", -0.5, 0.5, 0.01),
    ("gear_2_ratio", -0.5, 0.5, 0.01),
    ("gear_3_ratio", -0.5, 0.5, 0.01),
    ("gear_4_ratio", -0.5, 0.5, 0.01),
    ("gear_5_ratio", -0.5, 0.5, 0.01),
    ("gear_6_ratio", -0.5, 0.5, 0.01),
    ("gear_7_ratio", -0.5, 0.5, 0.01),
)

_ROAD_MISC = (
    ("front_tyre_pressure", -0.6, 0.6, 0.05),
    ("rear_tyre_pressure", -0.6, 0.6, 0.05),
    ("brake_bias_pct", -6.0, 6.0, 0.25),
    ("brake_max_per_wheel", -0.25, 0.25, 0.01),
    ("diff_power_lock", -30.0, 30.0, 5.0),
    ("diff_coast_lock", -30.0, 30.0, 5.0),
    ("diff_preload_nm", -300.0, 300.0, 20.0),
) + _TRANSMISSION_RATIOS

_GTR_ALIGNMENT = (
    ("front_camber_deg", -1.4, 1.4, 0.05),
    ("rear_camber_deg", -1.2, 1.2, 0.05),
    ("front_toe_deg", -0.3, 0.3, 0.01),
    ("rear_toe_deg", -0.3, 0.3, 0.01),
    ("caster_deg", -0.8, 0.8, 0.1),
    ("steering_lock_deg", -4.0, 4.0, 0.5),
    ("parallel_steer", -0.4, 0.4, 0.025),
)

_GTR_SUSPENSION = (
    ("front_spring_stiffness", -60.0, 60.0, 2.0),
    ("rear_spring_stiffness", -60.0, 60.0, 2.0),
    ("front_rebound_clicks", -12.0, 12.0, 1.0),
    ("rear_rebound_clicks", -12.0, 12.0, 1.0),
    ("front_compression_clicks", -12.0, 12.0, 1.0),
    ("rear_compression_clicks", -12.0, 12.0, 1.0),
    ("front_arb_steps", -8.0, 8.0, 1.0),
    ("rear_arb_steps", -8.0, 8.0, 1.0),
    ("front_ride_height", -12.0, 12.0, 0.5),
    ("rear_ride_height", -12.0, 12.0, 0.5),
)

_GTR_MISC_BASE = (
    ("front_tyre_pressure", -0.4, 0.4, 0.05),
    ("rear_tyre_pressure", -0.4, 0.4, 0.05),
    ("brake_bias_pct", -4.0, 4.0, 0.25),
    ("brake_max_per_wheel", -0.2, 0.2, 0.01),
    ("diff_power_lock", -40.0, 40.0, 5.0),
    ("diff_coast_lock", -40.0, 40.0, 5.0),
    ("diff_preload_nm", -400.0, 400.0, 25.0),
    ("rear_wing_angle", -6.0, 6.0, 0.5),
) + _TRANSMISSION_RATIOS

_FORMULA_ALIGNMENT = (
    ("front_camber_deg", -1.8, 1.8, 0.05),
    ("rear_camber_deg", -1.6, 1.6, 0.05),
    ("front_toe_deg", -0.4, 0.4, 0.01),
    ("rear_toe_deg", -0.4, 0.4, 0.01),
    ("caster_deg", -0.6, 0.6, 0.05),
    ("steering_lock_deg", -3.0, 3.0, 0.25),
    ("parallel_steer", -0.3, 0.3, 0.02),
)

_FORMULA_SUSPENSION = (
    ("front_spring_stiffness", -90.0, 90.0, 2.0),
    ("rear_spring_stiffness", -90.0, 90.0, 2.0),
    ("front_rebound_clicks", -16.0, 16.0, 1.0),
    ("rear_rebound_clicks", -16.0, 16.0, 1.0),
    ("front_compression_clicks", -16.0, 16.0, 1.0),
    ("rear_compression_clicks", -16.0, 16.0, 1.0),
    ("front_arb_steps", -10.0, 10.0, 1.0),
    ("rear_arb_steps", -10.0, 10.0, 1.0),
    ("front_ride_height", -10.0, 10.0, 0.5),
    ("rear_ride_height", -10.0, 10.0, 0.5),
)

_FORMULA_MISC_BASE = (
    ("front_tyre_pressure", -0.3, 0.3, 0.02),
    ("rear_tyre_pressure", -0.3, 0.3, 0.02),
    ("brake_bias_pct", -3.0, 3.0, 0.25),
    ("brake_max_per_wheel", -0.15, 0.15, 0.01),
    ("diff_power_lock", -50.0, 50.0, 5.0),
    ("diff_coast_lock", -50.0, 50.0, 5.0),
    ("diff_preload_nm", -500.0, 500.0, 25.0),
    ("rear_wing_angle", -10.0, 10.0, 0.5),
) + _TRANSMISSION_RATIOS

_GTR_FRONT_WING_LIMITS: Mapping[str, tuple[float, float, float]] = MappingProxyType(
    {
        "FXR": (-5.0, 5.0, 0.5),
        "XRR": (-5.0, 5.0, 0.5),
        "FZR": (-6.0, 6.0, 0.5),
        "XFR": (-4.0, 4.0, 0.5),
        "UFR": (-4.0, 4.0, 0.5),
    }
)

_FORMULA_FRONT_WING_LIMITS: Mapping[str, tuple[float, float, float]] = MappingProxyType(
    {
        "FOX": (-7.0, 7.0, 0.5),
        "FO8": (-8.0, 8.0, 0.5),
        "BF1": (-12.0, 12.0, 0.5),
        "FBM": (-6.0, 6.0, 0.5),
        "MRT": (-5.0, 5.0, 0.5),
    }
)


def _build_space(car_model: str, spec: Sequence[tuple[str, float, float, float]]) -> DecisionSpace:
    return DecisionSpace(
        car_model=car_model,
        variables=tuple(DecisionVariable(name, lower, upper, step) for name, lower, upper, step in spec),
    )


_ROAD_MODELS = ("UF1", "XFG", "XRG", "RB4", "FXO", "LX4", "LX6", "XRT", "RAC", "FZ5")
_GTR_MODELS = ("FXR", "XRR", "FZR", "XFR", "UFR")
_FORMULA_MODELS = ("FOX", "FO8", "BF1", "FBM", "MRT")

_LFS_DECISION_LIBRARY: Dict[str, DecisionSpace] = {}
for model in _ROAD_MODELS:
    _LFS_DECISION_LIBRARY[model] = _build_space(
        model, _ROAD_ALIGNMENT + _ROAD_SUSPENSION + _ROAD_MISC
    )
for model in _GTR_MODELS:
    spec = list(_GTR_ALIGNMENT + _GTR_SUSPENSION + _GTR_MISC_BASE)
    limits = _GTR_FRONT_WING_LIMITS.get(model)
    if limits:
        lower, upper, step = limits
        spec.append(("front_wing_angle", lower, upper, step))
    _LFS_DECISION_LIBRARY[model] = _build_space(model, tuple(spec))
for model in _FORMULA_MODELS:
    spec = list(_FORMULA_ALIGNMENT + _FORMULA_SUSPENSION + _FORMULA_MISC_BASE)
    limits = _FORMULA_FRONT_WING_LIMITS.get(model)
    if limits:
        lower, upper, step = limits
        spec.append(("front_wing_angle", lower, upper, step))
    _LFS_DECISION_LIBRARY[model] = _build_space(model, tuple(spec))


DEFAULT_DECISION_LIBRARY: Mapping[str, DecisionSpace] = {
    **_LFS_DECISION_LIBRARY,
    "generic_gt": _build_space(
        "generic_gt",
        _ROAD_ALIGNMENT
        + _ROAD_SUSPENSION
        + _ROAD_MISC
        + (("rear_wing_angle", -3.0, 3.0, 0.5),),
    ),
    "formula": _build_space(
        "formula",
        _FORMULA_ALIGNMENT
        + _FORMULA_SUSPENSION
        + _FORMULA_MISC_BASE
        + (("front_wing_angle", -8.0, 8.0, 0.5),),
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
        key = (car_model or "").strip()
        if not key:
            raise ValueError("Car model must be provided to resolve a decision space.")
        for candidate in (key, key.upper(), key.lower()):
            if candidate in self.decision_library:
                return self.decision_library[candidate]
        available = ", ".join(sorted(self.decision_library))
        raise ValueError(f"No decision space registered for car model '{car_model}'. Available: {available}")

    def plan(
        self,
        baseline: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        *,
        car_model: str = "XFG",
        track_name: str | None = None,
        simulator: Callable[[Mapping[str, float], Sequence[EPIBundle]], Sequence[EPIBundle]] | None = None,
    ) -> Plan:
        """Generate the final plan that blends search and rule-based guidance."""

        space = self._space_for_car(car_model)
        resolved_track = track_name or getattr(self.recommendation_engine, "track_name", "")
        space = self._adapt_space(space, car_model, resolved_track)
        cache: MutableMapping[tuple[tuple[str, float], ...], tuple[float, Sequence[EPIBundle]]] = {}
        session_payload = getattr(self.recommendation_engine, "session", None)
        session_hints: Mapping[str, Any] | None = None
        if isinstance(session_payload, Mapping):
            hints_payload = session_payload.get("hints")
            if isinstance(hints_payload, Mapping):
                session_hints = hints_payload

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
        recommendations = list(self.recommendation_engine.generate(telemetry, microsectors))
        if session_hints:
            extra: list[Recommendation] = []
            slip_bias = session_hints.get("slip_ratio_bias")
            if isinstance(slip_bias, str) and slip_bias:
                direction = "delantero" if slip_bias.lower() == "front" else "trasero"
                extra.append(
                    Recommendation(
                        category="aero",
                        message=f"Hint sesión: refuerza aero {direction}",
                        rationale=(
                            f"El hint slip_ratio_bias={slip_bias} indica priorizar ajustes aerodinámicos "
                            f"en el eje {direction}."
                        ),
                        priority=108,
                    )
                )
            surface = session_hints.get("surface")
            if isinstance(surface, str) and surface:
                extra.append(
                    Recommendation(
                        category="suspension",
                        message=f"Hint sesión: adapta amortiguación a superficie {surface}",
                        rationale=(
                            f"La sesión describe surface={surface}; prioriza amortiguación y alturas "
                            "para esa condición."
                        ),
                        priority=96,
                    )
                )
            if extra:
                recommendations.extend(extra)
        sensitivities, phase_sensitivities = self._compute_sensitivities(
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
            phase_sensitivities=phase_sensitivities,
        )

    def _adapt_space(
        self,
        space: DecisionSpace,
        car_model: str,
        track_name: str | None,
    ) -> DecisionSpace:
        manager = getattr(self.recommendation_engine, "profile_manager", None)
        if manager is None:
            return space
        history, _ = manager.gradient_history(car_model, track_name or "")
        if not history:
            return space

        magnitude: Dict[str, float] = {}
        for metric in ("sense_index", "delta_nfr_integral"):
            derivatives = history.get(metric, {})
            for parameter, value in derivatives.items():
                try:
                    magnitude[parameter] = magnitude.get(parameter, 0.0) + abs(float(value))
                except (TypeError, ValueError):
                    continue
        if not magnitude:
            return space

        def _score(variable: DecisionVariable) -> float:
            return magnitude.get(variable.name, 0.0)

        adapted: list[DecisionVariable] = []
        for variable in sorted(space.variables, key=_score, reverse=True):
            score = magnitude.get(variable.name, 0.0)
            if score > 1.0:
                factor = 0.5
            elif score < 0.05:
                factor = 1.5
            else:
                factor = 1.0
            span = max(variable.upper - variable.lower, 1e-6)
            adjusted_step = min(max(variable.step * factor, 1e-3), span)
            adapted.append(
                DecisionVariable(
                    variable.name,
                    variable.lower,
                    variable.upper,
                    adjusted_step,
                )
            )
        return DecisionSpace(space.car_model, tuple(adapted))

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
    ) -> tuple[Mapping[str, Mapping[str, float]], Mapping[str, Mapping[str, Mapping[str, float]]]]:
        if not telemetry:
            return {}, {}

        base_mean_si = fmean(bundle.sense_index for bundle in telemetry)
        sensitivities: Dict[str, Dict[str, float]] = {
            "objective_score": {},
            "sense_index": {},
            "delta_nfr_integral": {},
        }
        phase_sensitivities: Dict[str, Dict[str, Dict[str, float]]] = {}

        def _integral_metrics(
            results: Sequence[EPIBundle],
        ) -> tuple[float, Dict[str, float]]:
            return _absolute_delta_integral(results), _phase_integrals(results, microsectors)

        def _accumulate_phase_gradients(
            phase_deltas: Dict[str, Dict[str, Dict[str, float]]],
            plus_values: Mapping[str, float],
            minus_values: Mapping[str, float],
            denom: float,
            parameter: str,
        ) -> None:
            if not plus_values and not minus_values:
                return
            for phase in set(plus_values) | set(minus_values):
                gradient = (plus_values.get(phase, 0.0) - minus_values.get(phase, 0.0)) / denom
                metrics = phase_deltas.setdefault(phase, {})
                entry = metrics.setdefault("delta_nfr_integral", {})
                entry[parameter] = gradient

        base_integral, base_phase = _integral_metrics(telemetry)

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
                integral_plus, phase_plus = _integral_metrics(plus_telemetry)
                integral_minus, phase_minus = _integral_metrics(minus_telemetry)
                denom = 2.0 * central_step
                sensitivities["objective_score"][variable.name] = (plus_score - minus_score) / denom
                sensitivities["sense_index"][variable.name] = (si_plus - si_minus) / denom
                sensitivities["delta_nfr_integral"][variable.name] = (
                    (integral_plus - integral_minus) / denom
                )
                _accumulate_phase_gradients(
                    phase_sensitivities, phase_plus, phase_minus, denom, variable.name
                )
                continue

            if forward_room > 1e-9:
                step = min(raw_step, forward_room)
                plus_vector = dict(vector)
                plus_vector[variable.name] = base_value + step
                plus_clamped = space.clamp(plus_vector)
                plus_score, plus_telemetry = _simulate(plus_clamped)
                si_plus = fmean(bundle.sense_index for bundle in plus_telemetry)
                integral_plus, phase_plus = _integral_metrics(plus_telemetry)
                sensitivities["objective_score"][variable.name] = (plus_score - score) / step
                sensitivities["sense_index"][variable.name] = (si_plus - base_mean_si) / step
                sensitivities["delta_nfr_integral"][variable.name] = (
                    (integral_plus - base_integral) / step
                )
                _accumulate_phase_gradients(
                    phase_sensitivities,
                    phase_plus,
                    base_phase,
                    step,
                    variable.name,
                )
            elif backward_room > 1e-9:
                step = min(raw_step, backward_room)
                minus_vector = dict(vector)
                minus_vector[variable.name] = base_value - step
                minus_clamped = space.clamp(minus_vector)
                minus_score, minus_telemetry = _simulate(minus_clamped)
                si_minus = fmean(bundle.sense_index for bundle in minus_telemetry)
                integral_minus, phase_minus = _integral_metrics(minus_telemetry)
                sensitivities["objective_score"][variable.name] = (score - minus_score) / step
                sensitivities["sense_index"][variable.name] = (base_mean_si - si_minus) / step
                sensitivities["delta_nfr_integral"][variable.name] = (
                    (base_integral - integral_minus) / step
                )
                _accumulate_phase_gradients(
                    phase_sensitivities,
                    base_phase,
                    phase_minus,
                    step,
                    variable.name,
                )
            else:
                sensitivities["objective_score"][variable.name] = 0.0
                sensitivities["sense_index"][variable.name] = 0.0
                sensitivities["delta_nfr_integral"][variable.name] = 0.0

        return sensitivities, phase_sensitivities

