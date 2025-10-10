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

import math

from dataclasses import dataclass, field
from types import MappingProxyType
from statistics import fmean
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Sequence

from ..core.cache import LRUCache
from ..core.cache_settings import CacheOptions, resolve_recommender_cache_size
from ..core.dissonance import compute_useful_dissonance_stats
from ..core.epi_models import EPIBundle
from ..core.segmentation import Microsector
from .pareto import ParetoPoint
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
    sci: float
    iterations: int
    evaluations: int
    telemetry: Sequence[EPIBundle]


@dataclass
class Plan:
    """Aggregated plan mixing optimisation deltas with explainable rules."""

    decision_vector: Dict[str, float]
    sci: float
    telemetry: Sequence[EPIBundle]
    recommendations: Sequence[Recommendation]
    sensitivities: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    phase_sensitivities: Mapping[str, Mapping[str, Mapping[str, float]]] = field(
        default_factory=dict
    )
    sci_breakdown: Mapping[str, float] = field(default_factory=dict)


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


def objective_score(
    results: Sequence[EPIBundle],
    microsectors: Sequence[Microsector] | None = None,
    *,
    session_weights: Mapping[str, Mapping[str, float]] | None = None,
    session_hints: Mapping[str, object] | None = None,
    breakdown: MutableMapping[str, float] | None = None,
) -> float:
    """Compute the Integrated Control Score combining Si and stability penalties."""

    if breakdown is not None:
        breakdown.clear()
    if not results:
        return float("-inf")

    def _mean(values: Sequence[float], default: float = 0.0) -> float:
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        return fmean(numeric) if numeric else default

    def _hint_float(key: str, default: float) -> float:
        if not isinstance(session_hints, Mapping):
            return default
        candidate = session_hints.get(key)
        try:
            numeric = float(candidate)
        except (TypeError, ValueError):
            return default
        return numeric

    def _resolve_session_weight(category: str, default: float) -> float:
        if not isinstance(session_weights, Mapping):
            return default
        collected: list[float] = []
        for profile in session_weights.values():
            if not isinstance(profile, Mapping):
                continue
            value = profile.get(category)
            if value is None and category != "__default__":
                value = profile.get("__default__")
            try:
                collected.append(float(value))
            except (TypeError, ValueError):
                continue
        if collected:
            return max(0.0, fmean(collected))
        return default

    timestamps = [float(bundle.timestamp) for bundle in results]
    duration = max(timestamps[-1] - timestamps[0], 1e-3) if len(timestamps) >= 2 else 1.0

    def _rate_series(series: Sequence[float], stamps: Sequence[float]) -> list[float]:
        rates: list[float] = []
        limit = min(len(series), len(stamps))
        for index in range(1, limit):
            dt = stamps[index] - stamps[index - 1]
            if dt <= 0.0:
                continue
            delta = series[index] - series[index - 1]
            rate = delta / dt
            if math.isfinite(rate):
                rates.append(rate)
        return rates

    mean_si = fmean(bundle.sense_index for bundle in results)

    delta_integral = _absolute_delta_integral(results)
    delta_reference = max(1e-3, _hint_float("delta_reference", 6.0))
    delta_density = delta_integral / duration
    delta_score = max(0.0, 1.0 - delta_density / delta_reference)

    yaw_rates = [float(bundle.chassis.yaw_rate) for bundle in results]
    delta_series = [float(bundle.delta_nfr) for bundle in results]
    _, _, udr_ratio = compute_useful_dissonance_stats(timestamps, delta_series, yaw_rates)
    udr_score = max(0.0, min(1.0, udr_ratio))

    bottoming_threshold_front = max(0.0, _hint_float("bottoming_threshold_front", 0.015))
    bottoming_threshold_rear = max(0.0, _hint_float("bottoming_threshold_rear", 0.015))
    front_travel = [float(bundle.suspension.travel_front) for bundle in results]
    rear_travel = [float(bundle.suspension.travel_rear) for bundle in results]
    if front_travel:
        front_bottom_ratio = sum(1.0 for value in front_travel if value <= bottoming_threshold_front) / len(front_travel)
    else:
        front_bottom_ratio = 0.0
    if rear_travel:
        rear_bottom_ratio = sum(1.0 for value in rear_travel if value <= bottoming_threshold_rear) / len(rear_travel)
    else:
        rear_bottom_ratio = 0.0
    bottoming_score = max(0.0, 1.0 - max(front_bottom_ratio, rear_bottom_ratio))

    aero_reference = max(1e-3, _hint_float("aero_reference", 0.12))
    aero_samples = []
    for bundle in results:
        tyres = getattr(bundle, "tyres", None)
        if tyres is None:
            continue
        try:
            front_component = float(getattr(tyres, "mu_eff_front", 0.0))
            rear_component = float(getattr(tyres, "mu_eff_rear", 0.0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(front_component) and math.isfinite(rear_component):
            aero_samples.append(abs(front_component - rear_component))
    aero_imbalance = _mean(aero_samples, 0.0)
    aero_score = max(0.0, 1.0 - aero_imbalance / aero_reference)

    weight_map = {
        "sense": _resolve_session_weight("__default__", 1.0),
        "delta": _resolve_session_weight("tyres", 1.0),
        "udr": _resolve_session_weight("chassis", 0.8),
        "bottoming": _resolve_session_weight("suspension", 0.7),
        "aero": _resolve_session_weight("aero", 0.6),
    }
    weight_sum = sum(max(0.0, weight) for weight in weight_map.values())
    if weight_sum <= 1e-9:
        weight_sum = 1.0
    normalised_weights = {
        key: max(0.0, value) / weight_sum for key, value in weight_map.items()
    }

    component_scores = {
        "sense": max(0.0, min(1.0, mean_si)),
        "delta": max(0.0, min(1.0, delta_score)),
        "udr": max(0.0, min(1.0, udr_score)),
        "bottoming": max(0.0, min(1.0, bottoming_score)),
        "aero": max(0.0, min(1.0, aero_score)),
    }

    contributions = {
        key: normalised_weights[key] * component_scores[key] for key in component_scores
    }

    if breakdown is not None:
        breakdown.update(
            sense=contributions["sense"],
            delta=contributions["delta"],
            udr=contributions["udr"],
            bottoming=contributions["bottoming"],
            aero=contributions["aero"],
        )

    return sum(contributions.values())


def _penalty_breakdown(components: Mapping[str, float]) -> Mapping[str, float]:
    penalties: Dict[str, float] = {}
    for key, value in components.items():
        try:
            penalties[key] = max(0.0, 1.0 - float(value))
        except (TypeError, ValueError):
            penalties[key] = 1.0
    return MappingProxyType(penalties)


def evaluate_candidate(
    space: DecisionSpace,
    vector: Mapping[str, float],
    baseline: Sequence[EPIBundle],
    *,
    microsectors: Sequence[Microsector] | None = None,
    simulator: Callable[[Mapping[str, float], Sequence[EPIBundle]], Sequence[EPIBundle]] | None = None,
    session_weights: Mapping[str, Mapping[str, float]] | None = None,
    session_hints: Mapping[str, object] | None = None,
    cache: LRUCache[
        tuple[tuple[str, float], ...],
        tuple[float, tuple[EPIBundle, ...], Mapping[str, float], Mapping[str, float]],
    ]
    | None = None,
) -> ParetoPoint:
    """Evaluate ``vector`` returning a :class:`ParetoPoint`."""

    clamped = space.clamp(vector)
    key = tuple(sorted(clamped.items()))
    candidate_cache = cache or LRUCache(maxsize=0)

    def _compute() -> tuple[
        float,
        tuple[EPIBundle, ...],
        Mapping[str, float],
        Mapping[str, float],
    ]:
        simulated = simulator(clamped, baseline) if simulator else baseline
        breakdown_components: Dict[str, float] = {}
        score_value = objective_score(
            simulated,
            microsectors,
            session_weights=session_weights,
            session_hints=session_hints,
            breakdown=breakdown_components,
        )
        return (
            score_value,
            tuple(simulated),
            MappingProxyType(dict(breakdown_components)),
            _penalty_breakdown(breakdown_components),
        )

    score, telemetry, components, penalties = candidate_cache.get_or_create(
        key, _compute
    )
    return ParetoPoint(
        decision_vector=MappingProxyType(dict(clamped)),
        score=score,
        breakdown=penalties,
        objective_breakdown=components,
        telemetry=telemetry,
    )


def axis_sweep_vectors(
    space: DecisionSpace,
    centre: Mapping[str, float],
    *,
    radius: int = 1,
    include_centre: bool = True,
) -> list[Dict[str, float]]:
    """Generate axis-aligned candidates around ``centre``."""

    clamped = space.clamp(centre)
    vectors: list[Dict[str, float]] = []
    if include_centre:
        vectors.append(dict(clamped))
    radius = max(0, int(radius))
    if radius <= 0:
        return vectors
    for variable in space.variables:
        current = clamped.get(variable.name, 0.0)
        for step in range(1, radius + 1):
            for direction in (-1.0, 1.0):
                candidate_value = variable.clamp(current + direction * variable.step * step)
                if candidate_value == current:
                    continue
                candidate = dict(clamped)
                candidate[variable.name] = candidate_value
                vectors.append(candidate)
    unique: Dict[tuple[tuple[str, float], ...], Dict[str, float]] = {}
    for vector in vectors:
        key = tuple(sorted(vector.items()))
        unique[key] = vector
    return list(unique.values())


def _resolve_planner_cache_size(
    cache_options: CacheOptions | None = None,
    cache_size: int | None = None,
) -> int:
    """Resolve the planner cache size from options or legacy ``cache_size``."""

    if cache_options is not None:
        return resolve_recommender_cache_size(cache_options.recommender_cache_size)
    return resolve_recommender_cache_size(cache_size)


def sweep_candidates(
    space: DecisionSpace,
    centre: Mapping[str, float],
    baseline: Sequence[EPIBundle],
    *,
    microsectors: Sequence[Microsector] | None = None,
    simulator: Callable[[Mapping[str, float], Sequence[EPIBundle]], Sequence[EPIBundle]] | None = None,
    session_weights: Mapping[str, Mapping[str, float]] | None = None,
    session_hints: Mapping[str, object] | None = None,
    radius: int = 1,
    include_centre: bool = True,
    candidates: Iterable[Mapping[str, float]] | None = None,
    cache_size: int | None = None,
    cache_options: CacheOptions | None = None,
) -> list[ParetoPoint]:
    """Evaluate a sweep of candidates returning :class:`ParetoPoint` entries."""

    resolved_cache_size = _resolve_planner_cache_size(
        cache_options=cache_options, cache_size=cache_size
    )
    cache: LRUCache[
        tuple[tuple[str, float], ...],
        tuple[float, tuple[EPIBundle, ...], Mapping[str, float], Mapping[str, float]],
    ] = LRUCache(maxsize=resolved_cache_size)
    if candidates is None:
        vectors = axis_sweep_vectors(
            space,
            centre,
            radius=radius,
            include_centre=include_centre,
        )
    else:
        vectors = [space.clamp(vector) for vector in candidates]
    points: list[ParetoPoint] = []
    for vector in vectors:
        points.append(
            evaluate_candidate(
                space,
                vector,
                baseline,
                microsectors=microsectors,
                simulator=simulator,
                session_weights=session_weights,
                session_hints=session_hints,
                cache=cache,
            )
        )
    return points


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


_ALIAS_DECISION_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "gt_fzr": "FZR",
        "gt_xrr": "XRR",
        "formula_fo8": "FO8",
        "formula_fox": "FOX",
    }
)


DEFAULT_DECISION_LIBRARY: Mapping[str, DecisionSpace] = {
    **_LFS_DECISION_LIBRARY,
    **{
        alias: _LFS_DECISION_LIBRARY[target]
        for alias, target in _ALIAS_DECISION_KEYS.items()
        if target in _LFS_DECISION_LIBRARY
    },
}


class SetupPlanner:
    """High level API combining optimisation with explainable rules."""

    def __init__(
        self,
        recommendation_engine: RecommendationEngine | None = None,
        decision_library: Mapping[str, DecisionSpace] | None = None,
        optimiser: CoordinateDescentOptimizer | None = None,
        cache_size: int | None = None,
        cache_options: CacheOptions | None = None,
    ) -> None:
        self.recommendation_engine = recommendation_engine or RecommendationEngine()
        self.decision_library = decision_library or DEFAULT_DECISION_LIBRARY
        self.optimiser = optimiser or CoordinateDescentOptimizer()
        self._cache_size = _resolve_planner_cache_size(
            cache_options=cache_options, cache_size=cache_size
        )

    @property
    def cache_size(self) -> int:
        """Return the maximum entries stored in the planner cache."""

        return self._cache_size

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
        cache: LRUCache[
            tuple[tuple[str, float], ...],
            tuple[float, tuple[EPIBundle, ...], Mapping[str, float]],
        ] = LRUCache(maxsize=self._cache_size)
        session_payload = getattr(self.recommendation_engine, "session", None)
        session_hints: Mapping[str, Any] | None = None
        session_weights: Mapping[str, Mapping[str, float]] | None = None
        if isinstance(session_payload, Mapping):
            hints_payload = session_payload.get("hints")
            if isinstance(hints_payload, Mapping):
                session_hints = hints_payload
            weights_payload = session_payload.get("weights")
            if isinstance(weights_payload, Mapping):
                session_weights = weights_payload  # type: ignore[assignment]

        def _materialise(
            clamped_vector: Mapping[str, float]
        ) -> tuple[float, tuple[EPIBundle, ...], Mapping[str, float]]:
            simulated = simulator(clamped_vector, baseline) if simulator else baseline
            local_breakdown: Dict[str, float] = {}
            score_value = objective_score(
                simulated,
                microsectors,
                session_weights=session_weights,
                session_hints=session_hints,
                breakdown=local_breakdown,
            )
            return (
                score_value,
                tuple(simulated),
                MappingProxyType(dict(local_breakdown)),
            )

        def _simulate_and_score(
            vector: Mapping[str, float]
        ) -> tuple[float, tuple[EPIBundle, ...], Mapping[str, float]]:
            clamped = space.clamp(vector)
            key = tuple(sorted(clamped.items()))
            stored_score, stored_results, stored_breakdown = cache.get_or_create(
                key, lambda: _materialise(clamped)
            )
            return stored_score, stored_results, stored_breakdown

        def evaluate(vector: Mapping[str, float]) -> float:
            return _simulate_and_score(vector)[0]

        vector, score, iterations, evaluations = self.optimiser.optimise(evaluate, space)
        _, telemetry, sci_breakdown = _simulate_and_score(vector)
        recommendations = list(self.recommendation_engine.generate(telemetry, microsectors))
        if session_hints:
            extra: list[Recommendation] = []
            slip_bias = session_hints.get("slip_ratio_bias")
            if isinstance(slip_bias, str) and slip_bias:
                direction = "front axle" if slip_bias.lower() == "front" else "rear axle"
                extra.append(
                    Recommendation(
                        category="aero",
                        message=f"Session hint: reinforce aero balance on the {direction}",
                        rationale=(
                            f"Session hint slip_ratio_bias={slip_bias} prioritises aerodynamic adjustments "
                            f"on the {direction}."
                        ),
                        priority=108,
                    )
                )
            surface = session_hints.get("surface")
            if isinstance(surface, str) and surface:
                extra.append(
                    Recommendation(
                        category="suspension",
                        message=f"Session hint: adapt damping to {surface} surface",
                        rationale=(
                            f"Session metadata describes surface={surface}; prioritise damping and ride heights "
                            "for that condition."
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
            session_weights=session_weights,
            session_hints=session_hints,
        )
        return Plan(
            decision_vector=vector,
            sci=score,
            telemetry=telemetry,
            recommendations=tuple(recommendations),
            sensitivities=sensitivities,
            phase_sensitivities=phase_sensitivities,
            sci_breakdown=sci_breakdown,
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
        cache: LRUCache[
            tuple[tuple[str, float], ...],
            tuple[float, tuple[EPIBundle, ...], Mapping[str, float]],
        ],
        score: float,
        session_weights: Mapping[str, Mapping[str, float]] | None,
        session_hints: Mapping[str, object] | None,
    ) -> tuple[Mapping[str, Mapping[str, float]], Mapping[str, Mapping[str, Mapping[str, float]]]]:
        if not telemetry:
            return {}, {}

        base_mean_si = fmean(bundle.sense_index for bundle in telemetry)
        sensitivities: Dict[str, Dict[str, float]] = {
            "sci": {},
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

        def _materialise(
            clamped_vector: Mapping[str, float]
        ) -> tuple[float, tuple[EPIBundle, ...], Mapping[str, float]]:
            simulated = simulator(clamped_vector, baseline) if simulator else baseline
            local_breakdown: Dict[str, float] = {}
            score_value = objective_score(
                simulated,
                microsectors,
                session_weights=session_weights,
                session_hints=session_hints,
                breakdown=local_breakdown,
            )
            return (
                score_value,
                tuple(simulated),
                MappingProxyType(dict(local_breakdown)),
            )

        def _simulate(clamped_vector: Mapping[str, float]) -> tuple[float, tuple[EPIBundle, ...]]:
            key = tuple(sorted(clamped_vector.items()))
            stored_score, stored_results, _ = cache.get_or_create(
                key,
                lambda: _materialise(clamped_vector),
            )
            return stored_score, stored_results

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
                sensitivities["sci"][variable.name] = (plus_score - minus_score) / denom
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
                sensitivities["sci"][variable.name] = (plus_score - score) / step
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
                sensitivities["sci"][variable.name] = (score - minus_score) / step
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
                sensitivities["sci"][variable.name] = 0.0
                sensitivities["sense_index"][variable.name] = 0.0
                sensitivities["delta_nfr_integral"][variable.name] = 0.0

        return sensitivities, phase_sensitivities

