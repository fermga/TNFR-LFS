"""Recommendation engine used to produce setup suggestions."""

from .pareto import ParetoPoint, pareto_front
from .rules import Recommendation, RecommendationEngine
from .search import (
    axis_sweep_vectors,
    CoordinateDescentOptimizer,
    DecisionSpace,
    DecisionVariable,
    evaluate_candidate,
    Plan,
    SearchResult,
    SetupPlanner,
    objective_score,
    sweep_candidates,
)

__all__ = [
    "ParetoPoint",
    "pareto_front",
    "axis_sweep_vectors",
    "CoordinateDescentOptimizer",
    "DecisionSpace",
    "DecisionVariable",
    "evaluate_candidate",
    "Plan",
    "Recommendation",
    "RecommendationEngine",
    "SearchResult",
    "SetupPlanner",
    "objective_score",
    "sweep_candidates",
]
