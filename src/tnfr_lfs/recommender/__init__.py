"""Recommendation engine used to produce setup suggestions."""

from tnfr_lfs.recommender.pareto import ParetoPoint, pareto_front
from tnfr_lfs.recommender.rules import Recommendation, RecommendationEngine
from tnfr_lfs.recommender.search import (
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
