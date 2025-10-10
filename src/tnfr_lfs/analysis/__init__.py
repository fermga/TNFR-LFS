"""Analysis helpers for TNFR × LFS."""

from .abtest import ABResult, SUPPORTED_LAP_METRICS, ab_compare_by_lap
from .brake_thermal import BrakeThermalConfig, BrakeThermalEstimator
from .insights import InsightsResult, compute_insights
from .robustness import compute_session_robustness

__all__ = [
    "ABResult",
    "BrakeThermalConfig",
    "BrakeThermalEstimator",
    "SUPPORTED_LAP_METRICS",
    "ab_compare_by_lap",
    "compute_session_robustness",
    "InsightsResult",
    "compute_insights",
]
