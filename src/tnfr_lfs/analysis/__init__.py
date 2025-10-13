"""Analysis helpers for TNFR × LFS."""

from tnfr_lfs.analysis.abtest import ABResult, SUPPORTED_LAP_METRICS, ab_compare_by_lap
from tnfr_lfs.analysis.brake_thermal import BrakeThermalConfig, BrakeThermalEstimator
from tnfr_lfs.analysis.insights import InsightsResult, compute_insights
from tnfr_lfs.analysis.robustness import compute_session_robustness

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
