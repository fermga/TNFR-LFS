"""Top-level package for TNFR-LFS.

This package exposes utility functions to ingest telemetry, compute
Event Performance Indicators (EPI), and derive recommendations for
improving tyre normal force ratio (ΔNFR) and stability index (ΔSi).
"""

from .core.epi import (
    TelemetryRecord,
    EPIResult,
    EPIExtractor,
    DeltaCalculator,
    compute_coherence,
)
from .acquisition.outsim_client import OutSimClient
from .recommender.rules import Recommendation, RecommendationEngine
from .exporters import exporters_registry

__all__ = [
    "TelemetryRecord",
    "EPIResult",
    "EPIExtractor",
    "DeltaCalculator",
    "compute_coherence",
    "OutSimClient",
    "Recommendation",
    "RecommendationEngine",
    "exporters_registry",
]
