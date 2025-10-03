"""Top-level package for TNFR-LFS.

This package exposes utility functions to ingest telemetry, compute
Event Performance Indicators (EPI), and derive recommendations for
improving tyre normal force ratio (ΔNFR) and stability index (ΔSi).
"""

from .core.coherence import compute_node_delta_nfr, sense_index
from .core.epi import DeltaCalculator, EPIExtractor, TelemetryRecord, delta_nfr_by_node
from .core.epi_models import EPIBundle
from .core.segmentation import Goal, Microsector, segment_microsectors
from .acquisition.outsim_client import OutSimClient
from .recommender.rules import Recommendation, RecommendationEngine
from .exporters import exporters_registry

__all__ = [
    "TelemetryRecord",
    "EPIExtractor",
    "DeltaCalculator",
    "EPIBundle",
    "delta_nfr_by_node",
    "Goal",
    "Microsector",
    "segment_microsectors",
    "compute_node_delta_nfr",
    "sense_index",
    "OutSimClient",
    "Recommendation",
    "RecommendationEngine",
    "exporters_registry",
]
