"""Top-level package for TNFR × LFS.

This package exposes utility functions to ingest telemetry, compute
Event Performance Indicators (EPI), and derive recommendations for
improving tyre normal force ratio (ΔNFR) and stability index (ΔSi).

The Python sources now live under :mod:`src.tnfr_lfs`, matching the
modern layout adopted by the project.
"""

from ._version import __version__
from .core.coherence import compute_node_delta_nfr, sense_index
from .core.epi import (
    DeltaCalculator,
    EPIExtractor,
    NaturalFrequencyAnalyzer,
    NaturalFrequencySettings,
    TelemetryRecord,
    delta_nfr_by_node,
)
from .core.epi_models import EPIBundle
from .core.segmentation import Goal, Microsector, segment_microsectors
from .ingestion.live import OutSimClient
from .recommender.rules import Recommendation, RecommendationEngine
from .processing import InsightsResult, compute_insights
from .exporters import exporters_registry
from .config_loader import (
    Car,
    Profile,
    example_pipeline,
    load_cars,
    load_profiles,
    resolve_targets,
)
from .track_loader import (
    Track,
    TrackConfig,
    assemble_session_weights,
    load_modifiers,
    load_track,
    load_track_profiles,
)

__all__ = [
    "TelemetryRecord",
    "EPIExtractor",
    "NaturalFrequencyAnalyzer",
    "NaturalFrequencySettings",
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
    "InsightsResult",
    "compute_insights",
    "exporters_registry",
    "Car",
    "Profile",
    "load_cars",
    "load_profiles",
    "resolve_targets",
    "example_pipeline",
    "Track",
    "TrackConfig",
    "load_track",
    "load_track_profiles",
    "load_modifiers",
    "assemble_session_weights",
    "__version__",
]
