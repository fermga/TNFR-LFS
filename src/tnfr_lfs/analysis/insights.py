"""High-level processing helpers for telemetry insights."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from time import monotonic
from typing import Any, Mapping, Optional, Sequence

from tnfr_lfs.analysis.robustness import compute_session_robustness
from tnfr_core.equations.epi import EPIExtractor, TelemetryRecord
from tnfr_core.equations.epi_models import EPIBundle
from tnfr_core.metrics.segmentation import Microsector, segment_microsectors
from tnfr_lfs.ingestion.offline import ProfileManager, ProfileSnapshot
from tnfr_lfs.recommender import RecommendationEngine
from tnfr_lfs.recommender.rules import ThresholdProfile

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InsightsResult:
    """Structured output produced by :func:`compute_insights`."""

    bundles: Sequence[EPIBundle]
    microsectors: Sequence[Microsector]
    thresholds: ThresholdProfile
    snapshot: Optional[ProfileSnapshot]
    robustness: Mapping[str, Any]

    def with_robustness(
        self,
        *,
        bundles: Sequence[EPIBundle] | None = None,
        lap_indices: Sequence[int] | None = None,
        lap_metadata: Sequence[Mapping[str, Any]] | None = None,
        microsectors: Sequence[Microsector] | None = None,
        thresholds: Mapping[str, Mapping[str, float]] | None = None,
    ) -> "InsightsResult":
        """Return a new instance with robustness metrics recomputed."""

        reference_microsectors: Sequence[Microsector] = (
            microsectors if microsectors is not None else self.microsectors
        )
        thresholds_payload: Mapping[str, Mapping[str, float]] | None = thresholds
        if thresholds_payload is None:
            thresholds_payload = getattr(self.thresholds, "robustness", None)
        robustness_payload = compute_session_robustness(
            bundles or self.bundles,
            lap_indices=lap_indices,
            lap_metadata=lap_metadata,
            microsectors=reference_microsectors,
            thresholds=thresholds_payload,
        )
        return replace(self, robustness=robustness_payload)


def compute_insights(
    records: Sequence[TelemetryRecord],
    *,
    car_model: str,
    track_name: str,
    engine: RecommendationEngine | None = None,
    profile_manager: ProfileManager | None = None,
    robustness_bundles: Sequence[EPIBundle] | None = None,
    robustness_lap_indices: Sequence[int] | None = None,
    robustness_lap_metadata: Sequence[Mapping[str, Any]] | None = None,
    robustness_thresholds: Mapping[str, Mapping[str, float]] | None = None,
) -> InsightsResult:
    """Compute EPI bundles, microsectors and robustness metrics for a stint."""

    started = monotonic()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Computing insights",
            extra={
                "car_model": car_model,
                "track_name": track_name,
                "record_count": len(records),
            },
        )
    engine = engine or RecommendationEngine(
        car_model=car_model,
        track_name=track_name,
        profile_manager=profile_manager,
    )
    base_profile = engine._lookup_profile(car_model, track_name)
    snapshot: ProfileSnapshot | None = None
    if profile_manager is not None:
        session_payload = getattr(engine, "session", None)
        snapshot = profile_manager.resolve(
            car_model, track_name, base_profile, session=session_payload
        )
        thresholds = snapshot.thresholds
    else:
        thresholds = base_profile
    if not records:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "No records provided; returning empty insights",
                extra={
                    "car_model": car_model,
                    "track_name": track_name,
                    "duration": monotonic() - started,
                },
            )
        return InsightsResult(
            bundles=[],
            microsectors=[],
            thresholds=thresholds,
            snapshot=snapshot,
            robustness={},
        )
    extractor = EPIExtractor()
    bundles = extractor.extract(records)
    if not bundles:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "EPI extraction returned no bundles",
                extra={
                    "car_model": car_model,
                    "track_name": track_name,
                    "duration": monotonic() - started,
                },
            )
        return InsightsResult(
            bundles=bundles,
            microsectors=[],
            thresholds=thresholds,
            snapshot=snapshot,
            robustness={},
        )
    overrides = snapshot.phase_weights if snapshot is not None else thresholds.phase_weights
    microsectors = segment_microsectors(
        records,
        bundles,
        phase_weight_overrides=overrides if overrides else None,
    )
    robustness_reference = (
        robustness_thresholds
        if robustness_thresholds is not None
        else getattr(thresholds, "robustness", None)
    )
    robustness_payload = compute_session_robustness(
        robustness_bundles or bundles,
        lap_indices=robustness_lap_indices,
        lap_metadata=robustness_lap_metadata,
        microsectors=microsectors,
        thresholds=robustness_reference,
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Insight computation complete",
            extra={
                "car_model": car_model,
                "track_name": track_name,
                "bundle_count": len(bundles),
                "microsector_count": len(microsectors),
                "duration": monotonic() - started,
            },
        )
    return InsightsResult(
        bundles=bundles,
        microsectors=microsectors,
        thresholds=thresholds,
        snapshot=snapshot,
        robustness=robustness_payload,
    )


__all__ = ["InsightsResult", "compute_insights"]
