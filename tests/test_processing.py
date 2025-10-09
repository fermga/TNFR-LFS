from __future__ import annotations

from tnfr_lfs.analysis import compute_session_robustness
from tnfr_lfs.core.epi import EPIExtractor
from tnfr_lfs.core.segmentation import segment_microsectors
from tnfr_lfs.processing import InsightsResult, compute_insights

from tests.profile_manager_helpers import preloaded_profile_manager


def test_compute_insights_matches_components(
    synthetic_records, tmp_path
) -> None:
    manager = preloaded_profile_manager(tmp_path)
    extractor = EPIExtractor()
    reference_records = list(synthetic_records)
    expected_bundles = extractor.extract(reference_records)
    result = compute_insights(
        list(synthetic_records),
        car_model="FZR",
        track_name="generic",
        profile_manager=manager,
    )
    assert isinstance(result, InsightsResult)
    assert len(result.bundles) == len(expected_bundles)
    assert result.bundles and expected_bundles
    assert type(result.bundles[0]) is type(expected_bundles[0])
    overrides = (
        result.snapshot.phase_weights
        if result.snapshot is not None
        else result.thresholds.phase_weights
    )
    expected_microsectors = segment_microsectors(
        list(synthetic_records),
        result.bundles,
        phase_weight_overrides=overrides if overrides else None,
    )
    assert result.microsectors == expected_microsectors
    expected_robustness = compute_session_robustness(
        result.bundles,
        microsectors=expected_microsectors,
        thresholds=getattr(result.thresholds, "robustness", None),
    )
    assert result.robustness == expected_robustness


def test_compute_insights_handles_empty_records(tmp_path) -> None:
    manager = preloaded_profile_manager(tmp_path)
    result = compute_insights(
        [],
        car_model="FZR",
        track_name="generic",
        profile_manager=manager,
    )
    assert result.bundles == []
    assert result.microsectors == []
    assert result.robustness == {}


def test_insights_with_robustness_override(
    synthetic_records, tmp_path
) -> None:
    manager = preloaded_profile_manager(tmp_path)
    result = compute_insights(
        list(synthetic_records),
        car_model="FZR",
        track_name="generic",
        profile_manager=manager,
    )
    lap_indices = list(range(len(result.bundles)))
    updated = result.with_robustness(lap_indices=lap_indices)
    thresholds = getattr(result.thresholds, "robustness", None)
    expected = compute_session_robustness(
        result.bundles,
        lap_indices=lap_indices,
        microsectors=result.microsectors,
        thresholds=thresholds,
    )
    assert updated.robustness == expected
    assert result.robustness != expected
