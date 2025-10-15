from __future__ import annotations

import inspect

import pytest

from tnfr_lfs.analysis import compute_session_robustness
from tnfr_lfs.analysis import robustness as robustness_module

from tnfr_core.epi import DeltaCalculator, EPIExtractor
from tnfr_core.metrics import segmentation as segmentation_module
from tnfr_core.segmentation import segment_microsectors
from tnfr_lfs.analysis.insights import InsightsResult, compute_insights

from tests.helpers import preloaded_profile_manager


def test_compute_insights_matches_components(
    synthetic_records, tmp_path
) -> None:
    extractor = EPIExtractor()
    reference_records = list(synthetic_records)
    expected_bundles = extractor.extract(reference_records)
    baseline = extractor.baseline_record
    assert baseline is not None
    manager = preloaded_profile_manager(tmp_path)
    result = compute_insights(
        list(reference_records),
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
    overrides_payload = overrides if overrides else None
    expected_microsectors = segment_microsectors(
        list(reference_records),
        result.bundles,
        phase_weight_overrides=overrides_payload,
        baseline=baseline,
    )
    assert result.microsectors == expected_microsectors
    tuple_microsectors = segment_microsectors(
        tuple(reference_records),
        list(result.bundles),
        phase_weight_overrides=overrides_payload,
        baseline=baseline,
    )
    assert tuple_microsectors == expected_microsectors
    expected_robustness = compute_session_robustness(
        result.bundles,
        microsectors=expected_microsectors,
        thresholds=getattr(result.thresholds, "robustness", None),
    )
    assert result.robustness == expected_robustness


def test_compute_session_robustness_vectorised_matches_scalar(
    synthetic_records, tmp_path, monkeypatch
) -> None:
    numpy = pytest.importorskip("numpy")
    original_np = robustness_module.np

    extractor = EPIExtractor()
    reference_records = list(synthetic_records)
    bundles = extractor.extract(reference_records)
    baseline = extractor.baseline_record
    assert baseline is not None
    microsectors = segment_microsectors(
        reference_records,
        bundles,
        baseline=baseline,
    )

    thresholds = {
        "lap": {"delta_nfr": 0.5, "sense_index": 0.5},
        "phase": {"delta_nfr": 0.4, "sense_index": 0.4},
    }

    vectorised = compute_session_robustness(
        bundles,
        lap_indices=list(range(len(bundles))),
        microsectors=microsectors,
        thresholds=thresholds,
    )

    monkeypatch.setattr(robustness_module, "np", None)
    try:
        scalar = compute_session_robustness(
            bundles,
            lap_indices=list(range(len(bundles))),
            microsectors=microsectors,
            thresholds=thresholds,
        )
    finally:
        restored = original_np if original_np is not None else numpy
        monkeypatch.setattr(robustness_module, "np", restored)

    assert vectorised == scalar


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


def test_compute_insights_preserves_microsector_goals(
    synthetic_records, tmp_path
) -> None:
    manager = preloaded_profile_manager(tmp_path)
    reference_records = list(synthetic_records)
    first = compute_insights(
        list(reference_records),
        car_model="FZR",
        track_name="generic",
        profile_manager=manager,
    )
    second = compute_insights(
        list(reference_records),
        car_model="FZR",
        track_name="generic",
        profile_manager=manager,
    )
    assert len(first.microsectors) == len(second.microsectors)
    first_goals = [microsector.goals for microsector in first.microsectors]
    second_goals = [microsector.goals for microsector in second.microsectors]
    assert first_goals == second_goals


def test_compute_insights_reuses_sample_rate(
    synthetic_records, tmp_path, monkeypatch
) -> None:
    manager = preloaded_profile_manager(tmp_path)
    reference_records = list(synthetic_records)

    baseline = compute_insights(
        list(reference_records),
        car_model="FZR",
        track_name="generic",
        profile_manager=manager,
    )

    call_counter = {"count": 0}
    original = segmentation_module.estimate_sample_rate

    def counting(records):  # type: ignore[override]
        call_counter["count"] += 1
        return original(records)

    monkeypatch.setattr(segmentation_module, "estimate_sample_rate", counting)

    repeated = compute_insights(
        list(reference_records),
        car_model="FZR",
        track_name="generic",
        profile_manager=manager,
    )

    assert repeated == baseline
    assert call_counter["count"] == 1


def test_segment_microsectors_phase_samples_match_boundaries(
    synthetic_records,
) -> None:
    extractor = EPIExtractor()
    reference_records = list(synthetic_records)
    bundles = extractor.extract(reference_records)
    baseline = extractor.baseline_record
    assert baseline is not None
    microsectors = segment_microsectors(
        reference_records,
        bundles,
        baseline=baseline,
    )
    assert microsectors
    for microsector in microsectors:
        for phase, indices in microsector.phase_samples.items():
            assert tuple(indices) == tuple(microsector.phase_indices(phase))


def test_segment_microsectors_short_circuits_goal_cache(
    synthetic_records,
    synthetic_bundles,
    monkeypatch,
) -> None:
    baseline = DeltaCalculator.derive_baseline(synthetic_records)
    assert baseline is not None

    source_lines, start_line = inspect.getsourcelines(segmentation_module.segment_microsectors)
    call_sites = [
        start_line + offset
        for offset, line in enumerate(source_lines)
        if "_phase_nu_f_targets(" in line
    ]
    assert call_sites, "segment_microsectors should call _phase_nu_f_targets"
    goal_loop_line = max(call_sites)

    original_phase_targets = segmentation_module._phase_nu_f_targets
    original_invalidate = segmentation_module._invalidate_goal_cache
    call_lines_baseline: list[int] = []
    call_lines_short: list[int] = []
    mode = "baseline"

    def tracking_phase_targets(*args, **kwargs):
        frame = inspect.stack()[1]
        if mode == "baseline":
            call_lines_baseline.append(frame.lineno)
        else:
            call_lines_short.append(frame.lineno)
        return original_phase_targets(*args, **kwargs)

    monkeypatch.setattr(
        segmentation_module,
        "_phase_nu_f_targets",
        tracking_phase_targets,
    )
    segment_microsectors(list(synthetic_records), list(synthetic_bundles), baseline=baseline)

    assert goal_loop_line in call_lines_baseline

    mode = "short"
    original_signature = segmentation_module._phase_gradient_signature
    monkeypatch.setattr(
        segmentation_module,
        "_invalidate_goal_cache",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(segmentation_module, "_phase_gradient_signature", lambda _: None)

    segment_microsectors(list(synthetic_records), list(synthetic_bundles), baseline=baseline)

    monkeypatch.setattr(segmentation_module, "_phase_gradient_signature", original_signature)
    monkeypatch.setattr(segmentation_module, "_phase_nu_f_targets", original_phase_targets)
    monkeypatch.setattr(segmentation_module, "_invalidate_goal_cache", original_invalidate)

    assert call_lines_short, "the profiling wrapper should record calls"

    baseline_calls = call_lines_baseline.count(goal_loop_line)
    short_circuit_calls = call_lines_short.count(goal_loop_line)
    assert baseline_calls > 0
    assert short_circuit_calls < baseline_calls
