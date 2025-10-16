import pytest

import math
from dataclasses import replace
from statistics import mean, pstdev
from typing import List, Mapping, Sequence, Tuple

from tests.helpers import build_dynamic_record

from tnfr_core.archetypes import archetype_phase_targets
from tnfr_core import segmentation as segmentation_module
from tnfr_core.contextual_delta import (
    ContextFactors,
    apply_contextual_delta,
    load_context_matrix,
)
from tnfr_core.coherence import sense_index
from tnfr_core.epi import (
    DEFAULT_PHASE_WEIGHTS,
    DeltaCalculator,
    EPIExtractor,
    NaturalFrequencySettings,
    TelemetryRecord,
    delta_nfr_by_node,
    resolve_nu_f_by_node,
)
from tnfr_core.phases import PHASE_SEQUENCE, expand_phase_alias, phase_family
from tnfr_core.metrics import (
    AeroCoherence,
    BrakeHeadroom,
    BumpstopHistogram,
    LockingWindowScore,
    SlideCatchBudget,
    SuspensionVelocityBands,
    WindowMetrics,
)
from tnfr_core.operator_detection import silence_event_payloads
from tnfr_core.operators import cache as cache_module
from tnfr_core.operators.interfaces import SupportsTelemetrySample
from tnfr_core.metrics import segmentation as metrics_segmentation
from tnfr_core.segmentation import (
    Microsector,
    detect_quiet_microsector_streaks,
    microsector_stability_metrics,
    segment_microsectors,
)


def test_resolve_segment_adjustments_with_range_metadata() -> None:
    metadata = (1, 3)
    delta_series = [0.5, 1.0, 1.5, 2.0]
    sample_multipliers = [2.0, 2.0, 2.0, 2.0]
    default_adjusted_delta = [
        delta_series[i] * sample_multipliers[i] for i in range(len(delta_series))
    ]
    sense_series = [0.1, 0.2, 0.3, 0.4]

    adjusted, delta_sum, si_sum, count = segmentation_module._resolve_segment_adjustments(
        metadata,
        start=1,
        end=3,
        default_adjusted_delta=default_adjusted_delta,
        delta_series=delta_series,
        sense_series=sense_series,
        sample_multipliers=sample_multipliers,
    )

    assert adjusted == (2.0, 3.0, 4.0)
    assert delta_sum == pytest.approx(9.0)
    assert si_sum == pytest.approx(0.9)
    assert count == 3


def test_resolve_segment_adjustments_with_mapping_overrides() -> None:
    metadata = {2: 0.5}
    delta_series = [0.5, 1.0, 1.5, 2.0]
    default_adjusted_delta = delta_series[:]
    sense_series = [0.1, 0.2, 0.3, 0.4]
    sample_multipliers: List[float | None] = [None, None, None, None]

    adjusted, delta_sum, si_sum, count = segmentation_module._resolve_segment_adjustments(
        metadata,
        start=1,
        end=3,
        default_adjusted_delta=default_adjusted_delta,
        delta_series=delta_series,
        sense_series=sense_series,
        sample_multipliers=sample_multipliers,
    )

    assert adjusted == pytest.approx((1.0, 0.75, 2.0))
    assert delta_sum == pytest.approx(3.75)
    assert si_sum == pytest.approx(0.9)
    assert count == 3


def test_segment_microsectors_creates_goals_with_stable_assignments(
    synthetic_microsectors,
):
    assert len(synthetic_microsectors) == 2
    for microsector in synthetic_microsectors:
        assert isinstance(microsector, Microsector)
        assert microsector.brake_event is True
        assert microsector.support_event is True
        assert microsector.grip_rel >= 0.0
        assert set(microsector.filtered_measures) >= {
            "thermal_load",
            "style_index",
            "grip_rel",
            "support_effective",
            "load_support_ratio",
            "structural_expansion_longitudinal",
            "structural_contraction_longitudinal",
            "structural_expansion_lateral",
            "structural_contraction_lateral",
            "ackermann_parallel_index",
            "locking_window_score",
            "locking_window_score_on",
            "locking_window_score_off",
            "locking_window_transitions",
        }
        assert "si_variance" in microsector.filtered_measures
        assert "epi_derivative_abs" in microsector.filtered_measures
        assert "aero_medium_imbalance" in microsector.filtered_measures
        assert "aero_high_front_total" in microsector.filtered_measures
        assert "delta_nfr_std" in microsector.filtered_measures
        assert "nodal_delta_nfr_std" in microsector.filtered_measures
        assert "delta_nfr_entropy" in microsector.filtered_measures
        assert "node_entropy" in microsector.filtered_measures
        assert microsector.delta_nfr_std == pytest.approx(
            float(microsector.filtered_measures["delta_nfr_std"]), rel=1e-6
        )
        assert microsector.nodal_delta_nfr_std == pytest.approx(
            float(microsector.filtered_measures["nodal_delta_nfr_std"]), rel=1e-6
        )
        assert microsector.phase_delta_nfr_std
        assert microsector.phase_nodal_delta_nfr_std
        assert microsector.phase_delta_nfr_entropy
        assert microsector.phase_node_entropy
        assert any(
            key.startswith("delta_nfr_std_") for key in microsector.filtered_measures
        )
        assert any(
            key.startswith("nodal_delta_nfr_std_")
            for key in microsector.filtered_measures
        )
        assert {"entry1", "entry"}.issubset(microsector.phase_samples)
        assert {"apex3a", "apex"}.issubset(microsector.phase_samples)
        assert {"exit4", "exit"}.issubset(microsector.phase_samples)
        assert {"entry1", "entry"}.issubset(microsector.phase_axis_targets)
        assert {"apex3a", "apex"}.issubset(microsector.phase_axis_targets)
        assert {"entry1", "entry"}.issubset(microsector.phase_axis_weights)
        assert {"apex3a", "apex"}.issubset(microsector.phase_axis_weights)
        assert {
            "delta_nfr_std_entry1",
            "delta_nfr_std_entry",
        }.issubset(microsector.filtered_measures)
        assert {
            "nodal_delta_nfr_std_entry1",
            "nodal_delta_nfr_std_entry",
        }.issubset(microsector.filtered_measures)
        assert {
            "delta_nfr_entropy_entry1",
            "delta_nfr_entropy_entry",
        }.issubset(microsector.filtered_measures)
        assert {
            "node_entropy_entry1",
            "node_entropy_entry",
        }.issubset(microsector.filtered_measures)
        assert isinstance(microsector.recursivity_trace, tuple)
        assert microsector.last_mutation is None
        phases = [goal.phase for goal in microsector.goals]
        assert phases == list(PHASE_SEQUENCE)
        assert microsector.active_phase in phases
        boundaries = [microsector.phase_indices(phase) for phase in phases]
        seen = set()
        for phase_range in boundaries:
            assert phase_range.stop >= phase_range.start
            if phase_range.stop > phase_range.start:
                seen.update(phase_range)
        assert seen
        assert seen == set(range(min(seen), max(seen) + 1))
        assert all(goal.description for goal in microsector.goals)
        assert all(0.0 <= goal.target_sense_index <= 1.0 for goal in microsector.goals)
        intensity = {
            goal.phase: abs(goal.target_delta_nfr) + goal.nu_f_target for goal in microsector.goals
        }
        dominant = max(intensity, key=intensity.get)
        assert microsector.active_phase == dominant
        for goal in microsector.goals:
            assert goal.nu_f_target >= 0.0
            assert goal.slip_lat_window[0] <= goal.slip_lat_window[1]
            assert goal.slip_long_window[0] <= goal.slip_long_window[1]
            assert goal.yaw_rate_window[0] <= goal.yaw_rate_window[1]
            assert goal.dominant_nodes == microsector.dominant_nodes[goal.phase]
            assert -math.pi <= goal.target_phase_lag <= math.pi
            assert -1.0 <= goal.target_phase_alignment <= 1.0
            assert 0.0 <= goal.target_phase_synchrony <= 1.0
            assert -math.pi <= goal.measured_phase_lag <= math.pi
            assert -1.0 <= goal.measured_phase_alignment <= 1.0
            assert 0.0 <= goal.measured_phase_synchrony <= 1.0
            indices = list(microsector.phase_indices(goal.phase))
            if indices:
                assert goal.dominant_nodes
        assert microsector.phase_lag
        assert microsector.phase_alignment
        assert microsector.phase_synchrony
        assert set(microsector.phase_lag) >= set(PHASE_SEQUENCE)
        assert set(microsector.phase_alignment) >= set(PHASE_SEQUENCE)
        assert set(microsector.phase_synchrony) >= set(PHASE_SEQUENCE)
        if microsector.operator_events:
            for payloads in microsector.operator_events.values():
                for payload in payloads:
                    if isinstance(payload, Mapping):
                        assert "si_variance" in payload
                        assert "epi_derivative_abs" in payload


def test_segment_microsectors_uses_supplied_baseline(
    monkeypatch: "pytest.MonkeyPatch",
    synthetic_records: Sequence[TelemetryRecord],
    synthetic_bundles,
) -> None:
    baseline = DeltaCalculator.derive_baseline(synthetic_records)

    def _fail(*_args, **_kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("derive_baseline should not be invoked")

    monkeypatch.setattr(metrics_segmentation.DeltaCalculator, "derive_baseline", _fail)

    result = metrics_segmentation.segment_microsectors(
        synthetic_records,
        synthetic_bundles,
        baseline=baseline,
    )
    assert result


def test_segment_microsectors_incremental_recompute_matches_full(
    synthetic_records, synthetic_bundles, monkeypatch
) -> None:
    original_recompute = metrics_segmentation._recompute_bundles

    incremental_result = segment_microsectors(
        list(synthetic_records),
        list(synthetic_bundles),
        operator_state={},
    )

    def force_full_recompute(
        records,
        bundles,
        baseline,
        phase_assignments,
        weight_lookup,
        goal_nu_f_lookup=None,
        *,
        start_index=0,
        analyzer_states=None,
    ):
        return original_recompute(
            records,
            bundles,
            baseline,
            phase_assignments,
            weight_lookup,
            goal_nu_f_lookup=goal_nu_f_lookup,
            start_index=0,
            analyzer_states=None,
        )

    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            metrics_segmentation,
            "_recompute_bundles",
            force_full_recompute,
        )
        full_result = segment_microsectors(
            list(synthetic_records),
            list(synthetic_bundles),
            operator_state={},
        )

    assert incremental_result == full_result


def test_recompute_bundles_partial_reuse_reduces_resolve_calls(
    synthetic_records, synthetic_bundles, monkeypatch
) -> None:
    records = list(synthetic_records[:80])
    bundles = list(synthetic_bundles[: len(records)])
    baseline = DeltaCalculator.derive_baseline(records)
    phase_assignments = [PHASE_SEQUENCE[0] for _ in range(len(records))]
    weight_lookup = [DEFAULT_PHASE_WEIGHTS for _ in range(len(records))]

    base_result = metrics_segmentation._recompute_bundles(
        records,
        bundles,
        baseline,
        phase_assignments,
        weight_lookup,
    )
    base_bundles = list(base_result.bundles)
    base_states = list(base_result.analyzer_states)

    change_index = len(records) // 2
    adjusted_lookup = list(weight_lookup)
    for idx in range(change_index, len(records)):
        adjusted_lookup[idx] = {"__default__": 1.5}

    reference_result = metrics_segmentation._recompute_bundles(
        records,
        list(base_bundles),
        baseline,
        phase_assignments,
        adjusted_lookup,
    )
    reference_bundles = list(reference_result.bundles)

    resolve_call_count = 0
    original_resolve = metrics_segmentation.resolve_nu_f_by_node

    def _counting_resolve(*args, **kwargs):
        nonlocal resolve_call_count
        resolve_call_count += 1
        return original_resolve(*args, **kwargs)

    with monkeypatch.context() as patch_context:
        resolve_call_count = 0
        patch_context.setattr(
            metrics_segmentation,
            "resolve_nu_f_by_node",
            _counting_resolve,
        )
        naive_result = metrics_segmentation._recompute_bundles(
            records,
            list(base_bundles),
            baseline,
            phase_assignments,
            adjusted_lookup,
            start_index=change_index,
            analyzer_states=None,
        )
        naive_calls = resolve_call_count

    assert list(naive_result.bundles) == reference_bundles
    assert naive_calls == len(records)

    with monkeypatch.context() as patch_context:
        resolve_call_count = 0
        patch_context.setattr(
            metrics_segmentation,
            "resolve_nu_f_by_node",
            _counting_resolve,
        )
        cached_result = metrics_segmentation._recompute_bundles(
            records,
            list(base_bundles),
            baseline,
            phase_assignments,
            adjusted_lookup,
            start_index=change_index,
            analyzer_states=base_states,
        )
        cached_calls = resolve_call_count

    assert list(cached_result.bundles) == reference_bundles
    assert cached_calls < naive_calls
    assert cached_calls == len(records) - change_index


def test_recompute_bundles_reprojects_without_weight_changes(
    synthetic_records, synthetic_bundles, monkeypatch
) -> None:
    records = list(synthetic_records[:80])
    bundles = list(synthetic_bundles[: len(records)])
    baseline = DeltaCalculator.derive_baseline(records)
    phase_assignments = [PHASE_SEQUENCE[0] for _ in range(len(records))]
    weight_lookup = [DEFAULT_PHASE_WEIGHTS for _ in range(len(records))]

    base_result = metrics_segmentation._recompute_bundles(
        records,
        bundles,
        baseline,
        phase_assignments,
        weight_lookup,
    )
    base_bundles = list(base_result.bundles)
    base_states = list(base_result.analyzer_states)

    start_index = max(1, len(records) // 3)

    compute_calls = 0
    reproject_calls = 0
    original_compute = metrics_segmentation.DeltaCalculator.compute_bundle
    original_reproject = metrics_segmentation.DeltaCalculator.reproject_bundle_phase

    with monkeypatch.context() as patch_context:

        def _tracking_compute(*args, **kwargs):
            nonlocal compute_calls
            compute_calls += 1
            return original_compute(*args, **kwargs)

        def _tracking_reproject(bundle, **kwargs):
            nonlocal reproject_calls
            reproject_calls += 1
            return original_reproject(bundle, **kwargs)

        patch_context.setattr(
            metrics_segmentation.DeltaCalculator,
            "compute_bundle",
            staticmethod(_tracking_compute),
        )
        patch_context.setattr(
            metrics_segmentation.DeltaCalculator,
            "reproject_bundle_phase",
            staticmethod(_tracking_reproject),
        )

        recomputed = metrics_segmentation._recompute_bundles(
            records,
            list(base_bundles),
            baseline,
            phase_assignments,
            weight_lookup,
            start_index=start_index,
            analyzer_states=base_states,
        )

    assert list(recomputed.bundles) == base_bundles
    assert compute_calls == 0
    assert reproject_calls == len(records) - start_index


def test_recompute_bundles_reprojects_with_phase_overrides(
    synthetic_records, synthetic_bundles, monkeypatch
) -> None:
    records = list(synthetic_records[:80])
    bundles = list(synthetic_bundles[: len(records)])
    baseline = DeltaCalculator.derive_baseline(records)
    phase_assignments = [PHASE_SEQUENCE[0] for _ in range(len(records))]
    weight_lookup = [DEFAULT_PHASE_WEIGHTS for _ in range(len(records))]

    base_result = metrics_segmentation._recompute_bundles(
        records,
        bundles,
        baseline,
        phase_assignments,
        weight_lookup,
    )
    base_bundles = list(base_result.bundles)
    base_states = list(base_result.analyzer_states)

    change_index = len(records) // 2
    adjusted_lookup = list(weight_lookup)
    for idx in range(change_index, len(records)):
        adjusted_lookup[idx] = {"__default__": 1.5}

    reference_result = metrics_segmentation._recompute_bundles(
        records,
        list(base_bundles),
        baseline,
        phase_assignments,
        adjusted_lookup,
    )

    compute_calls = 0
    reproject_calls = 0
    original_compute = metrics_segmentation.DeltaCalculator.compute_bundle
    original_reproject = metrics_segmentation.DeltaCalculator.reproject_bundle_phase

    with monkeypatch.context() as patch_context:

        def _tracking_compute(*args, **kwargs):
            nonlocal compute_calls
            compute_calls += 1
            return original_compute(*args, **kwargs)

        def _tracking_reproject(bundle, **kwargs):
            nonlocal reproject_calls
            reproject_calls += 1
            return original_reproject(bundle, **kwargs)

        patch_context.setattr(
            metrics_segmentation.DeltaCalculator,
            "compute_bundle",
            staticmethod(_tracking_compute),
        )
        patch_context.setattr(
            metrics_segmentation.DeltaCalculator,
            "reproject_bundle_phase",
            staticmethod(_tracking_reproject),
        )

        recomputed = metrics_segmentation._recompute_bundles(
            records,
            list(base_bundles),
            baseline,
            phase_assignments,
            adjusted_lookup,
            start_index=change_index,
            analyzer_states=base_states,
        )

    assert list(recomputed.bundles) == list(reference_result.bundles)
    assert compute_calls == 0
    assert reproject_calls == len(records) - change_index


def test_recompute_bundles_restored_history_invalidates_dynamic_cache(
    synthetic_records, synthetic_bundles, monkeypatch
) -> None:
    records = list(synthetic_records[:60])
    bundles = list(synthetic_bundles[: len(records)])
    baseline = DeltaCalculator.derive_baseline(records)
    phase_assignments = [PHASE_SEQUENCE[0] for _ in range(len(records))]
    weight_lookup = [DEFAULT_PHASE_WEIGHTS for _ in range(len(records))]

    record_index = {id(record): idx for idx, record in enumerate(records)}

    class _TinyWindowAnalyzer(metrics_segmentation.NaturalFrequencyAnalyzer):
        instances: list["_TinyWindowAnalyzer"] = []

        def __init__(self) -> None:  # pragma: no cover - exercised via recompute
            settings = NaturalFrequencySettings(
                min_window_seconds=0.05,
                max_window_seconds=0.1,
            )
            super().__init__(settings)
            self.removed_records: list[SupportsTelemetrySample] = []
            _TinyWindowAnalyzer.instances.append(self)

        def _append_record(self, record):  # pragma: no cover - delegated to super
            before = tuple(self._history)
            super()._append_record(record)
            if not before:
                return
            remaining = set(self._history)
            for sample in before:
                if sample not in remaining:
                    self.removed_records.append(sample)

    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            metrics_segmentation,
            "NaturalFrequencyAnalyzer",
            _TinyWindowAnalyzer,
        )

        invalidated: list[int] = []
        original_invalidate = cache_module.invalidate_dynamic_record

        probe = metrics_segmentation.NaturalFrequencyAnalyzer()
        assert probe._dynamic_cache_active()
        _TinyWindowAnalyzer.instances.clear()

        def _tracking_invalidate(record):
            invalidated.append(record_index.get(id(record), -1))
            return original_invalidate(record)

        patch_context.setattr(
            cache_module,
            "invalidate_dynamic_record",
            _tracking_invalidate,
        )

        base_result = metrics_segmentation._recompute_bundles(
            records,
            bundles,
            baseline,
            phase_assignments,
            weight_lookup,
        )
        base_analyzer = _TinyWindowAnalyzer.instances[-1]
        base_removed = [
            record_index.get(id(sample), -1)
            for sample in base_analyzer.removed_records
        ]
        base_invalidations = [index for index in invalidated if index >= 0]
        assert any(index >= 0 for index in base_removed)
        if base_invalidations:
            assert base_invalidations[0] == base_removed[0]
        base_states = list(base_result.analyzer_states)

        change_index = max(2, len(records) // 2)
        state = base_states[change_index - 1]
        assert state is not None and state.history_start is not None

        invalidated.clear()
        _TinyWindowAnalyzer.instances.clear()
        cached_result = metrics_segmentation._recompute_bundles(
            records,
            list(base_result.bundles),
            baseline,
            phase_assignments,
            weight_lookup,
            start_index=change_index,
            analyzer_states=base_states,
        )
        cached_analyzer = _TinyWindowAnalyzer.instances[-1]
        cached_removed = [
            record_index.get(id(sample), -1)
            for sample in cached_analyzer.removed_records
        ]

        assert cached_result.bundles == base_result.bundles
        observed = [index for index in invalidated if index >= 0]
        assert cached_removed, "expected samples to be dropped from analyzer history"
        first_removed = next(index for index in cached_removed if index >= 0)
        if observed:
            assert observed[0] == state.history_start
        assert first_removed == state.history_start


def test_segment_microsectors_caches_delta_signature(
    synthetic_records, synthetic_bundles, monkeypatch
) -> None:
    captured_specs = []
    original_adjust = metrics_segmentation._adjust_phase_weights_with_dominance

    def _wrapped_adjust(specs, bundles, records, **kwargs):
        captured_specs.extend(specs)
        return original_adjust(specs, bundles, records, **kwargs)

    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            metrics_segmentation,
            "_adjust_phase_weights_with_dominance",
            _wrapped_adjust,
        )
        microsectors = segment_microsectors(
            list(synthetic_records), list(synthetic_bundles)
        )

    assert captured_specs, "expected cached specifications"
    assert len(captured_specs) == len(microsectors)
    for spec, microsector in zip(captured_specs, microsectors):
        assert "adjusted_deltas" in spec
        assert "delta_signature" in spec
        assert "avg_si" in spec
        adjusted = tuple(spec["adjusted_deltas"])
        if adjusted:
            assert spec["delta_signature"] == pytest.approx(mean(adjusted))
        assert microsector.delta_nfr_signature == pytest.approx(spec["delta_signature"])
        style_index = microsector.filtered_measures.get("style_index")
        assert style_index == pytest.approx(spec["avg_si"])


def test_segment_microsectors_partial_recompute_refreshes_metrics(
    synthetic_records, synthetic_bundles, monkeypatch
) -> None:
    forced_delta = -3.5
    forced_si = 0.11
    forced_nu_f = 42.0
    captured_specs: list[dict[str, object]] = []
    partial_triggered = False

    original_adjust = metrics_segmentation._adjust_phase_weights_with_dominance
    original_phase_targets = metrics_segmentation._phase_nu_f_targets
    original_recompute = metrics_segmentation._recompute_bundles

    def _capture_specs(specs, *args, **kwargs):
        captured_specs.clear()
        captured_specs.extend(specs)
        return original_adjust(specs, *args, **kwargs)

    def _forced_phase_targets(*args, **kwargs):
        dominant, nu_f_targets, dominance_weights, sample_lookup = original_phase_targets(
            *args, **kwargs
        )
        if isinstance(sample_lookup, Mapping):
            sample_indices = sample_lookup.keys()
        else:
            sample_indices = ()
        coerced_lookup = {int(idx): forced_nu_f for idx in sample_indices}
        if not coerced_lookup:
            boundaries = args[2] if len(args) >= 3 else {}
            if isinstance(boundaries, Mapping):
                for start, stop in boundaries.values():
                    for idx in range(start, stop):
                        coerced_lookup[int(idx)] = forced_nu_f
        return (
            dominant,
            dict(nu_f_targets),
            dict(dominance_weights),
            coerced_lookup,
        )

    def _mutating_recompute(
        records,
        bundles,
        baseline,
        phase_assignments,
        weight_lookup,
        goal_nu_f_lookup=None,
        **kwargs,
    ):
        nonlocal partial_triggered
        result = original_recompute(
            records,
            bundles,
            baseline,
            phase_assignments,
            weight_lookup,
            goal_nu_f_lookup=goal_nu_f_lookup,
            **kwargs,
        )
        if goal_nu_f_lookup:
            partial_triggered = True
            mutated = list(result.bundles)
            for idx, bundle in enumerate(mutated):
                if idx in goal_nu_f_lookup:
                    mutated[idx] = replace(
                        bundle,
                        delta_nfr=forced_delta,
                        sense_index=forced_si,
                    )
            result = metrics_segmentation._BundleRecomputeResult(
                mutated, result.analyzer_states
            )
        return result

    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            metrics_segmentation,
            "_adjust_phase_weights_with_dominance",
            _capture_specs,
        )
        patch_context.setattr(
            metrics_segmentation,
            "_phase_nu_f_targets",
            _forced_phase_targets,
        )
        patch_context.setattr(
            metrics_segmentation,
            "_recompute_bundles",
            _mutating_recompute,
        )
        patch_context.setattr(
            metrics_segmentation,
            "_resolve_context_multiplier",
            lambda *args, **kwargs: 1.0,
        )
        microsectors = segment_microsectors(
            list(synthetic_records), list(synthetic_bundles)
        )

    assert partial_triggered, "expected goal-driven recompute to run"
    assert captured_specs, "expected cached specifications"
    assert microsectors, "expected synthetic segmentation"

    spec = captured_specs[0]
    adjusted = tuple(spec.get("adjusted_deltas", ()))
    assert adjusted, "expected adjusted deltas to be recomputed"
    for value in adjusted:
        assert value == pytest.approx(forced_delta)
    assert spec.get("delta_signature") == pytest.approx(forced_delta)
    assert spec.get("avg_si") == pytest.approx(forced_si)

    first_microsector = microsectors[0]
    assert first_microsector.delta_nfr_signature == pytest.approx(forced_delta)
    style_index = first_microsector.filtered_measures.get("style_index")
    assert style_index == pytest.approx(forced_si)


def test_segment_microsectors_reuses_node_delta_cache(
    synthetic_records, synthetic_bundles, monkeypatch
) -> None:
    records = list(synthetic_records)
    bundles = list(synthetic_bundles)
    baseline = segment_microsectors(list(records), list(bundles))
    assert len(baseline) >= 2

    call_count = 0
    original = metrics_segmentation.delta_nfr_by_node

    def _wrapped(record):
        nonlocal call_count
        call_count += 1
        return original(record)

    with monkeypatch.context() as patch_context:
        patch_context.setattr(metrics_segmentation, "delta_nfr_by_node", _wrapped)
        result = segment_microsectors(list(records), list(bundles))

    assert result == baseline
    assert call_count == 0


def test_segment_microsectors_reuses_phase_nu_f_targets_cache(
    synthetic_records, synthetic_bundles, monkeypatch
) -> None:
    missing_cache_calls: list[dict[str, object]] = []
    original_build = metrics_segmentation._build_goals

    def _wrapped_build(*args, **kwargs):
        if kwargs.get("phase_gradients") is not None:
            if (
                kwargs.get("phase_dominant_nodes") is None
                or kwargs.get("phase_nu_f_targets") is None
                or kwargs.get("phase_dominance_weights") is None
            ):
                missing_cache_calls.append(dict(kwargs))
        return original_build(*args, **kwargs)

    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            metrics_segmentation,
            "_build_goals",
            _wrapped_build,
        )
        microsectors = segment_microsectors(
            list(synthetic_records), list(synthetic_bundles)
        )

    assert microsectors, "expected synthetic segmentation"
    assert not missing_cache_calls, "expected cached phase targets to be reused"


def test_segment_microsectors_node_delta_cache_consistency(
    synthetic_records, synthetic_bundles, monkeypatch
) -> None:
    original_phase_targets = metrics_segmentation._phase_nu_f_targets
    original_build_goals = metrics_segmentation._build_goals
    observed_delta_lengths: list[int] = []
    observed_nu_lengths: list[int] = []

    def _assert_delta_cache(
        bundles,
        cache,
        *,
        cache_window=None,
    ) -> None:
        if cache is None:
            return
        start = 0
        if cache_window is not None and isinstance(cache_window, tuple):
            if cache_window:
                start = min(cache_window)
        observed_delta_lengths.append(len(cache))
        for offset, distribution in enumerate(cache):
            bundle_index = start + offset
            if not (0 <= bundle_index < len(bundles)):
                continue
            expected = metrics_segmentation._bundle_node_delta(bundles[bundle_index])
            assert dict(distribution) == dict(expected)

    def _assert_nu_cache(
        bundles,
        cache,
        *,
        cache_window=None,
    ) -> None:
        if cache is None:
            return
        start = 0
        if cache_window is not None and isinstance(cache_window, tuple):
            if cache_window:
                start = min(cache_window)
        observed_nu_lengths.append(len(cache))
        for offset, distribution in enumerate(cache):
            bundle_index = start + offset
            if not (0 <= bundle_index < len(bundles)):
                continue
            expected = metrics_segmentation._bundle_node_nu_f(bundles[bundle_index])
            assert dict(distribution) == dict(expected)

    def _wrapped_phase_targets(*args, **kwargs):
        bundles = args[0]
        delta_cache = kwargs.get("node_delta_cache")
        nu_cache = kwargs.get("node_nu_cache")
        cache_window = kwargs.get("cache_window")
        _assert_delta_cache(bundles, delta_cache, cache_window=cache_window)
        _assert_nu_cache(bundles, nu_cache, cache_window=cache_window)
        return original_phase_targets(*args, **kwargs)

    def _wrapped_build_goals(*args, **kwargs):
        bundles = args[1]
        delta_cache = kwargs.get("node_delta_cache")
        nu_cache = kwargs.get("node_nu_cache")
        _assert_delta_cache(bundles, delta_cache)
        _assert_nu_cache(bundles, nu_cache)
        return original_build_goals(*args, **kwargs)

    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            metrics_segmentation,
            "_phase_nu_f_targets",
            _wrapped_phase_targets,
        )
        patch_context.setattr(
            metrics_segmentation,
            "_build_goals",
            _wrapped_build_goals,
        )
        microsectors = segment_microsectors(
            list(synthetic_records),
            list(synthetic_bundles),
            operator_state={},
        )

    assert observed_delta_lengths, "expected node delta cache to be provided"
    assert observed_nu_lengths, "expected node nu_f cache to be provided"
    assert microsectors, "expected synthetic segmentation"


def test_segment_microsectors_prefers_record_yaw_rate(
    synthetic_records, synthetic_bundles, monkeypatch
) -> None:
    calls: list[int] = []
    original_compute = metrics_segmentation._compute_yaw_rate

    def _tracking(*args, **kwargs):
        calls.append(kwargs.get("index") if "index" in kwargs else args[1])
        return original_compute(*args, **kwargs)

    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            metrics_segmentation,
            "_compute_yaw_rate",
            _tracking,
        )
        microsectors = segment_microsectors(
            list(synthetic_records),
            list(synthetic_bundles),
        )

    assert microsectors, "expected synthetic segmentation"
    assert calls == [], "expected direct yaw_rate usage without recomputation"


def test_segment_microsectors_computes_fallback_yaw_rate(
    synthetic_records, synthetic_bundles, monkeypatch
) -> None:
    fallback_indices = {0, 5, 12}
    records = list(synthetic_records)
    for index in fallback_indices:
        records[index] = replace(records[index], yaw_rate=math.nan)

    calls: list[int] = []
    original_compute = metrics_segmentation._compute_yaw_rate

    def _tracking(*args, **kwargs):
        call_index = kwargs.get("index") if "index" in kwargs else args[1]
        calls.append(call_index)
        return original_compute(*args, **kwargs)

    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            metrics_segmentation,
            "_compute_yaw_rate",
            _tracking,
        )
        microsectors = segment_microsectors(records, list(synthetic_bundles))

    assert microsectors, "expected synthetic segmentation"
    assert set(calls) >= fallback_indices


def test_segment_microsectors_computes_wheel_dispersion(
    synthetic_records, synthetic_bundles
) -> None:
    base_microsectors = segment_microsectors(synthetic_records, synthetic_bundles)
    assert base_microsectors, "expected at least one microsector"
    first = base_microsectors[0]
    indices: list[int] = sorted(
        {idx for samples in first.phase_samples.values() for idx in samples}
    )
    assert indices, "expected indices for synthetic microsector"
    records = list(synthetic_records)
    fl_temps: list[float] = []
    fr_temps: list[float] = []
    rl_temps: list[float] = []
    rr_temps: list[float] = []
    fl_pressures: list[float] = []
    fr_pressures: list[float] = []
    rl_pressures: list[float] = []
    rr_pressures: list[float] = []
    for offset, idx in enumerate(indices):
        record = records[idx]
        fl = 80.0 + offset
        fr = 78.5 + offset * 0.6
        rl = 77.0 + offset * 0.3
        rr = 76.5 + offset * 0.25
        pfl = 1.60 + (offset % 3) * 0.02
        pfr = 1.58 + (offset % 4) * 0.015
        prl = 1.52 + (offset % 2) * 0.01
        prr = 1.50 + (offset % 5) * 0.012
        records[idx] = replace(
            record,
            tyre_temp_fl=fl,
            tyre_temp_fr=fr,
            tyre_temp_rl=rl,
            tyre_temp_rr=rr,
            tyre_pressure_fl=pfl,
            tyre_pressure_fr=pfr,
            tyre_pressure_rl=prl,
            tyre_pressure_rr=prr,
        )
        fl_temps.append(fl)
        fr_temps.append(fr)
        rl_temps.append(rl)
        rr_temps.append(rr)
        fl_pressures.append(pfl)
        fr_pressures.append(pfr)
        rl_pressures.append(prl)
        rr_pressures.append(prr)

    recomputed = segment_microsectors(records, synthetic_bundles)
    assert recomputed, "expected segmentation with modified records"
    updated = recomputed[0]
    measures = updated.filtered_measures

    for key in (
        "tyre_temp_fl",
        "tyre_temp_fr",
        "tyre_temp_rl",
        "tyre_temp_rr",
        "tyre_pressure_fl",
        "tyre_pressure_fr",
        "tyre_pressure_rl",
        "tyre_pressure_rr",
        "tyre_temp_fl_std",
        "tyre_temp_fr_std",
        "tyre_temp_rl_std",
        "tyre_temp_rr_std",
        "tyre_pressure_fl_std",
        "tyre_pressure_fr_std",
        "tyre_pressure_rl_std",
        "tyre_pressure_rr_std",
    ):
        assert key not in measures

def test_detect_quiet_microsector_streaks_flags_sequences() -> None:
    def _microsector(index: int, *, quiet: bool) -> Microsector:
        measures = {
            "si_variance": 0.0004 if quiet else 0.01,
            "epi_derivative_abs": 0.05 if quiet else 0.3,
        }
        events = (
            {
                "duration": 0.8,
                "slack": 0.5,
                "structural_density_mean": 0.05,
            },
        )
        operator_events = {"SILENCE": events} if quiet else {}
        return Microsector(
            index=index,
            start_time=float(index),
            end_time=float(index) + 1.0,
            curvature=1.3,
            brake_event=False,
            support_event=False,
            delta_nfr_signature=0.1,
            goals=(),
            phase_boundaries={},
            phase_samples={},
            active_phase="entry1",
            dominant_nodes={},
            phase_weights={},
            grip_rel=1.0,
            phase_lag={},
            phase_alignment={},
            filtered_measures=measures,
            recursivity_trace=(),
            last_mutation=None,
            window_occupancy={},
            operator_events=operator_events,
        )

    microsectors = [
        _microsector(0, quiet=False),
        _microsector(1, quiet=True),
        _microsector(2, quiet=True),
        _microsector(3, quiet=True),
        _microsector(4, quiet=False),
    ]
    streaks = detect_quiet_microsector_streaks(microsectors)
    assert streaks == [(1, 2, 3)]
    coverage, slack, si_variance, epi_abs = microsector_stability_metrics(microsectors[1])
    assert coverage == pytest.approx(0.8)
    assert slack == pytest.approx(0.5)
    assert si_variance == pytest.approx(0.0004)
    assert epi_abs == pytest.approx(0.05)


def test_segment_microsectors_returns_contextual_factors(
    synthetic_records, synthetic_bundles
):
    microsectors = segment_microsectors(synthetic_records, synthetic_bundles)
    assert microsectors
    matrix = load_context_matrix()
    for microsector in microsectors:
        assert set(microsector.context_factors) == {"curve", "surface", "traffic"}
        assert microsector.sample_context_factors
        indices = sorted(microsector.sample_context_factors)
        assert indices == list(range(indices[0], indices[-1] + 1))
        phase_indices: set[int] = set()
        for payload in microsector.phase_samples.values():
            phase_indices.update(int(value) for value in payload)
        assert phase_indices.issubset(set(indices))
        multipliers = [
            apply_contextual_delta(1.0, factors, context_matrix=matrix)
            for factors in microsector.sample_context_factors.values()
        ]
        assert any(not math.isclose(value, 1.0) for value in multipliers)


def test_segment_microsectors_returns_empty_when_no_curvature():
    records = [
        build_dynamic_record(
            i * 0.1,
            5200.0,
            0.02,
            0.2,
            0.1,
            500.0,
            0.9,
            speed=15.0,
            throttle=0.5,
            gear=3,
            vertical_load_front=2600.0,
            vertical_load_rear=2600.0,
            mu_eff_front=0.4,
            mu_eff_rear=0.4,
            mu_eff_front_lateral=0.4,
            mu_eff_front_longitudinal=0.3,
            mu_eff_rear_lateral=0.4,
            mu_eff_rear_longitudinal=0.3,
            suspension_travel_front=0.5,
            suspension_travel_rear=0.5,
        )
        for i in range(5)
    ]
    bundles = EPIExtractor().extract(records)
    assert segment_microsectors(records, bundles) == []
@pytest.fixture
def bottoming_segments(monkeypatch):
    high_lateral = 1.35
    low_lateral = 0.2
    samples = [
        (high_lateral, 0.012, 0.03),
        (high_lateral, 0.013, 0.03),
        (high_lateral, 0.011, 0.03),
        (high_lateral, 0.02, 0.03),
        (low_lateral, 0.02, 0.03),
        (low_lateral, 0.02, 0.03),
        (high_lateral, 0.025, 0.011),
        (high_lateral, 0.024, 0.012),
        (high_lateral, 0.023, 0.013),
        (high_lateral, 0.022, 0.012),
    ]
    records = [
        replace(
            build_dynamic_record(
                index * 0.2,
                5200.0,
                0.03,
                lat,
                -0.2,
                480.0,
                0.82,
                brake_pressure=0.5,
                speed=40.0 - index,
                slip_angle=0.02,
                steer=0.1,
                throttle=0.6,
                gear=3,
                vertical_load_front=2600.0,
                vertical_load_rear=2600.0,
                mu_eff_front=0.82,
                mu_eff_rear=0.78,
                mu_eff_front_lateral=0.84,
                mu_eff_front_longitudinal=0.74,
                mu_eff_rear_lateral=0.82,
                mu_eff_rear_longitudinal=0.72,
                suspension_travel_front=0.02,
                suspension_travel_rear=0.02,
                suspension_velocity_front=0.1,
                suspension_velocity_rear=0.1,
            ),
            suspension_travel_front=front,
            suspension_travel_rear=rear,
        )
        for index, (lat, front, rear) in enumerate(samples)
    ]
    bundles = EPIExtractor().extract(records)

    aero = AeroCoherence()
    smooth_histogram = BumpstopHistogram(
        front_density=(0.08, 0.04, 0.0, 0.0),
        rear_density=(0.02, 0.03, 0.0, 0.0),
        front_energy=(0.12, 0.06, 0.0, 0.0),
        rear_energy=(0.03, 0.05, 0.0, 0.0),
        front_total_density=0.12,
        rear_total_density=0.05,
        front_total_energy=0.18,
        rear_total_energy=0.08,
    )
    front_velocity_profile = SuspensionVelocityBands(
        compression_low_ratio=0.25,
        compression_medium_ratio=0.3,
        compression_high_ratio=0.45,
        rebound_low_ratio=0.3,
        rebound_medium_ratio=0.4,
        rebound_high_ratio=0.3,
        ar_index=1.5,
    )
    rear_velocity_profile = SuspensionVelocityBands(
        compression_low_ratio=0.35,
        compression_medium_ratio=0.35,
        compression_high_ratio=0.3,
        rebound_low_ratio=0.4,
        rebound_medium_ratio=0.35,
        rebound_high_ratio=0.25,
        ar_index=0.9,
    )
    rough_histogram = BumpstopHistogram(
        front_density=(0.03, 0.06, 0.09, 0.0),
        rear_density=(0.04, 0.07, 0.11, 0.0),
        front_energy=(0.05, 0.08, 0.11, 0.0),
        rear_energy=(0.06, 0.09, 0.13, 0.0),
        front_total_density=0.18,
        rear_total_density=0.22,
        front_total_energy=0.24,
        rear_total_energy=0.28,
    )
    smooth_metrics = WindowMetrics(
        si=0.8,
        si_variance=0.0004,
        d_nfr_couple=0.1,
        d_nfr_res=0.05,
        d_nfr_flat=0.02,
        nu_f=1.2,
        nu_exc=0.9,
        rho=0.85,
        phase_lag=0.0,
        phase_alignment=0.95,
        phase_synchrony_index=0.97,
        motor_latency_ms=0.0,
        phase_motor_latency_ms={},
        useful_dissonance_ratio=0.4,
        useful_dissonance_percentage=40.0,
        coherence_index=0.5,
        ackermann_parallel_index=0.0,
        slide_catch_budget=SlideCatchBudget(),
        locking_window_score=LockingWindowScore(),
        support_effective=0.12,
        load_support_ratio=0.00003,
        structural_expansion_longitudinal=0.1,
        structural_contraction_longitudinal=0.03,
        structural_expansion_lateral=0.05,
        structural_contraction_lateral=0.02,
        bottoming_ratio_front=0.6,
        bottoming_ratio_rear=0.12,
        bumpstop_histogram=smooth_histogram,
        mu_usage_front_ratio=0.0,
        mu_usage_rear_ratio=0.0,
        phase_mu_usage_front_ratio=0.0,
        phase_mu_usage_rear_ratio=0.0,
        mu_balance=0.05,
        mu_symmetry={"window": {"front": 0.12, "rear": -0.08}, "apex": {"front": 0.18, "rear": -0.1}},
        exit_gear_match=0.0,
        shift_stability=1.0,
        frequency_label="",
        aero_coherence=aero,
        aero_mechanical_coherence=0.5,
        epi_derivative_abs=0.05,
        brake_headroom=BrakeHeadroom(),
        suspension_velocity_front=front_velocity_profile,
        suspension_velocity_rear=rear_velocity_profile,
    )
    rough_metrics = replace(
        smooth_metrics,
        bottoming_ratio_front=0.18,
        bottoming_ratio_rear=0.68,
        useful_dissonance_ratio=0.46,
        useful_dissonance_percentage=46.0,
        bumpstop_histogram=rough_histogram,
        suspension_velocity_front=SuspensionVelocityBands(
            compression_low_ratio=0.2,
            compression_medium_ratio=0.25,
            compression_high_ratio=0.55,
            rebound_low_ratio=0.25,
            rebound_medium_ratio=0.35,
            rebound_high_ratio=0.4,
            ar_index=1.65,
        ),
        suspension_velocity_rear=SuspensionVelocityBands(
            compression_low_ratio=0.4,
            compression_medium_ratio=0.3,
            compression_high_ratio=0.3,
            rebound_low_ratio=0.45,
            rebound_medium_ratio=0.3,
            rebound_high_ratio=0.25,
            ar_index=0.8,
        ),
        mu_balance=-0.04,
        mu_symmetry={"window": {"front": -0.06, "rear": 0.09}},
    )
    metric_sequence = iter([smooth_metrics, rough_metrics])

    def _fake_window_metrics(*args, **kwargs):
        try:
            return next(metric_sequence)
        except StopIteration:
            return rough_metrics

    monkeypatch.setattr(segmentation_module, "compute_window_metrics", _fake_window_metrics)

    surface_sequence = iter(
        [
            ContextFactors(1.0, 0.95, 1.0),
            ContextFactors(1.0, 1.25, 1.0),
        ]
    )

    def _fake_context(*args, **kwargs):
        try:
            return next(surface_sequence)
        except StopIteration:
            return ContextFactors()

    monkeypatch.setattr(
        segmentation_module, "resolve_microsector_context", _fake_context
    )

    return records, bundles


def _classify_from_series(lateral: list[float], speeds: list[float], dt: float) -> str:
    records = [
        build_dynamic_record(
            index * dt,
            5200.0,
            0.03,
            lat,
            -0.2,
            480.0,
            0.82,
            brake_pressure=0.5,
            speed=speeds[index],
            slip_angle=0.02,
            steer=0.1,
            throttle=0.6,
            gear=3,
            vertical_load_front=2600.0,
            vertical_load_rear=2600.0,
            mu_eff_front=0.82,
            mu_eff_rear=0.78,
            mu_eff_front_lateral=0.84,
            mu_eff_front_longitudinal=0.74,
            mu_eff_rear_lateral=0.82,
            mu_eff_rear_longitudinal=0.72,
            suspension_travel_front=0.02,
            suspension_travel_rear=0.02,
            suspension_velocity_front=0.1,
            suspension_velocity_rear=0.1,
        )
        for index, lat in enumerate(lateral)
    ]
    bundles = EPIExtractor().extract(records)
    microsectors = segment_microsectors(records, bundles)
    assert microsectors
    return microsectors[0].goals[0].archetype


def test_archetype_detection_uses_dynamic_thresholds() -> None:
    hairpin_lateral = [2.5, 2.6, 2.7, 2.6, 2.5, 2.4]
    hairpin_speeds = [42.0, 36.0, 30.0, 28.0, 30.0, 34.0]
    hairpin = _classify_from_series(hairpin_lateral, hairpin_speeds, 0.5)
    assert hairpin == "hairpin"

    chicane_lateral = [1.8, 1.7, -1.8, -1.7, 1.6, 1.5]
    chicane_speeds = [48.0, 47.0, 46.0, 45.0, 46.0, 47.0]
    chicane = _classify_from_series(chicane_lateral, chicane_speeds, 0.35)
    assert chicane == "chicane"

    fast_lateral = [1.35, 1.4, 1.42, 1.38]
    fast_speeds = [62.0, 61.0, 60.5, 60.0]
    fast = _classify_from_series(fast_lateral, fast_speeds, 0.4)
    assert fast == "fast"


def test_segment_microsectors_emits_structural_silence_events() -> None:
    records = [
        build_dynamic_record(
            index * 0.1,
            4800.0,
            0.01,
            1.25,
            0.05,
            102.0,
            0.9,
            brake_pressure=0.04,
            speed=32.0,
            yaw_rate=0.02,
            slip_angle=0.02,
            steer=0.03,
            throttle=0.18,
            gear=3,
            vertical_load_front=2400.0,
            vertical_load_rear=2400.0,
            mu_eff_front=0.5,
            mu_eff_rear=0.5,
            mu_eff_front_lateral=0.5,
            mu_eff_front_longitudinal=0.4,
            mu_eff_rear_lateral=0.5,
            mu_eff_rear_longitudinal=0.4,
            suspension_travel_front=0.01,
            suspension_travel_rear=0.01,
        )
        for index in range(14)
    ]
    bundles = EPIExtractor().extract(records)
    microsectors = segment_microsectors(records, bundles)
    assert microsectors
    silence_events = silence_event_payloads(microsectors[0].operator_events)
    assert silence_events
    event = silence_events[0]
    assert event["duration"] > 0.5
    assert event["load_span"] < 200.0


def test_segment_microsectors_exposes_bottoming_ratios(bottoming_segments) -> None:
    records, bundles = bottoming_segments
    microsectors = segment_microsectors(records, bundles)
    assert len(microsectors) >= 2
    smooth, rough = microsectors[:2]
    assert smooth.filtered_measures["bottoming_ratio_front"] == pytest.approx(0.6)
    assert smooth.context_factors.get("surface") == pytest.approx(0.95)
    assert smooth.filtered_measures["bumpstop_front_density"] == pytest.approx(0.12)
    assert smooth.filtered_measures["bumpstop_front_energy_bin_0"] == pytest.approx(0.12)
    assert smooth.filtered_measures["suspension_velocity_front_high_speed_pct"] == pytest.approx(45.0)
    assert smooth.filtered_measures["suspension_velocity_front_ar_index"] == pytest.approx(1.5)
    assert smooth.filtered_measures["mu_balance"] == pytest.approx(0.05)
    assert smooth.filtered_measures["mu_symmetry_front"] == pytest.approx(0.12)
    assert smooth.filtered_measures["mu_symmetry_rear"] == pytest.approx(-0.08)
    assert smooth.filtered_measures["mu_symmetry_apex_front"] == pytest.approx(0.18)
    assert smooth.filtered_measures["mu_symmetry_apex_rear"] == pytest.approx(-0.1)
    assert rough.filtered_measures["bottoming_ratio_rear"] == pytest.approx(0.68)
    assert rough.context_factors.get("surface") == pytest.approx(1.25)
    assert rough.filtered_measures["bumpstop_rear_density"] == pytest.approx(0.22)
    assert rough.filtered_measures["bumpstop_rear_energy_bin_1"] == pytest.approx(0.09)
    assert rough.filtered_measures["suspension_velocity_front_high_speed_pct"] == pytest.approx(
        55.0
    )
    assert rough.filtered_measures["suspension_velocity_rear_ar_index"] == pytest.approx(0.8)
    assert rough.filtered_measures["mu_balance"] == pytest.approx(-0.04)
    assert rough.filtered_measures["mu_symmetry_front"] == pytest.approx(-0.06)
    assert rough.filtered_measures["mu_symmetry_rear"] == pytest.approx(0.09)


def _yaw_rate(records: list[TelemetryRecord], index: int) -> float:
    if index <= 0:
        return 0.0
    dt = records[index].timestamp - records[index - 1].timestamp
    if dt <= 0:
        return 0.0
    delta = records[index].yaw - records[index - 1].yaw
    wrapped = (delta + math.pi) % (2.0 * math.pi) - math.pi
    return wrapped / dt


def test_window_occupancy_matches_goal_windows(
    synthetic_microsectors,
    synthetic_records,
):
    for microsector in synthetic_microsectors:
        for goal in microsector.goals:
            occupancy = microsector.window_occupancy.get(goal.phase, {})
            indices = list(microsector.phase_indices(goal.phase))
            slip_values = [synthetic_records[i].slip_ratio for i in indices]
            yaw_rates = [_yaw_rate(synthetic_records, idx) for idx in indices]

            def _percentage(values: list[float], window: tuple[float, float]) -> float:
                if not values:
                    return 0.0
                lower, upper = window
                if lower > upper:
                    lower, upper = upper, lower
                count = sum(1 for value in values if lower <= value <= upper)
                return 100.0 * count / len(values)

            expected_lat = _percentage(slip_values, goal.slip_lat_window)
            expected_long = _percentage(slip_values, goal.slip_long_window)
            expected_yaw = _percentage(yaw_rates, goal.yaw_rate_window)

            assert occupancy
            assert 0.0 <= occupancy.get("slip_lat", -1.0) <= 100.0
            assert 0.0 <= occupancy.get("slip_long", -1.0) <= 100.0
            assert 0.0 <= occupancy.get("yaw_rate", -1.0) <= 100.0
            assert occupancy["slip_lat"] == pytest.approx(expected_lat, abs=1e-6)
            assert occupancy["slip_long"] == pytest.approx(expected_long, abs=1e-6)
            assert occupancy["yaw_rate"] == pytest.approx(expected_yaw, abs=1e-6)


def test_goal_targets_match_phase_averages(
    synthetic_microsectors,
    synthetic_records,
    synthetic_bundles,
):
    matrix = load_context_matrix()
    for microsector in synthetic_microsectors:
        for goal in microsector.goals:
            indices = list(microsector.phase_indices(goal.phase))
            phase_bundles = [synthetic_bundles[i] for i in indices]
            phase_records = [synthetic_records[i] for i in indices]
            if phase_bundles:
                adjusted_values = []
                multipliers: list[float] = []
                for idx, bundle in zip(indices, phase_bundles):
                    factors = microsector.sample_context_factors.get(idx)
                    if not factors:
                        factors = microsector.context_factors
                    multiplier = metrics_segmentation._resolve_context_multiplier(
                        factors,
                        context_matrix=matrix,
                    )
                    multipliers.append(multiplier)
                    adjusted_values.append(
                        apply_contextual_delta(
                            bundle.delta_nfr,
                            factors,
                            context_matrix=matrix,
                        )
                    )
                assert goal.target_delta_nfr == pytest.approx(
                    mean(adjusted_values)
                )
                assert goal.target_sense_index == pytest.approx(
                    mean(bundle.sense_index for bundle in phase_bundles)
                )
                expected_long = sum(
                    bundle.delta_nfr_proj_longitudinal * multiplier
                    for bundle, multiplier in zip(phase_bundles, multipliers)
                ) / len(phase_bundles)
                expected_lat = sum(
                    bundle.delta_nfr_proj_lateral * multiplier
                    for bundle, multiplier in zip(phase_bundles, multipliers)
                ) / len(phase_bundles)
                expected_abs_long = sum(
                    abs(bundle.delta_nfr_proj_longitudinal) * multiplier
                    for bundle, multiplier in zip(phase_bundles, multipliers)
                ) / len(phase_bundles)
                expected_abs_lat = sum(
                    abs(bundle.delta_nfr_proj_lateral) * multiplier
                    for bundle, multiplier in zip(phase_bundles, multipliers)
                ) / len(phase_bundles)
                total_axis = expected_abs_long + expected_abs_lat
                if total_axis > 1e-9:
                    expected_weights = {
                        "longitudinal": expected_abs_long / total_axis,
                        "lateral": expected_abs_lat / total_axis,
                    }
                else:
                    expected_weights = {"longitudinal": 0.5, "lateral": 0.5}

                assert goal.target_delta_nfr_long == pytest.approx(expected_long)
                assert goal.target_delta_nfr_lat == pytest.approx(expected_lat)
                assert goal.delta_axis_weights["longitudinal"] == pytest.approx(
                    expected_weights["longitudinal"]
                )
                assert goal.delta_axis_weights["lateral"] == pytest.approx(
                    expected_weights["lateral"]
                )
            lat_low, lat_high = goal.slip_lat_window
            long_low, long_high = goal.slip_long_window
            yaw_low, yaw_high = goal.yaw_rate_window
            for record in phase_records:
                assert lat_low - 1e-6 <= record.slip_ratio <= lat_high + 1e-6
                assert long_low - 1e-6 <= record.slip_ratio <= long_high + 1e-6
            yaw_rates = [_yaw_rate(synthetic_records, idx) for idx in indices]
            for value in yaw_rates:
                assert yaw_low - 1e-6 <= value <= yaw_high + 1e-6


def test_build_goals_prefers_cached_sample_context(
    synthetic_microsectors,
    synthetic_records,
    synthetic_bundles,
    monkeypatch,
) -> None:
    microsector = synthetic_microsectors[0]
    assert microsector.goals
    archetype = microsector.goals[0].archetype
    boundaries = microsector.phase_boundaries
    context_matrix = load_context_matrix()
    original_resolve = metrics_segmentation.resolve_series_context
    sample_context = list(
        original_resolve(synthetic_bundles, matrix=context_matrix)
    )
    sample_multipliers = [
        metrics_segmentation._resolve_context_multiplier(
            factors, context_matrix=context_matrix
        )
        for factors in sample_context
    ]
    yaw_rates = [
        metrics_segmentation._compute_yaw_rate(synthetic_records, idx)
        for idx in range(len(synthetic_records))
    ]
    phase_gradients = {phase: 0.0 for phase in boundaries}

    call_count = 0

    def _spy(segment, *, matrix):
        nonlocal call_count
        call_count += 1
        return original_resolve(segment, matrix=matrix)

    monkeypatch.setattr(metrics_segmentation, "resolve_series_context", _spy)

    goals_cached, dominant_cached, targets_cached, weights_cached = (
        metrics_segmentation._build_goals(
            archetype,
            synthetic_bundles,
            synthetic_records,
            boundaries,
            yaw_rates,
            context_matrix=context_matrix,
            sample_context=sample_context,
            sample_multipliers=sample_multipliers,
            phase_gradients=phase_gradients,
        )
    )

    assert call_count == 0

    goals_fallback, dominant_fallback, targets_fallback, weights_fallback = (
        metrics_segmentation._build_goals(
            archetype,
            synthetic_bundles,
            synthetic_records,
            boundaries,
            yaw_rates,
            context_matrix=context_matrix,
            sample_context=None,
            sample_multipliers=None,
            phase_gradients=phase_gradients,
        )
    )

    assert call_count > 0
    assert goals_cached == goals_fallback
    assert dominant_cached == dominant_fallback
    assert targets_cached == targets_fallback
    assert weights_cached == weights_fallback


def test_build_goals_uses_sample_multipliers_without_context(
    synthetic_microsectors,
    synthetic_records,
    synthetic_bundles,
    monkeypatch,
) -> None:
    microsector = synthetic_microsectors[0]
    assert microsector.goals
    archetype = microsector.goals[0].archetype
    boundaries = microsector.phase_boundaries
    context_matrix = load_context_matrix()
    original_resolve = metrics_segmentation.resolve_series_context
    sample_context = list(
        original_resolve(synthetic_bundles, matrix=context_matrix)
    )
    sample_multipliers = [
        metrics_segmentation._resolve_context_multiplier(
            factors, context_matrix=context_matrix
        )
        for factors in sample_context
    ]
    yaw_rates = [
        metrics_segmentation._compute_yaw_rate(synthetic_records, idx)
        for idx in range(len(synthetic_records))
    ]
    phase_gradients = {phase: 0.0 for phase in boundaries}

    call_count = 0

    def _spy(segment, *, matrix):
        nonlocal call_count
        call_count += 1
        return original_resolve(segment, matrix=matrix)

    monkeypatch.setattr(metrics_segmentation, "resolve_series_context", _spy)

    goals_direct, dominant_direct, targets_direct, weights_direct = (
        metrics_segmentation._build_goals(
            archetype,
            synthetic_bundles,
            synthetic_records,
            boundaries,
            yaw_rates,
            context_matrix=context_matrix,
            sample_context=None,
            sample_multipliers=sample_multipliers,
            phase_gradients=phase_gradients,
        )
    )

    assert call_count == 0

    goals_reference, dominant_reference, targets_reference, weights_reference = (
        metrics_segmentation._build_goals(
            archetype,
            synthetic_bundles,
            synthetic_records,
            boundaries,
            yaw_rates,
            context_matrix=context_matrix,
            sample_context=sample_context,
            sample_multipliers=sample_multipliers,
            phase_gradients=phase_gradients,
        )
    )

    assert goals_direct == goals_reference
    assert dominant_direct == dominant_reference
    assert targets_direct == targets_reference
    assert weights_direct == weights_reference


def test_build_goals_falls_back_when_sample_context_truncated(
    synthetic_microsectors,
    synthetic_records,
    synthetic_bundles,
    monkeypatch,
) -> None:
    microsector = synthetic_microsectors[0]
    assert microsector.goals
    archetype = microsector.goals[0].archetype
    boundaries = microsector.phase_boundaries
    context_matrix = load_context_matrix()
    original_resolve = metrics_segmentation.resolve_series_context
    sample_context = list(
        original_resolve(synthetic_bundles, matrix=context_matrix)
    )
    sample_multipliers = [
        metrics_segmentation._resolve_context_multiplier(
            factors, context_matrix=context_matrix
        )
        for factors in sample_context
    ]
    yaw_rates = [
        metrics_segmentation._compute_yaw_rate(synthetic_records, idx)
        for idx in range(len(synthetic_records))
    ]
    phase_gradients = {phase: 0.0 for phase in boundaries}
    all_indices = [
        idx
        for start, stop in boundaries.values()
        for idx in range(start, min(stop, len(synthetic_bundles)))
    ]
    assert all_indices
    max_index = max(all_indices)
    truncated_context = tuple(sample_context[: max_index])
    truncated_multipliers = tuple(sample_multipliers[: max_index])
    assert len(truncated_context) < len(sample_context)

    call_count = 0

    def _spy(segment, *, matrix):
        nonlocal call_count
        call_count += 1
        return original_resolve(segment, matrix=matrix)

    monkeypatch.setattr(metrics_segmentation, "resolve_series_context", _spy)

    goals_truncated, dominant_truncated, targets_truncated, weights_truncated = (
        metrics_segmentation._build_goals(
            archetype,
            synthetic_bundles,
            synthetic_records,
            boundaries,
            yaw_rates,
            context_matrix=context_matrix,
            sample_context=truncated_context,
            sample_multipliers=truncated_multipliers,
            phase_gradients=phase_gradients,
        )
    )

    assert call_count > 0

    goals_reference, dominant_reference, targets_reference, weights_reference = (
        metrics_segmentation._build_goals(
            archetype,
            synthetic_bundles,
            synthetic_records,
            boundaries,
            yaw_rates,
            context_matrix=context_matrix,
            sample_context=None,
            sample_multipliers=None,
            phase_gradients=phase_gradients,
        )
    )

    assert goals_truncated == goals_reference
    assert dominant_truncated == dominant_reference
    assert targets_truncated == targets_reference
    assert weights_truncated == weights_reference


def test_goals_expose_archetype_targets(synthetic_microsectors) -> None:
    for microsector in synthetic_microsectors:
        for goal in microsector.goals:
            targets = archetype_phase_targets(goal.archetype)
            family = phase_family(goal.phase)
            phase_target = targets.get(family)
            assert phase_target is not None
            assert goal.archetype_delta_nfr_long_target == pytest.approx(phase_target.delta_nfr_long)
            assert goal.archetype_delta_nfr_lat_target == pytest.approx(phase_target.delta_nfr_lat)
            assert goal.archetype_nu_f_target == pytest.approx(phase_target.nu_f)
            assert goal.archetype_si_phi_target == pytest.approx(phase_target.si_phi)


def test_segment_microsectors_applies_phase_weight_overrides(
    synthetic_records,
    synthetic_bundles,
):
    baseline_micro = segment_microsectors(
        synthetic_records,
        list(synthetic_bundles),
    )
    override_micro = segment_microsectors(
        synthetic_records,
        list(synthetic_bundles),
        phase_weight_overrides={"entry": {"tyres": 1.8}},
    )
    assert baseline_micro and override_micro
    entry_candidates = expand_phase_alias("entry")
    phase_key = next(
        (candidate for candidate in entry_candidates if candidate in baseline_micro[0].phase_weights),
        entry_candidates[0],
    )
    base_entry = baseline_micro[0].phase_weights.get(phase_key, {})
    override_entry = override_micro[0].phase_weights.get(phase_key, {})
    assert isinstance(base_entry, dict) and isinstance(override_entry, dict)
    assert override_entry.get("tyres", 0.0) > base_entry.get("tyres", 0.0)
    entry_samples: Tuple[int, ...] = ()
    for candidate in entry_candidates:
        samples = override_micro[0].phase_samples.get(candidate, ())
        if samples:
            entry_samples = samples
            phase_key = candidate
            break
    if not entry_samples:
        entry_samples = baseline_micro[0].phase_samples.get(phase_key, ())
    assert entry_samples
    sample_index = entry_samples[0]
    record = synthetic_records[sample_index]
    base_nu = resolve_nu_f_by_node(
        record,
        phase=phase_key,
        phase_weights=baseline_micro[0].phase_weights,
    ).by_node
    override_nu = resolve_nu_f_by_node(
        record,
        phase=phase_key,
        phase_weights=override_micro[0].phase_weights,
    ).by_node
    assert override_nu["tyres"] > base_nu["tyres"]


def test_phase_weighting_penalises_sense_index(
    synthetic_microsectors,
    synthetic_records,
    synthetic_bundles,
):
    baseline = DeltaCalculator.derive_baseline(synthetic_records)
    for microsector in synthetic_microsectors:
        weights = microsector.phase_weights
        goal_targets = {goal.phase: goal.nu_f_target for goal in microsector.goals}
        for phase, indices in microsector.phase_samples.items():
            for idx in indices:
                record = synthetic_records[idx]
                node_record = replace(record, reference=baseline)
                node_deltas = delta_nfr_by_node(node_record)
                nu_phase = resolve_nu_f_by_node(
                    record, phase=phase, phase_weights=weights
                ).by_node
                nu_default = resolve_nu_f_by_node(record).by_node
                weighted_index = sense_index(
                    record.nfr - baseline.nfr,
                    node_deltas,
                    baseline.nfr,
                    nu_f_by_node=nu_phase,
                    active_phase=phase,
                    w_phase=weights,
                    nu_f_targets=goal_targets,
                )
                neutral_index = sense_index(
                    record.nfr - baseline.nfr,
                    node_deltas,
                    baseline.nfr,
                    nu_f_by_node=nu_default,
                    active_phase=phase,
                    w_phase=DEFAULT_PHASE_WEIGHTS,
                    nu_f_targets=goal_targets,
                )
                assert weighted_index <= neutral_index + 1e-6


def test_integrator_matches_derivative_series(
    synthetic_records,
    synthetic_bundles,
):
    nodes = ("tyres", "suspension", "chassis", "brakes", "transmission", "track", "driver")
    for index, bundle in enumerate(synthetic_bundles):
        if index == 0:
            continue
        dt = synthetic_records[index].timestamp - synthetic_records[index - 1].timestamp
        derivative_expected = sum(
            getattr(bundle, node).nu_f * getattr(bundle, node).delta_nfr for node in nodes
        )
        assert bundle.dEPI_dt == pytest.approx(derivative_expected, rel=1e-6, abs=1e-6)
        expected_integrated = synthetic_bundles[index - 1].integrated_epi + (bundle.dEPI_dt * dt)
        assert bundle.integrated_epi == pytest.approx(expected_integrated, rel=1e-6, abs=1e-6)
        for node in nodes:
            integral, derivative = bundle.node_evolution[node]
            node_model = getattr(bundle, node)
            assert node_model.dEPI_dt == pytest.approx(derivative, rel=1e-6, abs=1e-9)
            assert node_model.integrated_epi == pytest.approx(integral, rel=1e-6, abs=1e-9)
        nodal_derivative = sum(bundle.node_evolution[node][1] for node in nodes)
        nodal_integral = sum(bundle.node_evolution[node][0] for node in nodes)
        assert nodal_derivative == pytest.approx(bundle.dEPI_dt, rel=1e-6)
        assert nodal_integral == pytest.approx(bundle.dEPI_dt * dt, rel=1e-6, abs=1e-9)


def test_segment_microsectors_preserves_operator_state(
    synthetic_records,
    synthetic_bundles,
):
    operator_state: dict[str, dict[str, dict[str, object]]] = {}
    first_pass = segment_microsectors(
        synthetic_records,
        list(synthetic_bundles),
        operator_state=operator_state,
    )
    second_pass = segment_microsectors(
        synthetic_records,
        list(synthetic_bundles),
        operator_state=operator_state,
    )

    assert len(first_pass) == len(second_pass) > 0
    for before, after in zip(first_pass, second_pass):
        assert before.recursivity_trace
        assert after.recursivity_trace
        assert len(after.recursivity_trace) >= len(before.recursivity_trace)
        assert after.filtered_measures.get("style_index") is not None
        assert after.filtered_measures.get("grip_rel") is not None
        assert after.grip_rel == pytest.approx(
            after.filtered_measures.get("grip_rel", after.grip_rel)
        )
        assert after.last_mutation is not None
        assert before.last_mutation is not None
        assert after.last_mutation.get("archetype") == before.last_mutation.get("archetype")


def test_segment_microsectors_recursivity_entropy_consistency(
    synthetic_records,
    synthetic_bundles,
    monkeypatch,
) -> None:
    records = list(synthetic_records)
    bundles = list(synthetic_bundles)

    captured_triggers: list[float] = []
    real_estimate_entropy = metrics_segmentation._estimate_entropy
    entropy_calls: list[dict[str, object]] = []

    def _capture_mutation_operator(state, triggers, **kwargs):
        entropy_value = float(triggers.get("entropy", 0.0))
        captured_triggers.append(entropy_value)
        return {
            "archetype": str(triggers.get("candidate_archetype", "")),
            "mutated": False,
            "entropy": entropy_value,
        }

    def _capture_entropy(records_arg, start, end, **kwargs):
        entropy_value = real_estimate_entropy(records_arg, start, end, **kwargs)
        baseline_value = real_estimate_entropy(
            records_arg,
            start,
            end,
            bundles=bundles,
        )
        entropy_calls.append(
            {
                "entropy": entropy_value,
                "baseline": baseline_value,
                "node_delta_cache": kwargs.get("node_delta_cache"),
                "cache_offset": kwargs.get("cache_offset"),
            }
        )
        return entropy_value

    monkeypatch.setattr(metrics_segmentation, "mutation_operator", _capture_mutation_operator)
    monkeypatch.setattr(metrics_segmentation, "_estimate_entropy", _capture_entropy)

    operator_state: dict[str, dict[str, dict[str, object]]] = {}
    segment_microsectors(
        records,
        bundles,
        operator_state=operator_state,
    )

    assert captured_triggers
    assert entropy_calls

    for call in entropy_calls:
        cache = call["node_delta_cache"]
        assert cache is not None
        assert isinstance(cache, Sequence)
        assert len(cache) == len(records)
        assert call["entropy"] == pytest.approx(call["baseline"], rel=1e-9)
        assert call["cache_offset"] == 0

    for trigger_entropy, call in zip(captured_triggers, entropy_calls):
        assert trigger_entropy == pytest.approx(call["entropy"], rel=1e-9)


def test_operator_detectors_receive_segment_windows(
    synthetic_records, synthetic_bundles, monkeypatch
) -> None:
    records = list(synthetic_records)
    bundles = list(synthetic_bundles)

    def _fake_detect_al(*args, **kwargs):
        window = args[0] if args else kwargs.get("window", ())
        signature = tuple(record.timestamp for record in window)
        if not window:
            return ()
        return (
            {
                "start_index": 0,
                "end_index": len(window) - 1,
                "window_signature": signature,
            },
        )

    monkeypatch.setattr(metrics_segmentation, "detect_al", _fake_detect_al)

    def _empty_detector(*args, **kwargs):
        return ()

    for name in (
        "detect_en",
        "detect_il",
        "detect_nul",
        "detect_oz",
        "detect_ra",
        "detect_remesh",
        "detect_thol",
        "detect_um",
        "detect_val",
        "detect_zhir",
        "detect_silence",
    ):
        monkeypatch.setattr(metrics_segmentation, name, _empty_detector)

    microsectors = metrics_segmentation.segment_microsectors(records, bundles)
    assert len(microsectors) >= 2

    signatures: List[Tuple[float, ...]] = []
    for microsector in microsectors[:2]:
        events = microsector.operator_events.get("AL")
        assert events, "expected synthetic AL detection event"
        event = events[0]
        signature = tuple(event["window_signature"])  # type: ignore[index]
        global_start = int(event["global_start_index"])  # type: ignore[index]
        global_end = int(event["global_end_index"])  # type: ignore[index]
        expected = tuple(
            record.timestamp for record in records[global_start : global_end + 1]
        )
        assert signature == expected
        signatures.append(signature)

    assert len(signatures) == 2
    assert signatures[0] != signatures[1]
