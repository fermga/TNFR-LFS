import pytest

import math
from dataclasses import replace
from statistics import mean
from typing import Tuple

from tnfr_lfs.core.archetypes import archetype_phase_targets
from tnfr_lfs.core.contextual_delta import apply_contextual_delta, load_context_matrix
from tnfr_lfs.core.coherence import sense_index
from tnfr_lfs.core.epi import (
    DEFAULT_PHASE_WEIGHTS,
    DeltaCalculator,
    EPIExtractor,
    TelemetryRecord,
    delta_nfr_by_node,
    resolve_nu_f_by_node,
)
from tnfr_lfs.core.phases import PHASE_SEQUENCE, expand_phase_alias, phase_family
from tnfr_lfs.core.segmentation import Microsector, segment_microsectors


def test_segment_microsectors_creates_goals_with_stable_assignments(
    synthetic_microsectors,
):
    assert len(synthetic_microsectors) == 2
    for microsector in synthetic_microsectors:
        assert isinstance(microsector, Microsector)
        assert microsector.brake_event is True
        assert microsector.support_event is True
        assert microsector.grip_rel >= 0.0
        assert set(microsector.filtered_measures) >= {"thermal_load", "style_index", "grip_rel"}
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
            assert -math.pi <= goal.measured_phase_lag <= math.pi
            assert -1.0 <= goal.measured_phase_alignment <= 1.0
            indices = list(microsector.phase_indices(goal.phase))
            if indices:
                assert goal.dominant_nodes
        assert microsector.phase_lag
        assert microsector.phase_alignment
        assert set(microsector.phase_lag) >= set(PHASE_SEQUENCE)
        assert set(microsector.phase_alignment) >= set(PHASE_SEQUENCE)


def test_segment_microsectors_returns_contextual_factors(
    synthetic_records, synthetic_bundles
):
    microsectors = segment_microsectors(synthetic_records, synthetic_bundles)
    assert microsectors
    matrix = load_context_matrix()
    for microsector in microsectors:
        assert set(microsector.context_factors) == {"curve", "surface", "traffic"}
        assert microsector.sample_context_factors
        multipliers = [
            apply_contextual_delta(1.0, factors, context_matrix=matrix)
            for factors in microsector.sample_context_factors.values()
        ]
        assert any(not math.isclose(value, 1.0) for value in multipliers)


def test_segment_microsectors_returns_empty_when_no_curvature():
    records = [
        TelemetryRecord(
            timestamp=i * 0.1,
            vertical_load=5200,
            slip_ratio=0.02,
            lateral_accel=0.2,
            longitudinal_accel=0.1,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            brake_pressure=0.0,
            locking=0.0,
            nfr=500,
            si=0.9,
            speed=15.0,
            yaw_rate=0.0,
            slip_angle=0.0,
            steer=0.0,
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
            suspension_velocity_front=0.0,
            suspension_velocity_rear=0.0,
        )
        for i in range(5)
    ]
    bundles = EPIExtractor().extract(records)
    assert segment_microsectors(records, bundles) == []


def _dynamic_record(timestamp: float, lateral: float, speed: float) -> TelemetryRecord:
    return TelemetryRecord(
        timestamp=timestamp,
        vertical_load=5200.0,
        slip_ratio=0.03,
        lateral_accel=lateral,
        longitudinal_accel=-0.2,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        brake_pressure=0.5,
        locking=0.0,
        nfr=480.0,
        si=0.82,
        speed=speed,
        yaw_rate=0.0,
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


def _classify_from_series(lateral: list[float], speeds: list[float], dt: float) -> str:
    records = [_dynamic_record(index * dt, lat, speeds[index]) for index, lat in enumerate(lateral)]
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
        TelemetryRecord(
            timestamp=index * 0.1,
            vertical_load=4800.0,
            slip_ratio=0.01,
            lateral_accel=1.25,
            longitudinal_accel=0.05,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            brake_pressure=0.04,
            locking=0.0,
            nfr=102.0,
            si=0.9,
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
            suspension_velocity_front=0.0,
            suspension_velocity_rear=0.0,
        )
        for index in range(14)
    ]
    bundles = EPIExtractor().extract(records)
    microsectors = segment_microsectors(records, bundles)
    assert microsectors
    silence_events = microsectors[0].operator_events.get("SILENCIO")
    assert silence_events
    event = silence_events[0]
    assert event["duration"] > 0.5
    assert event["load_span"] < 200.0


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
                for idx, bundle in zip(indices, phase_bundles):
                    factors = microsector.sample_context_factors.get(idx)
                    if not factors:
                        factors = microsector.context_factors
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
            lat_low, lat_high = goal.slip_lat_window
            long_low, long_high = goal.slip_long_window
            yaw_low, yaw_high = goal.yaw_rate_window
            for record in phase_records:
                assert lat_low - 1e-6 <= record.slip_ratio <= lat_high + 1e-6
                assert long_low - 1e-6 <= record.slip_ratio <= long_high + 1e-6
            yaw_rates = [_yaw_rate(synthetic_records, idx) for idx in indices]
            for value in yaw_rates:
                assert yaw_low - 1e-6 <= value <= yaw_high + 1e-6


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
