import pytest

import math
from dataclasses import replace
from statistics import mean

from tnfr_lfs.core.coherence import sense_index
from tnfr_lfs.core.epi import (
    DEFAULT_PHASE_WEIGHTS,
    DeltaCalculator,
    EPIExtractor,
    TelemetryRecord,
    delta_nfr_by_node,
    resolve_nu_f_by_node,
)
from tnfr_lfs.core.segmentation import Microsector, segment_microsectors


def test_segment_microsectors_creates_goals_with_stable_assignments(
    synthetic_microsectors,
):
    assert len(synthetic_microsectors) == 2
    for microsector in synthetic_microsectors:
        assert isinstance(microsector, Microsector)
        assert microsector.brake_event is True
        assert microsector.support_event is True
        phases = [goal.phase for goal in microsector.goals]
        assert phases == ["entry", "apex", "exit"]
        assert microsector.active_phase in phases
        boundaries = [microsector.phase_indices(phase) for phase in phases]
        seen = set()
        for phase_range in boundaries:
            assert phase_range.stop > phase_range.start
            seen.update(phase_range)
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
            assert goal.slip_lat_window[0] < goal.slip_lat_window[1]
            assert goal.slip_long_window[0] < goal.slip_long_window[1]
            assert goal.yaw_rate_window[0] <= goal.yaw_rate_window[1]
            assert goal.dominant_nodes == microsector.dominant_nodes[goal.phase]
            assert goal.dominant_nodes


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
            suspension_travel_front=0.5,
            suspension_travel_rear=0.5,
            suspension_velocity_front=0.0,
            suspension_velocity_rear=0.0,
        )
        for i in range(5)
    ]
    bundles = EPIExtractor().extract(records)
    assert segment_microsectors(records, bundles) == []


def _yaw_rate(records: list[TelemetryRecord], index: int) -> float:
    if index <= 0:
        return 0.0
    dt = records[index].timestamp - records[index - 1].timestamp
    if dt <= 0:
        return 0.0
    delta = records[index].yaw - records[index - 1].yaw
    wrapped = (delta + math.pi) % (2.0 * math.pi) - math.pi
    return wrapped / dt


def test_goal_targets_match_phase_averages(
    synthetic_microsectors,
    synthetic_records,
    synthetic_bundles,
):
    for microsector in synthetic_microsectors:
        for goal in microsector.goals:
            indices = list(microsector.phase_indices(goal.phase))
            phase_bundles = [synthetic_bundles[i] for i in indices]
            phase_records = [synthetic_records[i] for i in indices]
            if phase_bundles:
                assert goal.target_delta_nfr == pytest.approx(
                    mean(bundle.delta_nfr for bundle in phase_bundles)
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


def test_phase_weighting_penalises_sense_index(
    synthetic_microsectors,
    synthetic_records,
    synthetic_bundles,
):
    baseline = DeltaCalculator.derive_baseline(synthetic_records)
    for microsector in synthetic_microsectors:
        weights = microsector.phase_weights
        for phase, indices in microsector.phase_samples.items():
            for idx in indices:
                record = synthetic_records[idx]
                node_record = replace(record, reference=baseline)
                node_deltas = delta_nfr_by_node(node_record)
                nu_phase = resolve_nu_f_by_node(record, phase=phase, phase_weights=weights)
                nu_default = resolve_nu_f_by_node(record)
                weighted_index = sense_index(
                    record.nfr - baseline.nfr,
                    node_deltas,
                    baseline.nfr,
                    nu_f_by_node=nu_phase,
                    active_phase=phase,
                    w_phase=weights,
                )
                neutral_index = sense_index(
                    record.nfr - baseline.nfr,
                    node_deltas,
                    baseline.nfr,
                    nu_f_by_node=nu_default,
                    active_phase=phase,
                    w_phase=DEFAULT_PHASE_WEIGHTS,
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
