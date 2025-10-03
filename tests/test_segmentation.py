import pytest

import math
from statistics import mean

from tnfr_lfs.core.epi import EPIExtractor, TelemetryRecord
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
