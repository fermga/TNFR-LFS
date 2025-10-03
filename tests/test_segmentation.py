import pytest

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
        boundaries = [microsector.phase_indices(phase) for phase in phases]
        seen = set()
        for phase_range in boundaries:
            assert phase_range.stop > phase_range.start
            seen.update(phase_range)
        assert seen == set(range(min(seen), max(seen) + 1))
        assert all(goal.description for goal in microsector.goals)
        assert all(0.0 <= goal.target_sense_index <= 1.0 for goal in microsector.goals)


def test_segment_microsectors_returns_empty_when_no_curvature():
    records = [
        TelemetryRecord(
            timestamp=i * 0.1,
            vertical_load=5200,
            slip_ratio=0.02,
            lateral_accel=0.2,
            longitudinal_accel=0.1,
            nfr=500,
            si=0.9,
        )
        for i in range(5)
    ]
    bundles = EPIExtractor().extract(records)
    assert segment_microsectors(records, bundles) == []
