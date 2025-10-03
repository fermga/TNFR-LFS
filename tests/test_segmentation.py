import pytest

from tnfr_lfs.core.epi import TelemetryRecord
from tnfr_lfs.core.epi_models import (
    BrakesNode,
    ChassisNode,
    DriverNode,
    EPIBundle,
    SuspensionNode,
    TrackNode,
    TransmissionNode,
    TyresNode,
)
from tnfr_lfs.core.segmentation import Goal, Microsector, segment_microsectors


def _build_bundle(timestamp: float, delta_nfr: float, sense_index: float) -> EPIBundle:
    nodes = dict(
        tyres=TyresNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
        suspension=SuspensionNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
        chassis=ChassisNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
        brakes=BrakesNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
        transmission=TransmissionNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
        track=TrackNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
        driver=DriverNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
    )
    return EPIBundle(
        timestamp=timestamp,
        epi=0.5,
        delta_nfr=delta_nfr,
        sense_index=sense_index,
        **nodes,
    )


def test_segment_microsectors_creates_goals_with_stable_assignments():
    # Synthetic telemetry describing a single corner with clear phases.
    timestamps = [i * 0.1 for i in range(12)]
    lateral = [0.2, 0.3, 0.4, 0.5, 1.6, 2.1, 2.4, 2.0, 1.5, 0.4, 0.3, 0.2]
    longitudinal = [0.2, 0.1, 0.05, -0.1, -0.5, -0.7, -0.2, 0.3, 0.6, 0.4, 0.2, 0.1]
    vertical = [5200, 5250, 5300, 5400, 5600, 5850, 6100, 6000, 5800, 5500, 5400, 5300]
    records = [
        TelemetryRecord(
            timestamp=t,
            vertical_load=v,
            slip_ratio=0.05,
            lateral_accel=l,
            longitudinal_accel=lon,
            nfr=500 + i,
            si=0.65,
        )
        for i, (t, l, lon, v) in enumerate(zip(timestamps, lateral, longitudinal, vertical))
    ]
    bundles = [
        _build_bundle(record.timestamp, 8.0 if 4 <= idx <= 8 else 1.0, 0.65)
        for idx, record in enumerate(records)
    ]

    microsectors = segment_microsectors(records, bundles)
    assert len(microsectors) == 1
    microsector = microsectors[0]

    assert isinstance(microsector, Microsector)
    assert microsector.brake_event is True
    assert microsector.support_event is True

    # Boundaries must partition the segment without gaps or overlaps.
    ranges = [microsector.phase_indices(phase) for phase in ("entry", "apex", "exit")]
    covered_indices = set()
    for phase_range in ranges:
        assert phase_range.stop > phase_range.start
        for index in phase_range:
            assert index not in covered_indices
            covered_indices.add(index)
    expected_segment = set(range(min(covered_indices), max(covered_indices) + 1))
    assert covered_indices == expected_segment

    # Goals must be aligned with the selected archetype and provided for each phase.
    assert len(microsector.goals) == 3
    phases = [goal.phase for goal in microsector.goals]
    assert phases == ["entry", "apex", "exit"]
    archetypes = {goal.archetype for goal in microsector.goals}
    assert len(archetypes) == 1
    for goal in microsector.goals:
        assert isinstance(goal, Goal)
        assert goal.description
        assert goal.target_sense_index == pytest.approx(0.65)


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
    bundles = [_build_bundle(record.timestamp, 0.5, 0.9) for record in records]
    assert segment_microsectors(records, bundles) == []
