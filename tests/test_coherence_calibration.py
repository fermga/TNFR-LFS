"""Tests for the coherence calibration store and baseline resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest

from tests.helpers import build_calibration_record

from tnfr_lfs.core.coherence_calibration import CoherenceCalibrationStore
from tnfr_lfs.core.epi import DeltaCalculator, EPIExtractor, TelemetryRecord


def _make_stint(vertical_load: float, *, nfr: float = 0.7, samples: int = 5) -> List[TelemetryRecord]:
    return [
        build_calibration_record(vertical_load, nfr=nfr + index * 0.001)
        for index in range(samples)
    ]


def test_coherence_calibration_converges_per_player(tmp_path: Path) -> None:
    store = CoherenceCalibrationStore(
        path=tmp_path / "coherence_calibration.toml",
        decay=0.5,
        min_laps=2,
        max_laps=6,
    )
    schedule = [
        ("alice", "xfg", 5000.0),
        ("bob", "xfg", 4800.0),
        ("alice", "xfg", 5200.0),
        ("bob", "xfg", 5000.0),
        ("alice", "xfg", 5400.0),
        ("bob", "xfg", 4900.0),
    ]
    resolved: Dict[str, List[float]] = {"alice": [], "bob": []}
    for player, car, load in schedule:
        records = _make_stint(load)
        baseline = DeltaCalculator.resolve_baseline(
            records,
            calibration=store,
            player_name=player,
            car_model=car,
        )
        resolved[player].append(baseline.vertical_load)

    assert resolved["alice"] == pytest.approx([5000.0, 5200.0, 5100.0])
    assert resolved["bob"] == pytest.approx([4800.0, 5000.0, 4900.0])
    assert abs(resolved["alice"][2] - resolved["alice"][1]) < abs(5400.0 - 5200.0)

    alice_snapshot = store.snapshot("alice", "xfg")
    assert alice_snapshot is not None
    assert alice_snapshot.laps == 3
    assert alice_snapshot.baseline.vertical_load == pytest.approx(5250.0)
    low, high = alice_snapshot.ranges["vertical_load"]
    assert low < alice_snapshot.baseline.vertical_load < high

    bob_snapshot = store.snapshot("bob", "xfg")
    assert bob_snapshot is not None
    assert bob_snapshot.laps == 3
    assert bob_snapshot.baseline.vertical_load == pytest.approx(4900.0)

    store.save()
    reloaded = CoherenceCalibrationStore(
        path=store.path,
        decay=0.5,
        min_laps=2,
        max_laps=6,
    )
    reloaded_alice = reloaded.snapshot("alice", "xfg")
    assert reloaded_alice is not None
    assert reloaded_alice.baseline.vertical_load == pytest.approx(
        alice_snapshot.baseline.vertical_load
    )


def test_epi_extractor_uses_calibration_baseline(tmp_path: Path) -> None:
    store = CoherenceCalibrationStore(
        path=tmp_path / "calibration.toml",
        decay=0.5,
        min_laps=2,
        max_laps=6,
    )
    for load, nfr in [(5000.0, 0.7), (5200.0, 0.72)]:
        records = _make_stint(load, nfr=nfr)
        DeltaCalculator.resolve_baseline(
            records,
            calibration=store,
            player_name="alice",
            car_model="xfg",
        )

    snapshot_before = store.snapshot("alice", "xfg")
    assert snapshot_before is not None
    baseline_nfr = snapshot_before.baseline.nfr

    extractor = EPIExtractor()
    new_records = _make_stint(5400.0, nfr=0.75)
    bundles = extractor.extract(
        new_records,
        calibration=store,
        player_name="alice",
        car_model="xfg",
    )
    assert bundles
    assert bundles[0].delta_nfr == pytest.approx(new_records[0].nfr - baseline_nfr)

    snapshot_after = store.snapshot("alice", "xfg")
    assert snapshot_after is not None
    assert snapshot_after.laps == 3
    assert snapshot_after.baseline.nfr > baseline_nfr
