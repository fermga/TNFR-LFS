"""Tests for SplitPredictor (live SPB / predicted lap)."""
from __future__ import annotations

from lfs_telemetry.telemetry.predict import SplitPredictor


def _feed_lap(p: SplitPredictor, splits_ms: list[int], lap_ms: int) -> None:
    for i, cum in enumerate(splits_ms, start=1):
        p.observe_split(i, cum)
    p.observe_lap(lap_ms)


def test_first_lap_populates_all_segments():
    p = SplitPredictor(n_splits=2)
    _feed_lap(p, [28_500, 58_200], 86_500)
    assert p.best_lap_ms == 86_500
    # segments: 28.5, 29.7, 28.3
    assert p.best_segments_ms == {1: 28_500, 2: 29_700, 3: 28_300}
    assert p.spb_ms() == 86_500


def test_spb_strictly_le_actual_best():
    p = SplitPredictor(n_splits=2)
    _feed_lap(p, [28_500, 58_200], 86_500)   # lap 1
    _feed_lap(p, [28_300, 58_500], 87_100)   # lap 2: better S1, worse S2/S3
    # SPB = best S1 (28.3) + best S2 (29.7) + best S3 (28.3) = 86.3
    assert p.spb_ms() == 86_300
    assert p.spb_ms() <= p.best_lap_ms


def test_predicted_uses_best_remaining():
    p = SplitPredictor(n_splits=2)
    _feed_lap(p, [28_500, 58_200], 86_500)
    # Mid-lap: completed S1 in 28.0, query at 35s elapsed (mid S2).
    p.observe_split(1, 28_000)
    pred = p.predicted_lap_ms(elapsed_ms=35_000, last_split_idx=1)
    # elapsed (35000) + best segments 2 (29700) + best segment 3 (28300) = 93000
    assert pred == 35_000 + 29_700 + 28_300


def test_predicted_falls_back_to_best_lap_when_segments_missing():
    p = SplitPredictor(n_splits=2, best_lap_ms=86_500)
    # No segments observed → fallback path used.
    pred = p.predicted_lap_ms(elapsed_ms=10_000, last_split_idx=0)
    assert pred == 10_000 + 86_500


def test_observe_split_ignores_out_of_range():
    p = SplitPredictor(n_splits=2)
    p.observe_split(0, 10_000)         # invalid
    p.observe_split(3, 80_000)         # invalid (only 2 splits)
    p.observe_split(-1, 1_000)         # invalid
    p.observe_lap(-5)                  # invalid
    assert p.best_segments_ms == {}
    assert p.best_lap_ms is None


def test_delta_to_best_after_split():
    p = SplitPredictor(n_splits=2)
    _feed_lap(p, [28_500, 58_200], 86_500)
    # Current lap S1 a bit slower than PB
    p.observe_split(1, 28_900)
    delta = p.delta_to_best_ms(elapsed_ms=29_000, last_split_idx=1)
    # actual cum at S1 = 28_900, best cum at S1 = 28_500 → +400
    assert delta == 400


def test_roundtrip_persistence():
    p = SplitPredictor(n_splits=3)
    _feed_lap(p, [20_000, 40_000, 60_000], 80_000)
    blob = p.to_dict()
    q = SplitPredictor.from_dict(blob)
    assert q.n_splits == 3
    assert q.best_lap_ms == 80_000
    assert q.best_segments_ms == p.best_segments_ms
    assert q.spb_ms() == 80_000


def test_reset_lap_drops_transient_only():
    p = SplitPredictor(n_splits=2)
    _feed_lap(p, [28_500, 58_200], 86_500)
    p.observe_split(1, 50_000)        # absurdly slow
    p.reset_lap()                      # restart sim
    # Best segments must NOT include the absurd 50s (already worse than PB).
    assert p.best_segments_ms[1] == 28_500
    # Transient buffer cleared:
    assert p._current_splits_ms == {}
