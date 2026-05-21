"""Tests for the multi-mode lap-average helper."""

from __future__ import annotations

from lfs_telemetry.telemetry.lap_averages import (
    CLEAN_THRESHOLD,
    compute_lap_averages,
)


def test_empty_returns_none_triplet() -> None:
    out = compute_lap_averages([])
    assert out == {"stint": None, "clean": None, "total": None}


def test_total_is_arithmetic_mean() -> None:
    out = compute_lap_averages([90_000, 92_000, 91_000])
    assert out["total"] == 91_000


def test_clean_excludes_laps_above_103pct() -> None:
    best = 90_000
    slow = int(best * CLEAN_THRESHOLD) + 500  # outside threshold
    keep = int(best * CLEAN_THRESHOLD) - 500  # inside threshold
    out = compute_lap_averages([best, keep, slow, best])
    # slow lap excluded → mean of [best, keep, best]
    expected = round((best + keep + best) / 3)
    assert out["clean"] == expected


def test_stint_excludes_lap1_pit_in_and_outlap() -> None:
    # 6 laps: L1 out-lap, L2 flying, L3 flying, L4 in-lap (pit),
    # L5 out-lap (post-pit), L6 flying.
    laps = [95_000, 90_000, 90_500, 96_000, 95_500, 90_200]
    out = compute_lap_averages(laps, pit_in_laps=[4])
    # stint keeps L2, L3, L6
    expected = round((90_000 + 90_500 + 90_200) / 3)
    assert out["stint"] == expected
    # total includes everything
    assert out["total"] == round(sum(laps) / len(laps))


def test_stint_none_when_only_lap1() -> None:
    out = compute_lap_averages([90_000])
    assert out["stint"] is None
    assert out["total"] == 90_000
    assert out["clean"] == 90_000


def test_non_positive_lap_times_are_dropped() -> None:
    out = compute_lap_averages([0, -5, 90_000, 91_000])
    # Only the two positive laps remain → both are also "lap 1 & 2"
    # after filtering, so stint = mean of laps after skipping lap 1 = 91000.
    assert out["total"] == 90_500
    assert out["stint"] == 91_000
