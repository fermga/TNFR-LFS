"""Phase 2: filter_consecutive_laps tests using synthetic v2 stints."""
from __future__ import annotations

from pathlib import Path

import pytest

from lfs_telemetry.telemetry.lap import LapTelemetry
from lfs_telemetry.tnfr_racing.lap_filters import (
    StintFilterResult,
    filter_consecutive_laps,
)

ASSETS = Path(__file__).resolve().parents[2] / "assets"
BL1_LAPS = sorted(ASSETS.glob("synthetic_BL1_FBM_v2_lap*.csv"))
AS3_LAPS = sorted(ASSETS.glob("synthetic_AS3_FOX_v2_lap*.csv"))


def _load(p: Path) -> LapTelemetry:
    return LapTelemetry.from_csv(p)


@pytest.fixture(scope="module")
def bl1_laps() -> list[LapTelemetry]:
    assert len(BL1_LAPS) == 5, f"expected 5 BL1 laps, got {len(BL1_LAPS)}"
    return [_load(p) for p in BL1_LAPS]


@pytest.fixture(scope="module")
def as3_laps() -> list[LapTelemetry]:
    assert len(AS3_LAPS) == 5, f"expected 5 AS3 laps, got {len(AS3_LAPS)}"
    return [_load(p) for p in AS3_LAPS]


def test_accepts_5_consecutive_bl1_laps(bl1_laps: list[LapTelemetry]) -> None:
    res = filter_consecutive_laps(bl1_laps, min_count=5)
    assert res.ok, f"expected ok, got {res.reason}; rejected={res.rejected}"
    assert len(res.laps) == 5


def test_rejects_too_few_laps(bl1_laps: list[LapTelemetry]) -> None:
    res = filter_consecutive_laps(bl1_laps[:4], min_count=5)
    assert not res.ok
    assert "4" in res.reason


def test_rejects_mixed_cars(
    bl1_laps: list[LapTelemetry], as3_laps: list[LapTelemetry]
) -> None:
    mixed = bl1_laps[:3] + as3_laps[:2]
    res = filter_consecutive_laps(mixed, min_count=5)
    assert not res.ok
    # The longest contiguous run is 3 (BL1) before the car_mismatch break.
    assert any(r == "car_mismatch" for _, r in res.rejected)


def test_empty_input() -> None:
    res = filter_consecutive_laps([], min_count=5)
    assert isinstance(res, StintFilterResult)
    assert res.reason == "empty_input"
