"""Phase 2: ObservableMapper smoke + contract tests on synthetic v2 stints.

Verifies that every seed_* method returns a finite NodeSeed with:
* epi ∈ [0, 1]
* vf ∈ [0.5, 25] Hz (the analysis band)
* theta ∈ (-π, π]
* deterministic across repeated calls.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from lfs_telemetry.telemetry.lap import LapTelemetry
from lfs_telemetry.tnfr_racing.mapping import (
    NodeSeed,
    ObservableMapper,
    WHEEL_ORDER,
)

ASSETS = Path(__file__).resolve().parents[2] / "assets"


@pytest.fixture(scope="module")
def bl1_first() -> tuple[pd.DataFrame, ObservableMapper]:
    p = sorted(ASSETS.glob("synthetic_BL1_FBM_v2_lap*.csv"))[0]
    lap = LapTelemetry.from_csv(p)
    return lap.enriched, ObservableMapper(lap.car)


def _assert_seed_ok(seed: NodeSeed) -> None:
    assert isinstance(seed, NodeSeed)
    assert math.isfinite(seed.epi)
    assert 0.0 <= seed.epi <= 1.0, f"{seed.name}: epi={seed.epi}"
    assert math.isfinite(seed.vf)
    assert 0.0 < seed.vf <= 25.0, f"{seed.name}: vf={seed.vf}"
    assert math.isfinite(seed.theta)
    assert -math.pi - 1e-9 <= seed.theta <= math.pi + 1e-9


@pytest.mark.parametrize("wheel", WHEEL_ORDER)
def test_seed_wheel_contract(
    bl1_first: tuple[pd.DataFrame, ObservableMapper], wheel: str
) -> None:
    df, mapper = bl1_first
    _assert_seed_ok(mapper.seed_wheel(wheel, df))


@pytest.mark.parametrize("axle", ["front", "rear"])
def test_seed_axle_contract(
    bl1_first: tuple[pd.DataFrame, ObservableMapper], axle: str
) -> None:
    df, mapper = bl1_first
    _assert_seed_ok(mapper.seed_axle(axle, df))


@pytest.mark.parametrize("axle", ["front", "rear"])
def test_seed_brake_contract(
    bl1_first: tuple[pd.DataFrame, ObservableMapper], axle: str
) -> None:
    df, mapper = bl1_first
    _assert_seed_ok(mapper.seed_brake(axle, df))


def test_seed_engine_chassis_driver(
    bl1_first: tuple[pd.DataFrame, ObservableMapper]
) -> None:
    df, mapper = bl1_first
    _assert_seed_ok(mapper.seed_engine(df))
    _assert_seed_ok(mapper.seed_chassis(df))
    _assert_seed_ok(mapper.seed_driver(df))


@pytest.mark.parametrize("phase", ["entry", "apex", "exit"])
def test_seed_corner_contract(
    bl1_first: tuple[pd.DataFrame, ObservableMapper], phase: str
) -> None:
    df, mapper = bl1_first
    lap_len = float(df["current_lap_dist_m"].max())
    seed = mapper.seed_corner(
        sector_id=0, phase=phase, df=df,
        sector_start_m=0.0, sector_end_m=lap_len / 3.0,
    )
    _assert_seed_ok(seed)
    assert seed.meta["phase"] == phase


def test_reproducibility(bl1_first: tuple[pd.DataFrame, ObservableMapper]) -> None:
    df, mapper = bl1_first
    a = mapper.seed_chassis(df)
    b = mapper.seed_chassis(df)
    assert (a.epi, a.vf, a.theta) == (b.epi, b.vf, b.theta)


def test_physical_signatures(
    bl1_first: tuple[pd.DataFrame, ObservableMapper]
) -> None:
    """The v2 generator injects:
    * RR thermal overload (~146 °C) — RR temp_mean > FL temp_mean
    * RWD power-on slip — rear axle EPI's slip-diff > front axle's
    """
    df, mapper = bl1_first
    fl = mapper.seed_wheel("FL", df)
    rr = mapper.seed_wheel("RR", df)
    assert rr.meta["temp_mean_c"] > fl.meta["temp_mean_c"] + 5.0, (
        f"expected RR hotter than FL by >5C, got FL={fl.meta['temp_mean_c']:.1f} "
        f"RR={rr.meta['temp_mean_c']:.1f}"
    )
