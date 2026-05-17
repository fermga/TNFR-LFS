"""Phase 3: network builders + coupling, validated on synthetic v2 stints."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from lfs_telemetry.telemetry.lap import LapTelemetry
from lfs_telemetry.telemetry.sectors import lap_sectors
from lfs_telemetry.tnfr_racing.coupling import couple_track_and_car
from lfs_telemetry.tnfr_racing.network_car import build_car_network
from lfs_telemetry.tnfr_racing.network_track import build_track_network

ASSETS = Path(__file__).resolve().parents[2] / "assets"


@pytest.fixture(scope="module")
def bl1_stint() -> tuple[pd.DataFrame, list, "LapTelemetry"]:
    paths = sorted(ASSETS.glob("synthetic_BL1_FBM_v2_lap*.csv"))
    laps = [LapTelemetry.from_csv(p) for p in paths]
    # Concatenate enriched DataFrames as the stint-averaged frame.
    df = pd.concat([lap.enriched for lap in laps], ignore_index=True)
    sectors = lap_sectors(laps[0], n_equal=3)
    return df, sectors, laps[0]


def _assert_node_attrs(g: nx.Graph) -> None:
    for name, data in g.nodes(data=True):
        assert "EPI" in data, f"{name} missing EPI"
        assert "νf" in data, f"{name} missing νf"
        assert "theta" in data, f"{name} missing theta"
        assert 0.0 <= data["EPI"] <= 1.0
        assert data["νf"] > 0.0


def test_build_track_network_bl1(bl1_stint) -> None:
    df, sectors, lap = bl1_stint
    g, names = build_track_network("BL1", sectors, df, lap.car, seed=17)
    assert isinstance(g, nx.Graph)
    assert g.graph["track_code"] == "BL1"
    assert g.graph["kind"] == "track"
    # 3 sectors × 3 phases = 9 corner nodes
    assert len(names) == 9 == g.number_of_nodes()
    # Sequential ring → exactly N edges
    assert g.number_of_edges() == 9
    _assert_node_attrs(g)
    for _, _, d in g.edges(data=True):
        assert d["kind"] == "sequential"
        assert d["weight"] == 1.0


def test_build_car_network_fbm(bl1_stint) -> None:
    df, _, lap = bl1_stint
    g, names = build_car_network(lap.car, df, seed=17)
    # 4 wheels + 2 axles + 2 brakes + engine + chassis + driver = 11
    assert len(names) == 11 == g.number_of_nodes()
    _assert_node_attrs(g)
    # Expected edges: 4 wheel↔axle + 2 axle↔brake + 2 axle↔chassis
    # + 2 chassis↔(engine/driver) + 2 diagonal = 12
    assert g.number_of_edges() == 12
    # Static-load weights front: FL and FR each = 0.5 on 50/50 axle.
    assert g["wheel.FL"]["axle.front"]["weight"] == pytest.approx(0.5)
    assert g["wheel.FR"]["axle.front"]["weight"] == pytest.approx(0.5)


def test_couple_track_and_car(bl1_stint) -> None:
    df, sectors, lap = bl1_stint
    gt, nt = build_track_network("BL1", sectors, df, lap.car, seed=17)
    gc, nc = build_car_network(lap.car, df, seed=17)
    g = couple_track_and_car(gt, gc, df, lap.car)
    assert g.graph["kind"] == "coupled"
    assert g.number_of_nodes() == len(nt) + len(nc)
    # Inputs untouched
    assert gt.number_of_edges() == 9
    assert gc.number_of_edges() == 12
    # Every corner node gets up to 4 wheel edges → 9×4 = 36 new edges max.
    corner_wheel = [
        (u, v) for u, v, d in g.edges(data=True) if d.get("kind") == "corner_wheel"
    ]
    assert len(corner_wheel) == 9 * 4
    # Weights must be finite and roughly within [0.1, 2.0] (load ratio).
    for u, v, d in g.edges(data=True):
        if d.get("kind") == "corner_wheel":
            assert 0.1 <= d["weight"] <= 2.5, f"{u}-{v}: {d['weight']}"


def test_reproducibility(bl1_stint) -> None:
    df, sectors, lap = bl1_stint
    a, _ = build_car_network(lap.car, df, seed=17)
    b, _ = build_car_network(lap.car, df, seed=17)
    for n in a.nodes:
        assert a.nodes[n]["EPI"] == b.nodes[n]["EPI"]
        assert a.nodes[n]["νf"] == b.nodes[n]["νf"]
