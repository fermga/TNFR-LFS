"""Car network: wheel.*, axle.*, brake.*, engine, chassis, driver nodes.

The car is modelled as a topological graph of subsystems that couple
through load paths (suspension, brake hydraulics, drivetrain, driver
inputs). Each subsystem becomes a TNFR node seeded from the enriched
telemetry through :class:`ObservableMapper`.

Edges (all undirected, weights are dimensionless coupling strengths):

* wheel.<c>  ↔ axle.<f|r>    weight = static load fraction of that
                              wheel on its axle (FL/FR → axle.front,
                              RL/RR → axle.rear).
* axle.<a>   ↔ brake.<a>     weight = 1.0 (rigid hydraulic coupling).
* axle.<a>   ↔ chassis       weight = static axle load fraction.
* chassis    ↔ engine        weight = 1.0 (drivetrain mount).
* chassis    ↔ driver        weight = 1.0 (steering column / pedals).
* wheel diagonals (FL↔RR, FR↔RL)   weight = 0.25 (cross-weight /
                                                anti-roll bridge).
"""
from __future__ import annotations

import networkx as nx
import pandas as pd

from tnfr import create_nfr

from lfs_telemetry.telemetry.observables import CarSpec

from .mapping import (
    FRONT_WHEELS,
    NodeSeed,
    ObservableMapper,
    REAR_WHEELS,
    WHEEL_ORDER,
)


def build_car_network(
    car: CarSpec,
    df_avg: pd.DataFrame,
    *,
    seed: int = 20260516,
) -> tuple[nx.Graph, list[str]]:
    """Build the car coupling network.

    Parameters
    ----------
    car
        Static car spec (mass, geometry, μ). Used both for the mapper
        reference values and for the static-load edge weights.
    df_avg
        Enriched DataFrame covering the full stint.
    seed
        Reserved for downstream RNG-driven ops.
    """
    mapper = ObservableMapper(car)
    graph: nx.Graph | None = None
    names: list[str] = []
    seeds: dict[str, NodeSeed] = {}

    def _add(node: NodeSeed, kind: str, **extra: object) -> None:
        nonlocal graph
        graph, nm = create_nfr(
            node.name, epi=float(node.epi), vf=float(node.vf),
            theta=float(node.theta), graph=graph,
        )
        graph.nodes[nm].update(kind=kind, meta=dict(node.meta), **extra)
        names.append(nm)
        seeds[nm] = node

    # ---- nodes ----------------------------------------------------
    for w in WHEEL_ORDER:
        _add(mapper.seed_wheel(w, df_avg), kind="wheel", wheel=w)
    for ax in ("front", "rear"):
        _add(mapper.seed_axle(ax, df_avg), kind="axle", axle=ax)
        _add(mapper.seed_brake(ax, df_avg), kind="brake", axle=ax)
    _add(mapper.seed_engine(df_avg), kind="engine")
    _add(mapper.seed_chassis(df_avg), kind="chassis")
    _add(mapper.seed_driver(df_avg), kind="driver")

    assert graph is not None

    # ---- edges ----------------------------------------------------
    statics = car.static_corner_loads_n()
    front_total = statics["FL"] + statics["FR"]
    rear_total = statics["RL"] + statics["RR"]
    axle_total = front_total + rear_total

    for w in FRONT_WHEELS:
        graph.add_edge(
            f"wheel.{w}", "axle.front",
            weight=float(statics[w] / front_total), kind="wheel_axle",
        )
    for w in REAR_WHEELS:
        graph.add_edge(
            f"wheel.{w}", "axle.rear",
            weight=float(statics[w] / rear_total), kind="wheel_axle",
        )
    for ax, total in (("front", front_total), ("rear", rear_total)):
        graph.add_edge(f"axle.{ax}", f"brake.{ax}", weight=1.0, kind="hydraulic")
        graph.add_edge(
            f"axle.{ax}", "chassis",
            weight=float(total / axle_total), kind="axle_chassis",
        )
    graph.add_edge("chassis", "engine", weight=1.0, kind="drivetrain")
    graph.add_edge("chassis", "driver", weight=1.0, kind="control")
    # Diagonal cross-weight links
    graph.add_edge("wheel.FL", "wheel.RR", weight=0.25, kind="diagonal")
    graph.add_edge("wheel.FR", "wheel.RL", weight=0.25, kind="diagonal")

    graph.graph["car_name"] = getattr(car, "name", None) or "unknown"
    graph.graph["kind"] = "car"
    graph.graph["seed"] = int(seed)
    return graph, names
