"""Track network: ``corner.<sid>.{entry,apex,exit}`` nodes per sector.

The track is decomposed into sectors (caller supplies them — typically
from :func:`lfs_telemetry.telemetry.sectors.lap_sectors`). Each sector
spawns three TNFR nodes (entry, apex, exit) seeded from the enriched
telemetry through :class:`ObservableMapper`. Edges:

* **Sequential** corner→corner inside a sector and across the sector
  loop (weight 1.0) — the lap is a directed cycle in physical space
  but TNFR couplings are undirected.
* **Apex↔apex similarity** edges are reserved for v2 (requires per-
  corner curvature signatures we do not yet extract from telemetry).
  Hook is wired but currently a no-op.

Returns a plain :class:`networkx.Graph` (the same type returned by
:func:`tnfr.create_nfr`) plus the ordered node-name list.
"""
from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import pandas as pd

from tnfr import create_nfr

from lfs_telemetry.telemetry.observables import CarSpec
from lfs_telemetry.telemetry.sectors import Sector

from .mapping import NodeSeed, ObservableMapper

PHASES: tuple[str, ...] = ("entry", "apex", "exit")


def build_track_network(
    track_code: str,
    sectors: Sequence[Sector],
    df_avg: pd.DataFrame,
    car: CarSpec,
    *,
    seed: int = 20260516,
) -> tuple[nx.Graph, list[str]]:
    """Build the directed-flow track network.

    Parameters
    ----------
    track_code
        LFS track short code (e.g. ``"BL1"``). Stored as graph attribute.
    sectors
        Ordered list of :class:`Sector`. The sector index becomes the
        ``sid`` in node names.
    df_avg
        Enriched DataFrame covering the full stint (concatenated laps).
        Must contain ``current_lap_dist_m``.
    car
        :class:`CarSpec` used to seed :class:`ObservableMapper`.
    seed
        Reserved for downstream RNG-driven ops (no randomness used here).
    """
    if not sectors:
        raise ValueError("build_track_network requires >= 1 sector")
    if "current_lap_dist_m" not in df_avg.columns:
        raise KeyError("df_avg missing 'current_lap_dist_m'")

    mapper = ObservableMapper(car)
    graph: nx.Graph | None = None
    names: list[str] = []
    seeds_by_name: dict[str, NodeSeed] = {}

    for sec in sectors:
        for phase in PHASES:
            seed_node = mapper.seed_corner(
                sector_id=sec.index, phase=phase, df=df_avg,
                sector_start_m=sec.start_d_m, sector_end_m=sec.end_d_m,
            )
            graph, name = create_nfr(
                seed_node.name,
                epi=float(seed_node.epi),
                vf=float(seed_node.vf),
                theta=float(seed_node.theta),
                graph=graph,
            )
            graph.nodes[name].update(
                kind="corner", sector_id=sec.index, phase=phase,
                meta=dict(seed_node.meta),
            )
            names.append(name)
            seeds_by_name[name] = seed_node

    assert graph is not None  # at least 1 sector

    # Sequential edges: corner.s.entry -> corner.s.apex -> corner.s.exit
    # -> corner.(s+1).entry … wrapping back to corner.0.entry.
    for i in range(len(names)):
        a, b = names[i], names[(i + 1) % len(names)]
        graph.add_edge(a, b, weight=1.0, kind="sequential")

    graph.graph["track_code"] = track_code
    graph.graph["kind"] = "track"
    graph.graph["seed"] = int(seed)
    return graph, names
