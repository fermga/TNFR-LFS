"""Coupling: merge track and car networks with weighted ``corner↔wheel`` edges.

For every (sector, phase) corner node and every wheel, an edge is added
weighted by the mean vertical load on that wheel inside that distance
slice (normalized by the wheel's static corner load → dimensionless
ratio in roughly [0, 2]). Where the wheel-load column is missing the
edge is omitted.

The result is a NEW :class:`networkx.Graph` (the inputs are untouched);
it copies node attributes from both subgraphs and the union of edges.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from lfs_telemetry.telemetry.observables import CarSpec

from .mapping import WHEEL_ORDER


def couple_track_and_car(
    g_track: nx.Graph,
    g_car: nx.Graph,
    df_avg: pd.DataFrame,
    car: CarSpec,
) -> nx.Graph:
    """Compose ``g_track`` and ``g_car`` and add corner↔wheel edges.

    Parameters
    ----------
    g_track
        Output of :func:`build_track_network`. Each corner node carries
        ``sector_id``, ``phase`` and ``meta['dist_lo_m','dist_hi_m']``.
    g_car
        Output of :func:`build_car_network`.
    df_avg
        Enriched DataFrame covering the full stint (same as used for
        building both subgraphs).
    car
        :class:`CarSpec` (for static loads and normalization).
    """
    g = nx.compose(g_track, g_car)  # node attrs from g_car win on conflict
    g.graph["kind"] = "coupled"
    g.graph["track_code"] = g_track.graph.get("track_code")
    g.graph["car_name"] = g_car.graph.get("car_name")

    statics = car.static_corner_loads_n()
    have_dist = "current_lap_dist_m" in df_avg.columns
    d = df_avg["current_lap_dist_m"] if have_dist else None

    for node, data in g_track.nodes(data=True):
        if data.get("kind") != "corner":
            continue
        meta = data.get("meta", {})
        lo = float(meta.get("dist_lo_m", float("nan")))
        hi = float(meta.get("dist_hi_m", float("nan")))
        if not (have_dist and np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            continue
        mask = (d >= lo) & (d <= hi)
        if not mask.any():
            continue
        for w in WHEEL_ORDER:
            col = f"wheel_{w}_vertical_load_n"
            if col not in df_avg.columns:
                continue
            load = pd.to_numeric(df_avg.loc[mask, col], errors="coerce")
            mean_load = float(load.mean())
            if not np.isfinite(mean_load):
                continue
            weight = mean_load / statics[w]
            g.add_edge(node, f"wheel.{w}", weight=weight, kind="corner_wheel")
    return g
