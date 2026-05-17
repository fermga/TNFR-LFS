"""Network field snapshots: mean EPI / νf / phase aggregates.

A :class:`NetworkSnapshot` summarizes the canonical TNFR state of a
coupled track↔car graph in dimensionless, comparable scalars. The
advisor uses snapshots to (a) report "before / after" structural state
to the user in physics-grounded language and (b) feed the deterministic
ΔC surrogate in :mod:`coherence`.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import networkx as nx


@dataclass(frozen=True)
class NetworkSnapshot:
    """Aggregated TNFR field state of a graph.

    Attributes are population-level means restricted to finite values.
    All fields are safe to compare across snapshots of the same graph
    topology (e.g. baseline vs proposed setup).
    """

    n_nodes: int
    n_edges: int
    epi_mean: float
    epi_max: float
    vf_mean: float
    vf_max: float
    theta_mean: float
    kind_counts: dict[str, int]


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if isfinite(v)]


def network_snapshot(graph: nx.Graph) -> NetworkSnapshot:
    """Compute a :class:`NetworkSnapshot` of ``graph`` (read-only)."""
    epis: list[float] = []
    vfs: list[float] = []
    thetas: list[float] = []
    kinds: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        try:
            epis.append(float(data.get("EPI", float("nan"))))
            vfs.append(float(data.get("νf", float("nan"))))
            thetas.append(float(data.get("theta", float("nan"))))
        except (TypeError, ValueError):
            pass
        k = data.get("kind", "unknown")
        kinds[k] = kinds.get(k, 0) + 1
    epis_f = _finite(epis) or [0.0]
    vfs_f = _finite(vfs) or [0.0]
    thetas_f = _finite(thetas) or [0.0]
    return NetworkSnapshot(
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        epi_mean=float(sum(epis_f) / len(epis_f)),
        epi_max=float(max(epis_f)),
        vf_mean=float(sum(vfs_f) / len(vfs_f)),
        vf_max=float(max(vfs_f)),
        theta_mean=float(sum(thetas_f) / len(thetas_f)),
        kind_counts=dict(sorted(kinds.items())),
    )


__all__ = ("NetworkSnapshot", "network_snapshot")
