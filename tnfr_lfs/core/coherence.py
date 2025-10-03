"""Utilities for analysing ΔNFR coherence and entropy penalties."""

from __future__ import annotations

from math import log
from typing import Dict, Iterable, Mapping


NodeDeltaMap = Dict[str, float]


def _entropy(weights: Iterable[float]) -> float:
    filtered = [value for value in weights if value > 0]
    if not filtered:
        return 0.0
    entropy = -sum(value * log(value) for value in filtered)
    max_entropy = log(len(filtered)) if len(filtered) > 1 else 1.0
    if max_entropy == 0:
        return 0.0
    return min(1.0, entropy / max_entropy)


def compute_node_delta_nfr(
    node: str,
    delta_nfr: float,
    feature_map: Mapping[str, float],
    *,
    prefix: bool = True,
) -> NodeDeltaMap:
    """Distribute a node-level ΔNFR among its internal sub-features.

    Parameters
    ----------
    node:
        Name of the node whose signals are being analysed.  The node name is
        used to optionally prefix the resulting keys so that flattened maps do
        not collide when multiple subsystems expose the same signal name.
    delta_nfr:
        ΔNFR magnitude attributed to the node.  The sign of ``delta_nfr`` is
        preserved in the output while the magnitude is scaled according to the
        relative weights in ``feature_map``.
    feature_map:
        Mapping of sub-feature identifiers to their measured intensity.  The
        function uses absolute values for weighting, meaning the caller is
        responsible for carrying the directional information in
        ``delta_nfr``.
    prefix:
        When ``True`` (the default) each key is prefixed with ``"node."`` to
        avoid collisions across subsystems.
    """

    if not feature_map:
        return {}

    total = sum(abs(value) for value in feature_map.values())
    if total == 0:
        weight = delta_nfr / len(feature_map)
        return {
            (f"{node}.{name}" if prefix else name): weight
            for name in feature_map
        }

    sign = 1.0 if delta_nfr >= 0 else -1.0
    magnitude = abs(delta_nfr)
    return {
        (f"{node}.{name}" if prefix else name): sign * magnitude * (abs(value) / total)
        for name, value in feature_map.items()
    }


def sense_index(delta_nfr: float, node_deltas: Mapping[str, float], baseline_nfr: float) -> float:
    """Compute the entropy-penalised sense index for a ΔNFR distribution."""

    nfr_scale = abs(baseline_nfr) or 1.0
    base_index = 1.0 - min(1.0, abs(delta_nfr) / nfr_scale)
    weights_total = sum(abs(value) for value in node_deltas.values())
    if weights_total == 0:
        return max(0.0, min(1.0, base_index))
    weights = [abs(value) / weights_total for value in node_deltas.values()]
    penalty = _entropy(weights)
    adjusted = base_index * (1.0 - penalty)
    return max(0.0, min(1.0, adjusted))

