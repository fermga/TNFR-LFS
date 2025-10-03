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


def compute_node_delta_nfr(delta_nfr: float, feature_map: Mapping[str, float]) -> NodeDeltaMap:
    """Distribute ΔNFR across the different nodes."""

    total = sum(abs(value) for value in feature_map.values())
    if total == 0:
        # fall back to a uniform distribution when no feature dominates
        weight = 1.0 / len(feature_map) if feature_map else 0.0
        return {name: delta_nfr * weight for name in feature_map}
    sign = 1.0 if delta_nfr >= 0 else -1.0
    magnitude = abs(delta_nfr)
    return {
        name: sign * magnitude * (abs(value) / total)
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

