"""Utilities for analysing ΔNFR coherence and entropy penalties."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
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


def _frequency_gain(nu_f: float) -> float:
    """Return the monotonic gain applied to ΔNFR magnitudes.

    The gain models how subsystems with higher natural frequencies modulate the
    perceived coherence.  The relationship is intentionally simple—``g(ν_f) =
    1 + max(0, ν_f)``—yet strictly increasing so that faster subsystems reduce
    the sense index more aggressively than slower ones.
    """

    if not nu_f or not isinstance(nu_f, (int, float)):
        return 1.0
    return 1.0 + max(0.0, float(nu_f))


def _phase_weight(weights: Mapping[str, float] | float, node: str) -> float:
    if isinstance(weights, MappingABC):
        if node in weights:
            return float(weights[node])
        if "__default__" in weights:
            return float(weights["__default__"])
        return 1.0
    return float(weights)


def sense_index(
    delta_nfr: float,
    deltas_by_node: Mapping[str, float],
    baseline_nfr: float,
    *,
    nu_f_by_node: Mapping[str, float],
    active_phase: str,
    w_phase: Mapping[str, Mapping[str, float] | float] | Mapping[str, float],
    entropy_lambda: float = 0.1,
) -> float:
    """Compute the entropy-penalised sense index for a ΔNFR distribution.

    The metric follows the expression ``1 / (1 + Σ w · |ΔNFR| · g(ν_f)) - λ·H``
    combining phase-dependent weights ``w``, the magnitude of the ΔNFR
    contributions for every subsystem, the natural frequency gain ``g`` and the
    entropy term ``H`` derived from the distribution of the node deltas.
    ``λ`` is exposed as ``entropy_lambda`` for fine tuning while remaining
    backward compatible with previous heuristics.
    """

    phase_weights = 1.0
    if isinstance(w_phase, MappingABC):
        phase_weights = w_phase.get(active_phase, w_phase.get("__default__", 1.0))

    weighted_sum = 0.0
    for node, delta_value in deltas_by_node.items():
        node_weight = _phase_weight(phase_weights, node)
        nu_f = nu_f_by_node.get(node, 0.0)
        weighted_sum += node_weight * abs(delta_value) * _frequency_gain(nu_f)

    base_index = 1.0 / (1.0 + weighted_sum)

    weights_total = sum(abs(value) for value in deltas_by_node.values())
    if weights_total == 0:
        return max(0.0, min(1.0, base_index))

    weights = [abs(value) / weights_total for value in deltas_by_node.values()]
    penalty = entropy_lambda * _entropy(weights)
    adjusted = base_index - penalty
    return max(0.0, min(1.0, adjusted))

