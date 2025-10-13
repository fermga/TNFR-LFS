"""Utilities for analysing ΔNFR coherence and entropy penalties."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Dict, Mapping

from tnfr_lfs.core.delta_utils import distribute_weighted_delta
from tnfr_lfs.core.phases import expand_phase_alias, phase_family
from tnfr_lfs.core.utils import normalised_entropy

__all__ = ["compute_node_delta_nfr", "sense_index"]


NodeDeltaMap = Dict[str, float]


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

    distribution = distribute_weighted_delta(delta_nfr, feature_map)
    if not distribution:
        return {}
    if prefix:
        return {f"{node}.{name}": value for name, value in distribution.items()}
    return distribution


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


def _goal_frequency_factor(
    goal_spec: Mapping[str, float] | float | None,
    node: str,
) -> float:
    if goal_spec is None:
        return 1.0
    if isinstance(goal_spec, MappingABC):
        target_value = _phase_weight(goal_spec, node)
    else:
        target_value = float(goal_spec)
    return _frequency_gain(target_value)


def sense_index(
    delta_nfr: float,
    deltas_by_node: Mapping[str, float],
    baseline_nfr: float,
    *,
    nu_f_by_node: Mapping[str, float],
    active_phase: str,
    w_phase: Mapping[str, Mapping[str, float] | float] | Mapping[str, float],
    nu_f_targets: Mapping[str, Mapping[str, float] | float]
    | Mapping[str, float]
    | float
    | None = None,
    entropy_lambda: float = 0.1,
) -> float:
    """Compute the entropy-penalised sense index for a ΔNFR distribution.

    The metric follows the expression ``1 / (1 + Σ w · |ΔNFR| · g(ν_f) · g*(ν_f^*))
    - λ·H`` combining phase-dependent weights ``w``, the magnitude of the ΔNFR
    contributions for every subsystem, the natural frequency gain ``g`` derived
    from the measured natural frequency ``ν_f`` and an optional goal gain
    ``g*`` driven by the target natural frequency ``ν_f^*``.  The entropy term
    ``H`` is derived from the distribution of the node deltas.
    ``λ`` is exposed as ``entropy_lambda`` for fine tuning while remaining
    backward compatible with previous heuristics.
    """

    phase_weights: Mapping[str, float] | float = 1.0
    if isinstance(w_phase, MappingABC):
        phase_weights = w_phase.get("__default__", 1.0)
        for candidate in (*expand_phase_alias(active_phase), phase_family(active_phase), active_phase):
            if candidate is None:
                continue
            if candidate in w_phase:
                phase_weights = w_phase[candidate]
                break

    phase_targets: Mapping[str, float] | float | None
    if isinstance(nu_f_targets, MappingABC):
        phase_targets = nu_f_targets.get("__default__")
        for candidate in (*expand_phase_alias(active_phase), phase_family(active_phase), active_phase):
            if candidate is None:
                continue
            if candidate in nu_f_targets:
                phase_targets = nu_f_targets[candidate]
                break
    else:
        phase_targets = nu_f_targets

    weighted_sum = 0.0
    for node, delta_value in deltas_by_node.items():
        node_weight = _phase_weight(phase_weights, node)
        nu_f = nu_f_by_node.get(node, 0.0)
        goal_factor = _goal_frequency_factor(phase_targets, node)
        weighted_sum += node_weight * goal_factor * abs(delta_value) * _frequency_gain(nu_f)

    base_index = 1.0 / (1.0 + weighted_sum)

    weights_total = sum(abs(value) for value in deltas_by_node.values())
    if weights_total == 0:
        return max(0.0, min(1.0, base_index))

    weights = [abs(value) / weights_total for value in deltas_by_node.values()]
    penalty = entropy_lambda * normalised_entropy(weights)
    adjusted = base_index - penalty
    return max(0.0, min(1.0, adjusted))

