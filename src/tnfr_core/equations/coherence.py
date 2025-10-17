"""Utilities for analysing ΔNFR coherence and entropy penalties."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Any, Dict, Mapping

import numpy as np

try:  # pragma: no cover - exercised indirectly when operators import is deferred
    from tnfr_core.operators._shared import _HAS_JAX, jnp
except ImportError:  # pragma: no cover - fallback path for circular imports
    _HAS_JAX = False
    jnp = None

from .delta_utils import distribute_weighted_delta
from .phases import expand_phase_alias, phase_family
from .utils import normalised_entropy

__all__ = ["compute_node_delta_nfr", "sense_index"]


NodeDeltaMap = Dict[str, float]

xp = jnp if _HAS_JAX and jnp is not None else np


def _ensure_backend() -> None:
    global _HAS_JAX, jnp, xp
    if _HAS_JAX and jnp is not None:
        return
    try:
        from tnfr_core.operators._shared import _HAS_JAX as resolved_has_jax, jnp as resolved_jnp
    except ImportError:  # pragma: no cover - defensive fallback
        return
    _HAS_JAX = resolved_has_jax
    jnp = resolved_jnp
    if _HAS_JAX and jnp is not None:
        xp = jnp


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


def _frequency_gain(nu_f: Any) -> Any:
    """Return the monotonic gain applied to ΔNFR magnitudes.

    The gain models how subsystems with higher natural frequencies modulate the
    perceived coherence.  The relationship is intentionally simple—``g(ν_f) =
    1 + max(0, ν_f)``—yet strictly increasing so that faster subsystems reduce
    the sense index more aggressively than slower ones.
    """

    _ensure_backend()
    try:
        values = xp.asarray(nu_f, dtype=float)
    except (TypeError, ValueError):
        return xp.asarray(1.0, dtype=float)

    finite_mask = xp.isfinite(values)
    safe_values = xp.where(finite_mask, values, 0.0)
    gains = xp.asarray(1.0, dtype=float) + xp.maximum(0.0, safe_values)
    return xp.where(finite_mask, gains, xp.asarray(1.0, dtype=float))


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

    _ensure_backend()
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

    nodes: tuple[str, ...] = tuple(deltas_by_node)

    abs_delta = (
        xp.abs(xp.asarray([deltas_by_node[node] for node in nodes], dtype=float))
        if nodes
        else xp.asarray([], dtype=float)
    )
    node_weights = (
        xp.asarray([_phase_weight(phase_weights, node) for node in nodes], dtype=float)
        if nodes
        else xp.asarray([], dtype=float)
    )
    nu_f_values = (
        xp.asarray([nu_f_by_node.get(node, 0.0) for node in nodes], dtype=float)
        if nodes
        else xp.asarray([], dtype=float)
    )
    natural_gain = _frequency_gain(nu_f_values)
    if nodes and phase_targets is not None:
        if isinstance(phase_targets, MappingABC):
            phase_target_values = xp.asarray(
                [_phase_weight(phase_targets, node) for node in nodes],
                dtype=float,
            )
        else:
            scalar_target = float(phase_targets)
            phase_target_values = xp.full(len(nodes), scalar_target, dtype=float)
        goal_factors = _frequency_gain(phase_target_values)
    else:
        goal_factors = (
            xp.ones(len(nodes), dtype=float) if nodes else xp.asarray([], dtype=float)
        )

    weighted_components = node_weights * goal_factors * abs_delta * natural_gain
    base_denominator = xp.asarray(1.0, dtype=float) + xp.sum(weighted_components)
    base_index = xp.asarray(1.0, dtype=float) / base_denominator

    weights_total = xp.sum(abs_delta)
    if not nodes or float(weights_total) <= 0.0:
        return float(xp.clip(base_index, 0.0, 1.0))

    normalised_weights = abs_delta / weights_total
    penalty = xp.asarray(
        entropy_lambda * normalised_entropy(normalised_weights),
        dtype=float,
    )
    adjusted = base_index - penalty
    return float(xp.clip(adjusted, 0.0, 1.0))

