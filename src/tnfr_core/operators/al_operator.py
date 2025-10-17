"""Coherence, coupling and resonance operators."""

from __future__ import annotations

from statistics import mean
from typing import Dict, List, Mapping, Sequence, Tuple
import warnings

import numpy as np

from tnfr_core.equations.contextual_delta import (
    apply_contextual_delta,
    load_context_matrix,
    resolve_series_context,
)
from tnfr_core.equations.dissonance import compute_useful_dissonance_stats
from tnfr_core.equations.phases import LEGACY_PHASE_MAP
from tnfr_core.runtime.shared import (
    SupportsChassisNode,
    SupportsEPIBundle,
    SupportsMicrosector,
    SupportsTyresNode,
    _HAS_JAX,
    jnp,
)
from tnfr_core.operators._types import DissonanceBreakdown


if _HAS_JAX:  # pragma: no cover - exercised only when JAX is installed
    xp_module = jnp
else:
    xp_module = np


_APEX_PHASE_CANDIDATES: Tuple[str, ...] = LEGACY_PHASE_MAP.get("apex", tuple())
if "apex" not in _APEX_PHASE_CANDIDATES:
    _APEX_PHASE_CANDIDATES = _APEX_PHASE_CANDIDATES + ("apex",)
def _zero_dissonance_breakdown() -> DissonanceBreakdown:
    """Return a :class:`DissonanceBreakdown` filled with zeros."""

    return DissonanceBreakdown(
        value=0.0,
        useful_magnitude=0.0,
        parasitic_magnitude=0.0,
        useful_ratio=0.0,
        parasitic_ratio=0.0,
        useful_percentage=0.0,
        parasitic_percentage=0.0,
        total_events=0,
        useful_events=0,
        parasitic_events=0,
        useful_dissonance_ratio=0.0,
        useful_dissonance_percentage=0.0,
        high_yaw_acc_samples=0,
        useful_dissonance_samples=0,
    )


def coherence_operator(series: Sequence[float], window: int = 3) -> List[float]:
    """Smooth a numeric series while preserving its average value."""

    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if not series:
        return []
    xp = xp_module
    array = xp.asarray(series, dtype=xp.float64)
    kernel = xp.ones(window, dtype=array.dtype)
    numerator = xp.convolve(array, kernel, mode="same")
    counts = xp.convolve(xp.ones_like(array), kernel, mode="same")
    smoothed = numerator / counts
    bias = float(array.mean() - smoothed.mean())
    if abs(bias) >= 1e-12:
        smoothed = smoothed + bias
    return np.asarray(smoothed, dtype=float).tolist()


def dissonance_operator(series: Sequence[float], target: float) -> float:
    """Compute the mean absolute deviation relative to a target value."""

    if not series:
        return 0.0
    xp = xp_module
    array = xp.asarray(series, dtype=xp.float64)
    differences = xp.abs(array - target)
    return float(xp.mean(differences))


def dissonance_breakdown_operator(
    series: Sequence[float],
    target: float,
    *,
    microsectors: Sequence[SupportsMicrosector] | None = None,
    bundles: Sequence[SupportsEPIBundle] | None = None,
) -> DissonanceBreakdown:
    """Classify support events into useful (positive) and parasitic dissonance."""

    base_value = dissonance_operator(series, target)
    useful_events = 0
    parasitic_events = 0
    useful_magnitude = 0.0
    parasitic_magnitude = 0.0
    useful_dissonance_samples = 0
    high_yaw_acc_samples = 0

    context_matrix = load_context_matrix()
    bundle_context = (
        resolve_series_context(bundles, matrix=context_matrix) if bundles else []
    )

    if microsectors and bundles:
        bundle_count = len(bundles)
        tyre_nodes: Sequence[SupportsTyresNode] = [bundle.tyres for bundle in bundles]
        for microsector in microsectors:
            if not microsector.support_event:
                continue
            apex_goal = None
            for apex_phase in _APEX_PHASE_CANDIDATES:
                apex_goal = next(
                    (goal for goal in microsector.goals if goal.phase == apex_phase),
                    None,
                )
                if apex_goal is not None:
                    break
            if apex_goal is None:
                continue
            apex_indices: List[int] = []
            for apex_phase in _APEX_PHASE_CANDIDATES:
                indices = microsector.phase_samples.get(apex_phase) or ()
                apex_indices.extend(idx for idx in indices if 0 <= idx < bundle_count)
            if not apex_indices:
                continue
            tyre_delta = []
            for idx in apex_indices:
                multiplier = 1.0
                if 0 <= idx < len(bundle_context):
                    multiplier = max(
                        context_matrix.min_multiplier,
                        min(
                            context_matrix.max_multiplier,
                            bundle_context[idx].multiplier,
                        ),
                    )
                tyre_delta.append(tyre_nodes[idx].delta_nfr * multiplier)
            if not tyre_delta:
                continue
            deviation = mean(tyre_delta) - apex_goal.target_delta_nfr
            contribution = abs(deviation)
            if contribution <= 1e-12:
                continue
            if deviation >= 0.0:
                useful_events += 1
                useful_magnitude += contribution
            else:
                parasitic_events += 1
                parasitic_magnitude += contribution

    useful_dissonance_ratio = 0.0
    if bundles:
        timestamps = [bundle.timestamp for bundle in bundles]
        delta_series = [
            apply_contextual_delta(
                bundle.delta_nfr,
                bundle_context[idx],
                context_matrix=context_matrix,
            )
            for idx, bundle in enumerate(bundles)
        ]
        chassis_nodes: Sequence[SupportsChassisNode] = [
            bundle.chassis for bundle in bundles
        ]
        yaw_rates = [node.yaw_rate for node in chassis_nodes]
        (
            useful_dissonance_samples,
            high_yaw_acc_samples,
            useful_dissonance_ratio,
        ) = compute_useful_dissonance_stats(timestamps, delta_series, yaw_rates)

    total_events = useful_events + parasitic_events
    total_magnitude = useful_magnitude + parasitic_magnitude
    if total_magnitude > 1e-12:
        useful_ratio = useful_magnitude / total_magnitude
        parasitic_ratio = parasitic_magnitude / total_magnitude
    elif total_events > 0:
        useful_ratio = useful_events / total_events
        parasitic_ratio = parasitic_events / total_events
    else:
        useful_ratio = 0.0
        parasitic_ratio = 0.0

    return DissonanceBreakdown(
        value=base_value,
        useful_magnitude=useful_magnitude,
        parasitic_magnitude=parasitic_magnitude,
        useful_ratio=useful_ratio,
        parasitic_ratio=parasitic_ratio,
        useful_percentage=useful_ratio * 100.0,
        parasitic_percentage=parasitic_ratio * 100.0,
        total_events=total_events,
        useful_events=useful_events,
        parasitic_events=parasitic_events,
        useful_dissonance_ratio=useful_dissonance_ratio,
        useful_dissonance_percentage=useful_dissonance_ratio * 100.0,
        high_yaw_acc_samples=high_yaw_acc_samples,
        useful_dissonance_samples=useful_dissonance_samples,
    )


def _prepare_series_pair(
    series_a: Sequence[float] | np.ndarray,
    series_b: Sequence[float] | np.ndarray,
    *,
    strict_length: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return truncated NumPy arrays for ``series_a`` and ``series_b``."""

    array_a = np.asarray(series_a, dtype=float).ravel()
    array_b = np.asarray(series_b, dtype=float).ravel()
    if strict_length and array_a.shape[0] != array_b.shape[0]:
        raise ValueError("series must have the same length")

    length = min(array_a.shape[0], array_b.shape[0])
    return array_a[:length], array_b[:length]


def _batch_coupling(
    values: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Compute coupling values for stacked series using vectorised operations."""

    xp = xp_module
    data = xp.asarray(values, dtype=xp.float64)
    if mask is None:
        mask_xp = xp.ones(data.shape[:-1], dtype=bool)
    else:
        mask_xp = xp.asarray(mask, dtype=bool)

    counts = mask_xp.sum(axis=1, keepdims=True)
    counts_float = counts.astype(data.dtype)
    safe_counts = xp.where(counts > 0, counts_float, xp.ones_like(counts_float))
    sums = xp.sum(xp.where(mask_xp[..., None], data, 0.0), axis=1, keepdims=True)
    means = xp.where(
        (counts > 0)[..., None],
        sums / safe_counts[..., None],
        0.0,
    )
    centered = xp.where(mask_xp[..., None], data - means, 0.0)
    covariance = xp.sum(centered[..., 0] * centered[..., 1], axis=1)
    variance_a = xp.sum(centered[..., 0] ** 2, axis=1)
    variance_b = xp.sum(centered[..., 1] ** 2, axis=1)
    denominator = xp.sqrt(variance_a * variance_b)

    if xp is np:
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(denominator > 0, covariance / denominator, 0.0)
    else:  # pragma: no cover - exercised only when JAX is available
        result = xp.where(denominator > 0, covariance / denominator, 0.0)
    return np.asarray(result, dtype=float)


def coupling_operator(
    series_a: Sequence[float], series_b: Sequence[float], *, strict_length: bool = True
) -> float:
    """Return the normalised coupling (correlation) between two series."""

    values_a, values_b = _prepare_series_pair(
        series_a, series_b, strict_length=strict_length
    )
    if values_a.size == 0:
        return 0.0

    stacked = np.stack((values_a, values_b), axis=-1)[np.newaxis, ...]
    return float(_batch_coupling(stacked)[0])


def acoplamiento_operator(
    series_a: Sequence[float], series_b: Sequence[float], *, strict_length: bool = True
) -> float:
    """Compatibility wrapper for :func:`coupling_operator`."""

    warnings.warn(
        "acoplamiento_operator has been renamed to coupling_operator; "
        "please update imports before the legacy name is removed.",
        DeprecationWarning,
        stacklevel=2,
    )
    return coupling_operator(series_a, series_b, strict_length=strict_length)


def pairwise_coupling_operator(
    series_by_node: Mapping[str, Sequence[float]],
    *,
    pairs: Sequence[tuple[str, str]] | None = None,
) -> Dict[str, float]:
    """Compute coupling metrics for each node pair using :func:`coupling_operator`."""

    if pairs is None:
        ordered_nodes = list(series_by_node.keys())
        pairs = [(a, b) for idx, a in enumerate(ordered_nodes) for b in ordered_nodes[idx + 1 :]]

    pairs = list(pairs)
    if not pairs:
        return {}

    prepared = [
        _prepare_series_pair(
            series_by_node.get(first, ()),
            series_by_node.get(second, ()),
            strict_length=False,
        )
        for first, second in pairs
    ]

    lengths = [values_a.shape[0] for values_a, _ in prepared]
    max_length = max(lengths, default=0)
    stacked = np.zeros((len(prepared), max_length, 2), dtype=float)
    mask = np.zeros((len(prepared), max_length), dtype=bool)

    for idx, ((values_a, values_b), length) in enumerate(zip(prepared, lengths)):
        if length == 0:
            continue
        stacked[idx, :length, 0] = values_a
        stacked[idx, :length, 1] = values_b
        mask[idx, :length] = True

    results = _batch_coupling(stacked, mask)
    return {
        f"{first}↔{second}": float(results[idx])
        for idx, (first, second) in enumerate(pairs)
    }


def resonance_operator(series: Sequence[float]) -> float:
    """Compute the root-mean-square (RMS) resonance of a series."""

    xp = xp_module
    array = xp.asarray(series, dtype=xp.float64)
    array = xp.ravel(array)
    if xp.size(array) == 0:
        return 0.0
    rms = xp.sqrt(xp.mean(array ** 2))
    return float(rms)


__all__ = [
    "DissonanceBreakdown",
    "_batch_coupling",
    "_prepare_series_pair",
    "_zero_dissonance_breakdown",
    "acoplamiento_operator",
    "coherence_operator",
    "coupling_operator",
    "dissonance_breakdown_operator",
    "dissonance_operator",
    "pairwise_coupling_operator",
    "resonance_operator",
]

