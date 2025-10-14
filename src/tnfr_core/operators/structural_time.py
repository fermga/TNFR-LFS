"""Structural time helpers based on event density heuristics."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from tnfr_core.operators.interfaces import SupportsTelemetrySample

__all__ = ["compute_structural_timestamps", "resolve_time_axis"]

DEFAULT_STRUCTURAL_WEIGHTS: Mapping[str, float] = {
    "brake_pressure": 0.35,
    "throttle": 0.25,
    "yaw_rate": 0.25,
    "steer": 0.15,
}


def _normalise_weights(weights: Mapping[str, float]) -> Mapping[str, float]:
    positive = {feature: max(0.0, float(value)) for feature, value in weights.items()}
    total = sum(positive.values())
    if total <= 0.0:
        count = len(DEFAULT_STRUCTURAL_WEIGHTS)
        return {feature: 1.0 / count for feature in DEFAULT_STRUCTURAL_WEIGHTS}
    return {feature: value / total for feature, value in positive.items()}


def _coerce_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _resolve_feature_series(
    records: Sequence[SupportsTelemetrySample], feature: str
) -> np.ndarray:
    series: list[float] = []
    for record in records:
        numeric = _coerce_float(getattr(record, feature, 0.0))
        series.append(numeric if numeric is not None else 0.0)
    return np.asarray(series, dtype=float)


def _normalised_derivative(series: Sequence[float]) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    if values.size == 0:
        return values
    derivatives = np.zeros_like(values)
    if values.size > 1:
        derivatives[1:] = np.abs(np.diff(values))
    peak = float(np.max(derivatives)) if derivatives.size else 0.0
    if peak <= 1e-9:
        return derivatives
    return derivatives / peak


def _moving_average(values: Sequence[float], window_size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    kernel = np.ones(window_size, dtype=float)
    sums = np.convolve(arr, kernel, mode="full")[: arr.size]
    counts = np.minimum(np.arange(1, arr.size + 1, dtype=float), float(window_size))
    return sums / counts


def compute_structural_timestamps(
    records: Sequence[SupportsTelemetrySample],
    *,
    window_size: int = 5,
    weights: Mapping[str, float] | None = None,
    base_timestamp: float | None = None,
) -> list[float]:
    """Return a structural time axis derived from event density.

    The helper inspects a subset of high-activation telemetry channels and
    computes a moving average of their normalised derivatives.  The resulting
    event density inflates the elapsed time for dense segments while keeping
    quiet stretches close to the chronological axis.  The first timestamp is
    anchored to ``base_timestamp`` when provided, otherwise it reuses the
    earliest available reference in ``records``.
    """

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not records:
        return []

    weight_profile = _normalise_weights(weights or DEFAULT_STRUCTURAL_WEIGHTS)
    features = tuple(DEFAULT_STRUCTURAL_WEIGHTS)

    series_map = {feature: _resolve_feature_series(records, feature) for feature in features}
    derivative_matrix = np.stack(
        [_normalised_derivative(series_map[feature]) for feature in features],
        axis=0,
    )
    averages = np.stack(
        [_moving_average(derivative_matrix[index], window_size) for index in range(len(features))],
        axis=0,
    )
    weights_vector = np.asarray(
        [weight_profile.get(feature, 0.0) for feature in features], dtype=float
    )
    densities = np.minimum(averages.T @ weights_vector, 5.0)

    first_record = records[0]
    base = _coerce_float(base_timestamp)
    if base is None:
        base = _coerce_float(getattr(first_record, "structural_timestamp", None))
    if base is None:
        base = _coerce_float(first_record.timestamp)
    if base is None:
        base = 0.0

    chronological = np.asarray(
        [(_coerce_float(getattr(record, "timestamp", None)) or 0.0) for record in records],
        dtype=float,
    )
    deltas = np.diff(chronological, prepend=chronological[:1])
    deltas = np.maximum(deltas, 0.0)
    increments = deltas * (1.0 + densities)
    structural = base + np.cumsum(increments)

    return structural.tolist()


def resolve_time_axis(
    sequence: Sequence[object],
    *,
    fallback_to_chronological: bool = True,
    structural_attr: str = "structural_timestamp",
    chronological_attr: str = "timestamp",
) -> list[float] | None:
    """Return a monotonically increasing time axis for ``sequence``.

    When the structural axis is incomplete or not strictly ordered and
    ``fallback_to_chronological`` is true the chronological attribute is used
    instead.  ``None`` is returned when no valid axis can be extracted.
    """

    if not sequence:
        return []

    structural: list[float] | None = []
    for item in sequence:
        candidate = _coerce_float(getattr(item, structural_attr, None))
        if candidate is None:
            structural = None
            break
        structural.append(candidate)

    if structural and _is_monotonic(structural):
        return structural

    if not fallback_to_chronological:
        return None

    chronological: list[float] = []
    for item in sequence:
        candidate = _coerce_float(getattr(item, chronological_attr, None))
        if candidate is None:
            return None
        chronological.append(candidate)
    if not _is_monotonic(chronological):
        return None
    return chronological


def _is_monotonic(values: Sequence[float]) -> bool:
    return all(values[index] >= values[index - 1] for index in range(1, len(values)))

