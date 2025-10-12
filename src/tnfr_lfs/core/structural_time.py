"""Structural time helpers based on event density heuristics."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .interfaces import SupportsTelemetrySample

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
) -> list[float]:
    series: list[float] = []
    for record in records:
        numeric = _coerce_float(getattr(record, feature, 0.0))
        series.append(numeric if numeric is not None else 0.0)
    return series


def _normalised_derivative(series: Sequence[float]) -> list[float]:
    derivatives = [0.0] * len(series)
    for index in range(1, len(series)):
        derivatives[index] = abs(series[index] - series[index - 1])
    peak = max(derivatives) if derivatives else 0.0
    if peak <= 1e-9:
        return derivatives
    return [value / peak for value in derivatives]


def _window_average(values: Sequence[float], index: int, window_size: int) -> float:
    start = max(0, index - window_size + 1)
    window = values[start : index + 1]
    if not window:
        return 0.0
    return sum(window) / len(window)


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
    derivative_map = {feature: _normalised_derivative(values) for feature, values in series_map.items()}

    densities: list[float] = []
    for index in range(len(records)):
        density = 0.0
        for feature in features:
            derivative = derivative_map[feature]
            if not derivative:
                continue
            density += weight_profile.get(feature, 0.0) * _window_average(derivative, index, window_size)
        densities.append(min(density, 5.0))

    first_record = records[0]
    base = _coerce_float(base_timestamp)
    if base is None:
        base = _coerce_float(getattr(first_record, "structural_timestamp", None))
    if base is None:
        base = _coerce_float(first_record.timestamp)
    if base is None:
        base = 0.0

    structural_timestamps = [base]
    for index in range(1, len(records)):
        previous = records[index - 1]
        current = records[index]
        chronological_dt = _coerce_float(current.timestamp) or 0.0
        chronological_prev = _coerce_float(previous.timestamp) or 0.0
        dt = max(0.0, chronological_dt - chronological_prev)
        increment = dt * (1.0 + densities[index])
        structural_timestamps.append(structural_timestamps[-1] + increment)

    return structural_timestamps


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

