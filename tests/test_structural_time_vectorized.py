"""Regression tests for structural time vectorisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from tnfr_core.operators.structural_time import (
    DEFAULT_STRUCTURAL_WEIGHTS,
    compute_structural_timestamps,
    _coerce_float,
    _normalise_weights,
)


@dataclass
class SampleRecord:
    brake_pressure: float
    throttle: float
    yaw_rate: float
    steer: float
    timestamp: float
    structural_timestamp: float | None = None


def _legacy_normalised_derivative(series: Sequence[float]) -> list[float]:
    derivatives = [0.0] * len(series)
    for index in range(1, len(series)):
        derivatives[index] = abs(series[index] - series[index - 1])
    peak = max(derivatives) if derivatives else 0.0
    if peak <= 1e-9:
        return derivatives
    return [value / peak for value in derivatives]


def _legacy_window_average(values: Sequence[float], index: int, window_size: int) -> float:
    start = max(0, index - window_size + 1)
    window = values[start : index + 1]
    if not window:
        return 0.0
    return sum(window) / len(window)


def _legacy_resolve_feature_series(
    records: Sequence[SampleRecord], feature: str
) -> list[float]:
    series: list[float] = []
    for record in records:
        numeric = _coerce_float(getattr(record, feature, 0.0))
        series.append(numeric if numeric is not None else 0.0)
    return series


def _legacy_compute_structural_timestamps(
    records: Sequence[SampleRecord],
    *,
    window_size: int = 5,
    weights: Mapping[str, float] | None = None,
    base_timestamp: float | None = None,
) -> list[float]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not records:
        return []

    weight_profile = _normalise_weights(weights or DEFAULT_STRUCTURAL_WEIGHTS)
    features = tuple(DEFAULT_STRUCTURAL_WEIGHTS)

    series_map = {feature: _legacy_resolve_feature_series(records, feature) for feature in features}
    derivative_map = {
        feature: _legacy_normalised_derivative(values) for feature, values in series_map.items()
    }

    densities: list[float] = []
    for index in range(len(records)):
        density = 0.0
        for feature in features:
            derivative = derivative_map[feature]
            if not derivative:
                continue
            density += weight_profile.get(feature, 0.0) * _legacy_window_average(
                derivative, index, window_size
            )
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


def _build_sample_records() -> list[SampleRecord]:
    return [
        SampleRecord(brake_pressure=10.0, throttle=0.1, yaw_rate=0.5, steer=0.0, timestamp=0.0),
        SampleRecord(brake_pressure=12.0, throttle=0.3, yaw_rate=0.6, steer=-0.1, timestamp=0.2),
        SampleRecord(brake_pressure=15.0, throttle=0.5, yaw_rate=0.7, steer=-0.2, timestamp=0.5),
        SampleRecord(brake_pressure=13.0, throttle=0.4, yaw_rate=0.75, steer=-0.15, timestamp=0.7),
        SampleRecord(brake_pressure=11.5, throttle=0.35, yaw_rate=0.72, steer=-0.1, timestamp=1.1),
    ]


def test_vectorised_structural_time_matches_legacy():
    records = _build_sample_records()
    expected = _legacy_compute_structural_timestamps(records, window_size=3)
    result = compute_structural_timestamps(records, window_size=3)
    assert np.allclose(result, expected)


def test_custom_weight_profile_matches_legacy_behaviour():
    records = _build_sample_records()
    weights = {"throttle": 0.6, "steer": 0.4}
    expected = _legacy_compute_structural_timestamps(
        records, window_size=4, weights=weights, base_timestamp=5.0
    )
    result = compute_structural_timestamps(records, window_size=4, weights=weights, base_timestamp=5.0)
    assert np.allclose(result, expected)
