"""High-level TNFR-LFS operators for telemetry analytics pipelines."""

from __future__ import annotations

from dataclasses import replace
from math import sqrt
from statistics import mean
from typing import Dict, List, Mapping, Sequence

from .epi import EPIExtractor, TelemetryRecord
from .epi_models import EPIBundle


def emission_operator(target_delta_nfr: float, target_sense_index: float) -> Dict[str, float]:
    """Return normalised objectives for ΔNFR and sense index targets."""

    target_si = max(0.0, min(1.0, target_sense_index))
    return {"delta_nfr": float(target_delta_nfr), "sense_index": target_si}


def recepcion_operator(
    records: Sequence[TelemetryRecord], extractor: EPIExtractor | None = None
) -> List[EPIBundle]:
    """Convert raw telemetry records into EPI bundles."""

    if not records:
        return []
    extractor = extractor or EPIExtractor()
    return extractor.extract(records)


def coherence_operator(series: Sequence[float], window: int = 3) -> List[float]:
    """Smooth a numeric series while preserving its average value."""

    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if not series:
        return []
    half_window = window // 2
    smoothed: List[float] = []
    for index in range(len(series)):
        start = max(0, index - half_window)
        end = min(len(series), index + half_window + 1)
        smoothed.append(mean(series[start:end]))
    original_mean = mean(series)
    smoothed_mean = mean(smoothed)
    bias = original_mean - smoothed_mean
    if abs(bias) < 1e-12:
        return smoothed
    return [value + bias for value in smoothed]


def dissonance_operator(series: Sequence[float], target: float) -> float:
    """Compute the mean absolute deviation relative to a target value."""

    if not series:
        return 0.0
    return mean(abs(value - target) for value in series)


def acoplamiento_operator(series_a: Sequence[float], series_b: Sequence[float]) -> float:
    """Return the normalised coupling (correlation) between two series."""

    if len(series_a) != len(series_b):
        raise ValueError("series must have the same length")
    if not series_a:
        return 0.0
    mean_a = mean(series_a)
    mean_b = mean(series_b)
    covariance = sum((a - mean_a) * (b - mean_b) for a, b in zip(series_a, series_b))
    variance_a = sum((a - mean_a) ** 2 for a in series_a)
    variance_b = sum((b - mean_b) ** 2 for b in series_b)
    if variance_a == 0 or variance_b == 0:
        return 0.0
    return covariance / sqrt(variance_a * variance_b)


def resonance_operator(series: Sequence[float]) -> float:
    """Compute the root-mean-square (RMS) resonance of a series."""

    if not series:
        return 0.0
    return sqrt(mean(value * value for value in series))


def recursividad_operator(
    series: Sequence[float], *, seed: float = 0.0, decay: float = 0.5
) -> List[float]:
    """Apply a recursive filter to a series to capture hysteresis effects."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in the [0, 1) interval")
    if not series:
        return []
    state = seed
    trace: List[float] = []
    for value in series:
        state = (decay * state) + ((1.0 - decay) * value)
        trace.append(state)
    return trace


def _update_bundles(
    bundles: Sequence[EPIBundle],
    delta_series: Sequence[float],
    si_series: Sequence[float],
) -> List[EPIBundle]:
    updated: List[EPIBundle] = []
    for bundle, delta_value, si_value in zip(bundles, delta_series, si_series):
        updated.append(
            replace(
                bundle,
                delta_nfr=delta_value,
                sense_index=max(0.0, min(1.0, si_value)),
            )
        )
    return updated


def orchestrate_delta_metrics(
    telemetry_segments: Sequence[Sequence[TelemetryRecord]],
    target_delta_nfr: float,
    target_sense_index: float,
    *,
    coherence_window: int = 3,
    recursion_decay: float = 0.4,
) -> Mapping[str, object]:
    """Pipeline orchestration producing aggregated ΔNFR and Si metrics."""

    objectives = emission_operator(target_delta_nfr, target_sense_index)
    bundles: List[EPIBundle] = []
    for segment in telemetry_segments:
        bundles.extend(recepcion_operator(segment))
    if not bundles:
        return {
            "objectives": objectives,
            "bundles": [],
            "delta_nfr_series": [],
            "sense_index_series": [],
            "delta_nfr": 0.0,
            "sense_index": 0.0,
            "dissonance": 0.0,
            "coupling": 0.0,
            "resonance": 0.0,
            "recursive_trace": [],
        }

    delta_series = [bundle.delta_nfr for bundle in bundles]
    si_series = [bundle.sense_index for bundle in bundles]
    smoothed_delta = coherence_operator(delta_series, window=coherence_window)
    smoothed_si = coherence_operator(si_series, window=coherence_window)
    clamped_si = [max(0.0, min(1.0, value)) for value in smoothed_si]
    updated_bundles = _update_bundles(bundles, smoothed_delta, clamped_si)
    dissonance = dissonance_operator(smoothed_delta, objectives["delta_nfr"])
    coupling = acoplamiento_operator(smoothed_delta, clamped_si)
    resonance = resonance_operator(clamped_si)
    recursive_trace = recursividad_operator(
        clamped_si, seed=clamped_si[0], decay=recursion_decay
    )

    return {
        "objectives": objectives,
        "bundles": updated_bundles,
        "delta_nfr_series": smoothed_delta,
        "sense_index_series": clamped_si,
        "delta_nfr": mean(smoothed_delta),
        "sense_index": mean(clamped_si),
        "dissonance": dissonance,
        "coupling": coupling,
        "resonance": resonance,
        "recursive_trace": recursive_trace,
    }


__all__ = [
    "emission_operator",
    "recepcion_operator",
    "coherence_operator",
    "dissonance_operator",
    "acoplamiento_operator",
    "resonance_operator",
    "recursividad_operator",
    "orchestrate_delta_metrics",
]

