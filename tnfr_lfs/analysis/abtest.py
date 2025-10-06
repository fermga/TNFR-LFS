"""Lap-level A/B comparison utilities."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean
from typing import Mapping, Sequence


SUPPORTED_LAP_METRICS: tuple[str, ...] = (
    "sense_index",
    "delta_nfr",
    "coherence_index",
    "delta_nfr_proj_longitudinal",
    "delta_nfr_proj_lateral",
)


@dataclass(frozen=True, slots=True)
class ABResult:
    """Summary of an A/B comparison between two telemetry stints."""

    metric: str
    baseline_laps: tuple[float, ...]
    variant_laps: tuple[float, ...]
    baseline_mean: float
    variant_mean: float
    mean_difference: float
    bootstrap_low: float
    bootstrap_high: float
    permutation_p_value: float
    estimated_power: float
    alpha: float


def _percentile(samples: Sequence[float], q: float) -> float:
    if not samples:
        raise ValueError("Cannot compute percentile of an empty sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError("Percentile must be between 0 and 1")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    if lower == upper:
        return lower_value
    weight = position - lower
    return lower_value * (1.0 - weight) + upper_value * weight


def _extract_lap_values(metrics: Mapping[str, object], metric: str) -> tuple[float, ...]:
    stages = metrics.get("stages") if isinstance(metrics, Mapping) else None
    if not isinstance(stages, Mapping):
        return ()
    coherence_stage = stages.get("coherence")
    reception_stage = stages.get("recepcion")
    if not isinstance(coherence_stage, Mapping):
        return ()
    bundles = coherence_stage.get("bundles")
    if not isinstance(bundles, Sequence):
        return ()
    lap_indices: Sequence[int | None]
    if isinstance(reception_stage, Mapping):
        candidate = reception_stage.get("lap_indices")
        if isinstance(candidate, Sequence):
            lap_indices = tuple(
                int(index) if isinstance(index, (int, float)) else None for index in candidate
            )
        else:
            lap_indices = ()
    else:
        lap_indices = ()
    values_by_lap: dict[int, list[float]] = {}
    last_index: int = 0
    for position, bundle in enumerate(bundles):
        lap_index: int
        if position < len(lap_indices) and lap_indices[position] is not None:
            lap_index = int(lap_indices[position])
            last_index = lap_index
        else:
            lap_index = last_index if lap_indices else 0
        try:
            raw_value = getattr(bundle, metric)
        except AttributeError:
            continue
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        values_by_lap.setdefault(lap_index, []).append(numeric)
    if not values_by_lap:
        return ()
    aggregated: list[float] = []
    for lap_index in sorted(values_by_lap):
        lap_samples = values_by_lap[lap_index]
        if lap_samples:
            aggregated.append(mean(lap_samples))
    return tuple(aggregated)


def _bootstrap_differences(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    iterations: int,
    rng: random.Random,
) -> list[float]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    diffs: list[float] = []
    for _ in range(iterations):
        sample_a = [rng.choice(values_a) for _ in values_a]
        sample_b = [rng.choice(values_b) for _ in values_b]
        diffs.append(mean(sample_b) - mean(sample_a))
    return diffs


def _permutation_differences(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    iterations: int,
    rng: random.Random,
) -> list[float]:
    combined = list(values_a) + list(values_b)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not combined:
        return []
    indices = list(range(len(combined)))
    diffs: list[float] = []
    size_a = len(values_a)
    for _ in range(iterations):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        sample_a = [combined[idx] for idx in shuffled[:size_a]]
        sample_b = [combined[idx] for idx in shuffled[size_a:]]
        diffs.append(mean(sample_b) - mean(sample_a))
    return diffs


def ab_compare_by_lap(
    baseline_metrics: Mapping[str, object],
    variant_metrics: Mapping[str, object],
    *,
    metric: str,
    iterations: int = 2000,
    alpha: float = 0.05,
    rng: random.Random | None = None,
) -> ABResult:
    """Compare two telemetry stints by aggregating lap-level metrics."""

    if metric not in SUPPORTED_LAP_METRICS:
        raise ValueError(f"Metric '{metric}' is not supported for A/B comparison")
    values_a = _extract_lap_values(baseline_metrics, metric)
    values_b = _extract_lap_values(variant_metrics, metric)
    if not values_a or not values_b:
        raise ValueError("Both telemetry inputs must contain lap aggregates")
    random_source = rng or random.Random()
    baseline_mean = mean(values_a)
    variant_mean = mean(values_b)
    observed_difference = variant_mean - baseline_mean
    bootstrap_diffs = _bootstrap_differences(
        values_a, values_b, iterations=iterations, rng=random_source
    )
    low_q = alpha / 2.0
    high_q = 1.0 - alpha / 2.0
    bootstrap_low = _percentile(bootstrap_diffs, low_q)
    bootstrap_high = _percentile(bootstrap_diffs, high_q)
    permutation_diffs = _permutation_differences(
        values_a, values_b, iterations=iterations, rng=random_source
    )
    if permutation_diffs:
        extreme = sum(
            1 for diff in permutation_diffs if abs(diff) >= abs(observed_difference)
        )
        permutation_p = (extreme + 1.0) / (len(permutation_diffs) + 1.0)
        critical = _percentile([abs(diff) for diff in permutation_diffs], 1.0 - alpha)
    else:
        permutation_p = 1.0
        critical = 0.0
    if critical <= 0.0:
        power = 1.0
    else:
        power = sum(1 for diff in bootstrap_diffs if abs(diff) >= critical) / float(
            len(bootstrap_diffs)
        )
    return ABResult(
        metric=metric,
        baseline_laps=tuple(values_a),
        variant_laps=tuple(values_b),
        baseline_mean=baseline_mean,
        variant_mean=variant_mean,
        mean_difference=observed_difference,
        bootstrap_low=bootstrap_low,
        bootstrap_high=bootstrap_high,
        permutation_p_value=permutation_p,
        estimated_power=power,
        alpha=alpha,
    )


__all__ = ["ABResult", "SUPPORTED_LAP_METRICS", "ab_compare_by_lap"]
