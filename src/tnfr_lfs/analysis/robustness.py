"""Session robustness metrics helpers."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

try:  # pragma: no cover - exercised in tests when NumPy is missing
    import numpy as np
except Exception:  # pragma: no cover - NumPy is optional
    np = None  # type: ignore[assignment]

from tnfr_core.equations.epi_models import EPIBundle
from tnfr_core.equations.phases import PHASE_SEQUENCE, phase_family
def _lap_groups(
    bundles: Sequence[EPIBundle],
    lap_indices: Sequence[int] | None,
    lap_metadata: Sequence[Mapping[str, Any]] | None,
) -> list[tuple[int, str, list[int]]]:
    if not bundles or not lap_indices:
        return []
    sample_count = min(len(bundles), len(lap_indices))
    lap_samples: dict[Any, list[int]] = {}
    for position in range(sample_count):
        lap_value = lap_indices[position]
        lap_samples.setdefault(lap_value, []).append(position)

    groups: list[tuple[int, str, list[int]]] = []
    processed: set[int] = set()
    if lap_metadata:
        for entry in lap_metadata:
            try:
                lap_index = int(entry.get("index", 0))
            except (TypeError, ValueError):
                continue
            indices = lap_samples.get(lap_index)
            if not indices:
                continue
            label = str(entry.get("label", lap_index))
            processed.add(lap_index)
            groups.append((lap_index, label, indices))

    remaining_laps = sorted(
        lap_value
        for lap_value, indices in lap_samples.items()
        if lap_value not in processed and lap_value is not None and indices
    )
    for lap_value in remaining_laps:
        indices = lap_samples.get(lap_value)
        if not indices:
            continue
        lap_index = int(lap_value)
        groups.append((lap_index, f"Lap {lap_index + 1}", indices))
    return groups


def _phase_sample_map(
    microsectors: Sequence[Any] | None,
) -> dict[str, list[int]]:
    assignments: dict[str, set[int]] = defaultdict(set)
    if not microsectors:
        return {}
    for microsector in microsectors:
        samples = getattr(microsector, "phase_samples", {}) or {}
        if not isinstance(samples, Mapping):
            continue
        for phase, indices in samples.items():
            if not isinstance(indices, Iterable):
                continue
            family = phase_family(str(phase))
            for index in indices:
                try:
                    sample_index = int(index)
                except (TypeError, ValueError):
                    continue
                if sample_index < 0:
                    continue
                assignments[family].add(sample_index)
    ordered: dict[str, list[int]] = {}
    for phase in ("entry", "apex", "exit"):
        if phase in assignments:
            ordered[phase] = sorted(assignments.pop(phase))
    for phase in PHASE_SEQUENCE:
        family = phase_family(phase)
        if family in ordered or family not in assignments:
            continue
        ordered[family] = sorted(assignments.pop(family))
    for phase, indices in sorted(assignments.items()):
        ordered[phase] = sorted(indices)
    return ordered


def _bundle_delta_and_sense_values(
    bundles: Sequence[EPIBundle],
) -> tuple[Sequence[float], Sequence[float]]:
    """Return series with per-bundle ``delta_nfr`` and ``sense_index`` values.

    The results prefer :class:`numpy.ndarray` when NumPy is available so callers can
    leverage vectorised math, while continuing to behave like generic sequences for
    existing scalar consumers.
    """

    delta_list: list[float] = []
    sense_list: list[float] = []

    for bundle in bundles:
        delta_raw = getattr(bundle, "delta_nfr", math.nan)
        sense_raw = getattr(bundle, "sense_index", math.nan)

        try:
            delta_numeric = float(delta_raw)
        except (TypeError, ValueError):
            delta_numeric = math.nan
        if not math.isfinite(delta_numeric):
            delta_numeric = math.nan
        delta_list.append(delta_numeric)

        try:
            sense_numeric = float(sense_raw)
        except (TypeError, ValueError):
            sense_numeric = math.nan
        if not math.isfinite(sense_numeric):
            sense_numeric = math.nan
        sense_list.append(sense_numeric)

    if np is None:
        return delta_list, sense_list

    delta_array = np.asarray(delta_list, dtype=float)
    sense_array = np.asarray(sense_list, dtype=float)
    return delta_array, sense_array


def _coefficient_summary_from_moments(
    count: int,
    total: float,
    total_abs: float,
    total_sq: float,
    *,
    threshold: float | None = None,
) -> dict[str, float | int | bool]:
    """Compute summary statistics from running moments."""

    if count <= 0:
        payload: dict[str, float | int | bool] = {
            "samples": 0,
            "mean": 0.0,
            "mean_abs": 0.0,
            "stdev": 0.0,
            "coefficient_of_variation": 0.0,
        }
        if threshold is not None:
            payload["threshold"] = float(threshold)
            payload["ok"] = True
        return payload

    average = total / count
    mean_abs = total_abs / count
    variance = max(total_sq / count - average * average, 0.0)
    stdev = math.sqrt(variance)
    baseline = max(abs(average), mean_abs, 1e-9)
    coefficient = stdev / baseline if baseline > 0.0 else 0.0

    payload: dict[str, float | int | bool] = {
        "samples": count,
        "mean": average,
        "mean_abs": mean_abs,
        "stdev": stdev,
        "coefficient_of_variation": coefficient,
    }
    if threshold is not None:
        limit = float(threshold)
        payload["threshold"] = limit
        payload["ok"] = coefficient <= limit
    return payload


def _coefficient_summary_for_indices(
    delta_values: Sequence[float],
    sense_values: Sequence[float],
    sample_indices: Sequence[int],
    *,
    delta_threshold: float | None = None,
    sense_threshold: float | None = None,
) -> tuple[dict[str, float | int | bool], dict[str, float | int | bool]]:
    """Summaries for ``delta_nfr`` and ``sense_index`` over the given indices."""

    def _scalar_summary() -> tuple[
        dict[str, float | int | bool],
        dict[str, float | int | bool],
    ]:
        max_index = min(len(delta_values), len(sense_values))

        delta_count = 0
        delta_sum = 0.0
        delta_sum_abs = 0.0
        delta_sum_sq = 0.0

        sense_count = 0
        sense_sum = 0.0
        sense_sum_abs = 0.0
        sense_sum_sq = 0.0

        for index in sample_indices:
            if not 0 <= index < max_index:
                continue

            delta_value = delta_values[index]
            if math.isfinite(delta_value):
                delta_count += 1
                delta_sum += delta_value
                delta_sum_abs += abs(delta_value)
                delta_sum_sq += delta_value * delta_value

            sense_value = sense_values[index]
            if math.isfinite(sense_value):
                sense_count += 1
                sense_sum += sense_value
                sense_sum_abs += abs(sense_value)
                sense_sum_sq += sense_value * sense_value

        delta_summary = _coefficient_summary_from_moments(
            delta_count,
            delta_sum,
            delta_sum_abs,
            delta_sum_sq,
            threshold=delta_threshold,
        )
        sense_summary = _coefficient_summary_from_moments(
            sense_count,
            sense_sum,
            sense_sum_abs,
            sense_sum_sq,
            threshold=sense_threshold,
        )
        return delta_summary, sense_summary

    if np is None:
        return _scalar_summary()

    try:
        delta_array = np.asarray(delta_values, dtype=float)
        sense_array = np.asarray(sense_values, dtype=float)
        index_array = np.asarray(sample_indices, dtype=int)
    except (TypeError, ValueError):
        return _scalar_summary()

    max_index = min(delta_array.size, sense_array.size)
    if max_index <= 0:
        return _coefficient_summary_from_moments(
            0, 0.0, 0.0, 0.0, threshold=delta_threshold
        ), _coefficient_summary_from_moments(
            0, 0.0, 0.0, 0.0, threshold=sense_threshold
        )

    if index_array.size:
        valid_mask = (index_array >= 0) & (index_array < max_index)
        valid_indices = index_array[valid_mask]
    else:
        valid_indices = index_array

    def _moments(values: Sequence[float]) -> tuple[int, float, float, float]:
        if np is None:
            raise RuntimeError("NumPy unexpectedly unavailable during vectorised path")
        if valid_indices.size:
            selected = np.take(values, valid_indices)
        else:
            selected = values[:0]  # type: ignore[index]
        finite_mask = np.isfinite(selected)
        if not np.any(finite_mask):
            return 0, 0.0, 0.0, 0.0
        finite_values = selected[finite_mask]
        count = int(finite_values.size)
        total = float(finite_values.sum(dtype=float))
        total_abs = float(np.abs(finite_values).sum(dtype=float))
        total_sq = float(np.square(finite_values).sum(dtype=float))
        return count, total, total_abs, total_sq

    try:
        (
            delta_count,
            delta_sum,
            delta_sum_abs,
            delta_sum_sq,
        ) = _moments(delta_array)
        (
            sense_count,
            sense_sum,
            sense_sum_abs,
            sense_sum_sq,
        ) = _moments(sense_array)
    except Exception:
        return _scalar_summary()

    delta_summary = _coefficient_summary_from_moments(
        delta_count,
        delta_sum,
        delta_sum_abs,
        delta_sum_sq,
        threshold=delta_threshold,
    )
    sense_summary = _coefficient_summary_from_moments(
        sense_count,
        sense_sum,
        sense_sum_abs,
        sense_sum_sq,
        threshold=sense_threshold,
    )
    return delta_summary, sense_summary


def compute_session_robustness(
    bundles: Sequence[EPIBundle] | None,
    *,
    lap_indices: Sequence[int] | None = None,
    lap_metadata: Sequence[Mapping[str, Any]] | None = None,
    microsectors: Sequence[Any] | None = None,
    thresholds: Mapping[str, Mapping[str, float]] | None = None,
) -> Mapping[str, Any]:
    """Return robustness statistics grouped by lap and driving phase."""

    if not bundles:
        return {}
    bundle_list: Sequence[EPIBundle] = bundles
    lap_thresholds: Mapping[str, float] = {}
    phase_thresholds: Mapping[str, float] | Mapping[str, Mapping[str, float]] = {}
    if isinstance(thresholds, Mapping):
        lap_thresholds = thresholds.get("lap", {}) or {}
        phase_thresholds = thresholds.get("phase", {}) or {}
    robustness: dict[str, Any] = {}

    delta_series, sense_series = _bundle_delta_and_sense_values(bundle_list)

    lap_groups = _lap_groups(bundle_list, lap_indices or [], lap_metadata or [])
    if lap_groups:
        entries: list[dict[str, Any]] = []
        lap_delta_limit = (
            lap_thresholds.get("delta_nfr") if isinstance(lap_thresholds, Mapping) else None
        )
        lap_si_limit = (
            lap_thresholds.get("sense_index") if isinstance(lap_thresholds, Mapping) else None
        )
        for lap_index, label, indices in lap_groups:
            delta_summary, sense_summary = _coefficient_summary_for_indices(
                delta_series,
                sense_series,
                indices,
                delta_threshold=lap_delta_limit,
                sense_threshold=lap_si_limit,
            )
            entries.append(
                {
                    "index": lap_index,
                    "label": label,
                    "samples": len(indices),
                    "delta_nfr": delta_summary,
                    "sense_index": sense_summary,
                }
            )
        robustness["laps"] = entries

    phase_map = _phase_sample_map(microsectors)
    if phase_map:
        phase_payload: dict[str, Any] = {}
        phase_defaults: Mapping[str, float] = {}
        per_phase_thresholds: Mapping[str, Mapping[str, float]] = {}
        if isinstance(phase_thresholds, Mapping):
            if all(isinstance(value, Mapping) for value in phase_thresholds.values()):
                per_phase_thresholds = {
                    str(key): value  # type: ignore[assignment]
                    for key, value in phase_thresholds.items()
                    if isinstance(value, Mapping)
                }
            else:
                phase_defaults = {
                    str(key): float(value)
                    for key, value in phase_thresholds.items()
                    if isinstance(value, (int, float))
                }
        for phase, indices in phase_map.items():
            phase_limits = per_phase_thresholds.get(phase, phase_defaults)
            delta_limit = None
            sense_limit = None
            if isinstance(phase_limits, Mapping):
                delta_limit = phase_limits.get("delta_nfr")
                sense_limit = phase_limits.get("sense_index")
            delta_summary, sense_summary = _coefficient_summary_for_indices(
                delta_series,
                sense_series,
                indices,
                delta_threshold=delta_limit,
                sense_threshold=sense_limit,
            )
            phase_payload[phase] = {
                "samples": len(indices),
                "delta_nfr": delta_summary,
                "sense_index": sense_summary,
            }
        robustness["phases"] = phase_payload

    return robustness


__all__ = ["compute_session_robustness"]
