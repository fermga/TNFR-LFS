"""Session robustness metrics helpers."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence

from tnfr_lfs.core.epi_models import EPIBundle
from tnfr_lfs.core.phases import PHASE_SEQUENCE, phase_family


def _numeric_series(values: Iterable[Any]) -> list[float]:
    """Return a list of finite floats extracted from ``values``."""

    numeric: list[float] = []
    for value in values:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_value):
            numeric.append(numeric_value)
    return numeric


def _robust_coefficient_summary(
    values: Iterable[Any], *, threshold: float | None = None
) -> dict[str, float | int | bool]:
    """Compute dispersion metrics with a coefficient of variation safeguard."""

    series = _numeric_series(values)
    samples = len(series)
    if not series:
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

    average = fmean(series)
    mean_abs = fmean(abs(value) for value in series)
    stdev = pstdev(series) if samples > 1 else 0.0
    baseline = max(abs(average), mean_abs, 1e-9)
    coefficient = stdev / baseline if baseline > 0.0 else 0.0

    payload = {
        "samples": samples,
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


def _lap_groups(
    bundles: Sequence[EPIBundle],
    lap_indices: Sequence[int] | None,
    lap_metadata: Sequence[Mapping[str, Any]] | None,
) -> list[tuple[int, str, list[int]]]:
    if not bundles or not lap_indices:
        return []
    sample_count = min(len(bundles), len(lap_indices))
    groups: list[tuple[int, str, list[int]]] = []
    processed: set[int] = set()
    if lap_metadata:
        for entry in lap_metadata:
            try:
                lap_index = int(entry.get("index", 0))
            except (TypeError, ValueError):
                continue
            label = str(entry.get("label", lap_index))
            indices = [
                position
                for position in range(sample_count)
                if lap_indices[position] == lap_index
            ]
            if not indices:
                continue
            processed.add(lap_index)
            groups.append((lap_index, label, indices))
    remaining = [
        index
        for index in range(sample_count)
        if lap_indices[index] not in processed
    ]
    if remaining:
        unique_laps = sorted(
            {
                lap_indices[position]
                for position in remaining
                if lap_indices[position] is not None
            }
        )
        for lap_index in unique_laps:
            indices = [
                position
                for position in remaining
                if lap_indices[position] == lap_index
            ]
            if not indices:
                continue
            groups.append((int(lap_index), f"Lap {int(lap_index) + 1}", indices))
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


def _bundle_metric_series(
    bundles: Sequence[EPIBundle], sample_indices: Sequence[int], attribute: str
) -> list[float]:
    values: list[float] = []
    bundle_count = len(bundles)
    for index in sample_indices:
        if not 0 <= index < bundle_count:
            continue
        value = getattr(bundles[index], attribute, None)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    return values


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
            delta_values = _bundle_metric_series(bundle_list, indices, "delta_nfr")
            si_values = _bundle_metric_series(bundle_list, indices, "sense_index")
            entries.append(
                {
                    "index": lap_index,
                    "label": label,
                    "samples": len(indices),
                    "delta_nfr": _robust_coefficient_summary(
                        delta_values, threshold=lap_delta_limit
                    ),
                    "sense_index": _robust_coefficient_summary(
                        si_values, threshold=lap_si_limit
                    ),
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
            delta_values = _bundle_metric_series(bundle_list, indices, "delta_nfr")
            si_values = _bundle_metric_series(bundle_list, indices, "sense_index")
            phase_limits = per_phase_thresholds.get(phase, phase_defaults)
            delta_limit = None
            sense_limit = None
            if isinstance(phase_limits, Mapping):
                delta_limit = phase_limits.get("delta_nfr")
                sense_limit = phase_limits.get("sense_index")
            phase_payload[phase] = {
                "samples": len(indices),
                "delta_nfr": _robust_coefficient_summary(
                    delta_values, threshold=delta_limit
                ),
                "sense_index": _robust_coefficient_summary(
                    si_values, threshold=sense_limit
                ),
            }
        robustness["phases"] = phase_payload

    return robustness


__all__ = ["compute_session_robustness"]
