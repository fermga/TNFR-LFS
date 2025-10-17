"""Microsector variability helpers for the ΔNFR×Si pipeline."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from tnfr_core.metrics.metrics import compute_window_metrics as _compute_window_metrics
from tnfr_core.runtime.shared import SupportsEPIBundle, SupportsMicrosector

VariancePayload = Callable[[Sequence[float]], Mapping[str, float]]
DeltaIntegral = Callable[..., Sequence[float]]
ArrayModule = Any


def compute_window_metrics(*args: Any, **kwargs: Any) -> Any:
    """Proxy to :func:`tnfr_core.metrics.metrics.compute_window_metrics`."""

    return _compute_window_metrics(*args, **kwargs)


def _microsector_sample_indices(microsector: SupportsMicrosector) -> List[int]:
    indices: set[int] = set()
    for samples in getattr(microsector, "phase_samples", {}).values():
        if samples is None:
            continue
        for idx in samples:
            indices.add(int(idx))
    if not indices:
        spans = [tuple(bounds) for bounds in getattr(microsector, "phase_boundaries", {}).values()]
        if spans:
            start = min(span[0] for span in spans)
            end = max(span[1] for span in spans)
            indices.update(range(start, end))
    return sorted(indices)


def _microsector_cphi_values(microsector: SupportsMicrosector) -> List[float]:
    measures = getattr(microsector, "filtered_measures", {}) or {}
    values: List[float] = []
    if isinstance(measures, Mapping):
        for measure in measures.values():
            if isinstance(measure, Mapping):
                cphi_value = measure.get("coherence_phi")
                if isinstance(cphi_value, (int, float)) and math.isfinite(cphi_value):
                    values.append(float(cphi_value))
    return values


def _microsector_phase_synchrony_values(microsector: SupportsMicrosector) -> List[float]:
    synchrony = getattr(microsector, "phase_synchrony", {}) or {}
    values: List[float] = []
    if isinstance(synchrony, Mapping):
        for value in synchrony.values():
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.append(float(value))
    return values


def _microsector_variability(
    microsectors: Sequence[SupportsMicrosector] | None,
    bundles: Sequence[SupportsEPIBundle],
    lap_indices: Sequence[int],
    lap_metadata: Sequence[Mapping[str, object]],
    *,
    xp: ArrayModule,
    has_jax: bool,
    delta_integral: DeltaIntegral,
    variance_payload: VariancePayload,
) -> List[Dict[str, object]]:
    if not microsectors:
        return []
    bundle_count = len(bundles)
    include_laps = len(lap_metadata) > 1
    delta_series = xp.asarray([float(bundle.delta_nfr) for bundle in bundles], dtype=float)
    sense_index_series = xp.asarray([float(bundle.sense_index) for bundle in bundles], dtype=float)
    timestamp_series = xp.asarray([float(bundle.timestamp) for bundle in bundles], dtype=float)
    lap_indices_array = xp.full((bundle_count,), -1, dtype=int)
    if lap_indices:
        limit = min(len(lap_indices), bundle_count)
        if limit > 0:
            trimmed = xp.asarray([int(lap_indices[idx]) for idx in range(limit)], dtype=int)
            target_indices = xp.arange(limit, dtype=int)
            if has_jax:
                lap_indices_array = lap_indices_array.at[target_indices].set(trimmed)  # type: ignore[attr-defined]
            else:
                lap_indices_array[:limit] = trimmed
    variability: List[Dict[str, object]] = []
    for microsector in microsectors:
        sample_indices = [
            idx for idx in _microsector_sample_indices(microsector) if 0 <= idx < bundle_count
        ]
        sample_index_array = xp.asarray(sample_indices, dtype=int)
        delta_slice = xp.take(delta_series, sample_index_array)
        sense_slice = xp.take(sense_index_series, sample_index_array)
        integral_values = delta_integral(
            bundles,
            sample_indices,
            delta_series=delta_series,
            timestamp_series=timestamp_series,
        )
        cphi_values = _microsector_cphi_values(microsector)
        synchrony_values = _microsector_phase_synchrony_values(microsector)
        cphi_stats = variance_payload(cphi_values)
        synchrony_stats = variance_payload(synchrony_values)
        entry: Dict[str, object] = {
            "microsector": microsector.index,
            "label": f"Curva {microsector.index + 1}",
            "overall": {
                "samples": len(sample_indices),
                "delta_nfr": variance_payload(delta_slice.tolist()),
                "sense_index": variance_payload(sense_slice.tolist()),
                "delta_nfr_integral": variance_payload(list(integral_values)),
                "cphi": cphi_stats,
                "phase_synchrony": synchrony_stats,
            },
        }
        if include_laps and lap_indices and sample_indices:
            lap_payload: Dict[str, Dict[str, object]] = {}
            microsector_mask = xp.zeros(bundle_count, dtype=bool)
            if has_jax:
                microsector_mask = microsector_mask.at[sample_index_array].set(True)  # type: ignore[attr-defined]
            else:
                microsector_mask[sample_indices] = True
            for lap_entry in lap_metadata:
                lap_index = int(lap_entry.get("index", 0))
                lap_label = str(lap_entry.get("label", lap_index))
                lap_mask = xp.equal(lap_indices_array, lap_index)
                combined_mask = xp.logical_and(microsector_mask, lap_mask)
                combined_indices_tuple = xp.nonzero(combined_mask)
                combined_indices = (
                    combined_indices_tuple[0]
                    if isinstance(combined_indices_tuple, tuple)
                    else combined_indices_tuple
                )
                if combined_indices.size == 0:
                    continue
                lap_specific_indices = [int(idx) for idx in combined_indices.tolist()]
                lap_delta_slice = xp.take(delta_series, combined_indices)
                lap_sense_slice = xp.take(sense_index_series, combined_indices)
                lap_payload[lap_label] = {
                    "samples": len(lap_specific_indices),
                    "delta_nfr": variance_payload(lap_delta_slice.tolist()),
                    "sense_index": variance_payload(lap_sense_slice.tolist()),
                    "delta_nfr_integral": variance_payload(
                        list(
                            delta_integral(
                                bundles,
                                lap_specific_indices,
                                delta_series=delta_series,
                                timestamp_series=timestamp_series,
                            )
                        )
                    ),
                    "phase_synchrony": synchrony_stats,
                }
            if lap_payload:
                entry["laps"] = lap_payload
        variability.append(entry)
    return variability


def _phase_context_from_microsectors(
    microsectors: Sequence[SupportsMicrosector] | None,
) -> tuple[Dict[int, str], Dict[int, Mapping[str, Mapping[str, float] | float]]]:
    assignments: Dict[int, str] = {}
    weight_lookup: Dict[int, Mapping[str, Mapping[str, float] | float]] = {}
    if not microsectors:
        return assignments, weight_lookup
    for microsector in microsectors:
        raw_weights = getattr(microsector, "phase_weights", {}) or {}
        weight_profile: Dict[str, Mapping[str, float] | float] = {}
        for phase, profile in raw_weights.items():
            if isinstance(profile, Mapping):
                weight_profile[str(phase)] = dict(profile)
            else:
                weight_profile[str(phase)] = float(profile)
        for phase, samples in getattr(microsector, "phase_samples", {}).items():
            if not samples:
                continue
            for sample in samples:
                index = int(sample)
                if index < 0:
                    continue
                assignments[index] = str(phase)
                weight_lookup[index] = weight_profile
    return assignments, weight_lookup
