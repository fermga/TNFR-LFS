"""Mutation operators and historical filters."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence
import warnings

import numpy as np

from tnfr_core.equations.archetypes import ARCHETYPE_MEDIUM
from tnfr_core.equations.constants import WHEEL_SUFFIXES

from ._shared import _HAS_JAX, jnp


def mutation_operator(
    state: MutableMapping[str, Dict[str, object]],
    triggers: Mapping[str, object],
    *,
    entropy_threshold: float = 0.65,
    entropy_increase: float = 0.08,
    style_threshold: float = 0.12,
) -> Dict[str, object]:
    """Update the target archetype when entropy or style shifts are detected."""

    microsector_id = triggers.get("microsector_id")
    if not isinstance(microsector_id, str) or not microsector_id:
        raise ValueError("microsector_id trigger must be a non-empty string")

    current_archetype = str(triggers.get("current_archetype", ARCHETYPE_MEDIUM))
    candidate_archetype = str(triggers.get("candidate_archetype", current_archetype))
    fallback_archetype = str(triggers.get("fallback_archetype", ARCHETYPE_MEDIUM))

    entropy = float(triggers.get("entropy", 0.0))
    style_index = float(triggers.get("style_index", 1.0))
    style_reference = float(triggers.get("style_reference", style_index))
    phase = triggers.get("phase") if isinstance(triggers.get("phase"), str) else None
    dynamic_flag = bool(triggers.get("dynamic_conditions", False))

    micro_state = state.setdefault(
        microsector_id,
        {
            "archetype": current_archetype,
            "entropy": entropy,
            "style_index": style_index,
            "phase": phase,
        },
    )

    previous_entropy = float(micro_state.get("entropy", entropy))
    previous_style = float(micro_state.get("style_index", style_index))
    previous_phase = micro_state.get("phase") if isinstance(micro_state.get("phase"), str) else None

    entropy_delta = entropy - previous_entropy
    style_delta = abs(style_index - style_reference)

    mutated = False
    selected_archetype = current_archetype
    if entropy >= entropy_threshold and entropy_delta >= entropy_increase:
        selected_archetype = fallback_archetype
        mutated = selected_archetype != current_archetype
    elif style_delta >= style_threshold or (dynamic_flag and style_delta >= style_threshold * 0.5):
        selected_archetype = candidate_archetype
        mutated = selected_archetype != current_archetype
    elif phase is not None and previous_phase is not None and phase != previous_phase:
        secondary_delta = abs(style_index - previous_style)
        if secondary_delta >= style_threshold * 0.5:
            selected_archetype = candidate_archetype
            mutated = selected_archetype != current_archetype

    micro_state.update(
        {
            "archetype": selected_archetype,
            "entropy": entropy,
            "style_index": style_index,
            "phase": phase if phase is not None else previous_phase,
        }
    )

    return {
        "microsector_id": microsector_id,
        "archetype": selected_archetype,
        "mutated": mutated,
        "entropy": entropy,
        "entropy_delta": entropy_delta,
        "style_delta": style_delta,
        "phase": micro_state["phase"],
    }


def recursive_filter_operator(
    series: Sequence[float], *, seed: float = 0.0, decay: float = 0.5
) -> List[float]:
    """Apply a recursive filter to a series to capture hysteresis effects."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in the [0, 1) interval")
    if not series:
        return []
    xp = jnp if _HAS_JAX else np
    array = xp.asarray(series, dtype=float)
    length = array.shape[0]
    dtype = array.dtype
    kernel = xp.power(decay, xp.arange(length, dtype=dtype)) * (1.0 - decay)
    filtered = xp.convolve(array, kernel, mode="full")[:length]
    seed_powers = xp.power(decay, xp.arange(1, length + 1, dtype=dtype)) * seed
    trace = filtered + seed_powers
    return np.asarray(trace, dtype=float).tolist()


def recursividad_operator(
    series: Sequence[float], *, seed: float = 0.0, decay: float = 0.5
) -> List[float]:
    """Compatibility wrapper for :func:`recursive_filter_operator`."""

    warnings.warn(
        "recursividad_operator has been renamed to recursive_filter_operator; "
        "please update imports before the legacy name is removed.",
        DeprecationWarning,
        stacklevel=2,
    )
    return recursive_filter_operator(series, seed=seed, decay=decay)


_STABILITY_COV_THRESHOLD = 0.15


def _variance_payload(values: Sequence[float]) -> Dict[str, float]:
    """Return statistical descriptors for ``values``."""

    if not values:
        return {
            "mean": 0.0,
            "variance": 0.0,
            "stdev": 0.0,
            "coefficient_of_variation": 0.0,
            "stability_score": 1.0,
        }
    xp = jnp if _HAS_JAX else np
    array = xp.asarray(values, dtype=float)
    average = float(xp.mean(array))
    variance = float(xp.var(array, ddof=0))
    if variance < 0.0 and abs(variance) < 1e-12:
        variance = 0.0
    stdev = float(xp.sqrt(variance)) if variance > 0.0 else 0.0
    baseline = max(abs(average), 1e-9)
    coefficient = stdev / baseline
    stability = 1.0 - min(1.0, coefficient / _STABILITY_COV_THRESHOLD)
    if stability < 0.0:
        stability = 0.0
    return {
        "mean": average,
        "variance": variance,
        "stdev": stdev,
        "coefficient_of_variation": coefficient,
        "stability_score": stability,
    }


def _delta_integral_series(
    bundles: Sequence[Any],
    sample_indices: Sequence[int],
    *,
    delta_series: Any | None = None,
    timestamp_series: Any | None = None,
) -> List[float]:
    """Return ΔNFR integrals aligned with ``sample_indices``."""

    if not bundles or not sample_indices:
        return []

    xp = jnp if _HAS_JAX else np
    if delta_series is None or timestamp_series is None:
        timestamps = xp.asarray(
            [float(bundles[idx].timestamp) for idx in sample_indices], dtype=float
        )
        delta_nfr = xp.asarray(
            [float(bundles[idx].delta_nfr) for idx in sample_indices], dtype=float
        )
    else:
        indices = xp.asarray(sample_indices, dtype=int)
        timestamps = xp.take(timestamp_series, indices)
        delta_nfr = xp.take(delta_series, indices)

    forward_dt = xp.diff(timestamps)
    forward_dt = xp.concatenate(
        (forward_dt, xp.zeros((1,), dtype=timestamps.dtype)),
        axis=0,
    )
    backward_dt = xp.diff(timestamps)
    backward_dt = xp.concatenate(
        (xp.zeros((1,), dtype=timestamps.dtype), backward_dt),
        axis=0,
    )

    forward_dt = xp.maximum(forward_dt, 0.0)
    backward_dt = xp.maximum(backward_dt, 0.0)

    positions = xp.arange(len(sample_indices))
    use_forward = positions < len(sample_indices) - 1
    dt = xp.where(use_forward, forward_dt, backward_dt)
    dt = xp.where(dt <= 0.0, 1.0, dt)

    integrals = xp.abs(delta_nfr) * dt
    return np.asarray(integrals, dtype=float).tolist()


@dataclass(frozen=True)
class TyreBalanceControlOutput:
    """Aggregated ΔP/Δcamber recommendations for a stint."""

    pressure_delta_front: float
    pressure_delta_rear: float
    camber_delta_front: float
    camber_delta_rear: float
    per_wheel_pressure: Mapping[str, float] = field(default_factory=dict)


def tyre_balance_controller(
    filtered_metrics: Mapping[str, float],
    *,
    delta_nfr_flat: float | None = None,
    target_front: float = 0.82,
    target_rear: float = 0.80,
    pressure_gain: float = 0.25,
    nfr_gain: float = 0.2,
    pressure_max_step: float = 0.16,
    camber_gain: float = 0.18,
    camber_max_step: float = 0.25,
    bias_gain: float = 0.04,
    offsets: Mapping[str, float] | None = None,
) -> TyreBalanceControlOutput:
    """Compute ΔP and camber tweaks from CPHI-derived tyre metrics."""

    def _safe_value(key: str) -> float | None:
        value = filtered_metrics.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return numeric

    xp = jnp if jnp is not None else np

    def _value_or_nan(key: str) -> float:
        value = _safe_value(key)
        return value if value is not None else float("nan")

    def _to_bool(value: object) -> bool:
        if hasattr(value, "item"):
            return bool(value.item())  # type: ignore[attr-defined]
        return bool(value)

    def _to_float(value: object) -> float:
        if isinstance(value, (float, int)):
            return float(value)
        if hasattr(value, "item"):
            return float(value.item())  # type: ignore[attr-defined]
        return float(value)  # pragma: no cover - defensive fallback

    cphi_values = xp.asarray(
        [_value_or_nan(f"cphi_{suffix}") for suffix in WHEEL_SUFFIXES], dtype=xp.float64
    )

    all_missing = _to_bool(xp.all(xp.isnan(cphi_values)))
    if all_missing:
        zero_map = {suffix: 0.0 for suffix in WHEEL_SUFFIXES}
        return TyreBalanceControlOutput(
            pressure_delta_front=0.0,
            pressure_delta_rear=0.0,
            camber_delta_front=0.0,
            camber_delta_rear=0.0,
            per_wheel_pressure=zero_map,
        )

    delta_flat = (
        float(delta_nfr_flat)
        if delta_nfr_flat is not None
        else float(filtered_metrics.get("d_nfr_flat", 0.0))
    )

    def _min_finite(values: Sequence[float] | object) -> float | None:
        array = xp.asarray(values, dtype=xp.float64)
        mask = xp.isfinite(array)
        if not _to_bool(xp.any(mask)):
            return None
        minimum = xp.nanmin(array)
        return _to_float(minimum)

    front_health = _min_finite(cphi_values[:2])
    rear_health = _min_finite(cphi_values[2:])

    pressure_front = nfr_gain * delta_flat
    pressure_rear = nfr_gain * delta_flat
    if front_health is not None:
        pressure_front += pressure_gain * (target_front - front_health)
    if rear_health is not None:
        pressure_rear += pressure_gain * (target_rear - rear_health)

    if offsets:
        pressure_front += float(offsets.get("pressure_front", 0.0))
        pressure_rear += float(offsets.get("pressure_rear", 0.0))

    pressure_front = _to_float(
        xp.clip(
            xp.asarray(pressure_front, dtype=xp.float64),
            -pressure_max_step,
            pressure_max_step,
        )
    )
    pressure_rear = _to_float(
        xp.clip(
            xp.asarray(pressure_rear, dtype=xp.float64),
            -pressure_max_step,
            pressure_max_step,
        )
    )

    def _component_average(suffixes: Sequence[str], key: str) -> float:
        values = xp.asarray(
            [_value_or_nan(f"cphi_{suffix}_{key}") for suffix in suffixes], dtype=xp.float64
        )
        mask = xp.isfinite(values)
        counts = xp.sum(mask)
        sums = xp.sum(xp.where(mask, values, 0.0))
        counts_float = counts.astype(values.dtype)
        safe_counts = xp.where(counts > 0, counts_float, xp.ones_like(counts_float))
        mean_value = xp.where(counts > 0, sums / safe_counts, 0.0)
        return _to_float(mean_value)

    front_gradient_component = _component_average(["fl", "fr"], "gradient")
    rear_gradient_component = _component_average(["rl", "rr"], "gradient")

    camber_front = _to_float(
        xp.clip(
            xp.asarray(-front_gradient_component * camber_gain, dtype=xp.float64),
            -camber_max_step,
            camber_max_step,
        )
    )
    camber_rear = _to_float(
        xp.clip(
            xp.asarray(-rear_gradient_component * camber_gain, dtype=xp.float64),
            -camber_max_step,
            camber_max_step,
        )
    )

    if offsets:
        camber_front += float(offsets.get("camber_front", 0.0))
        camber_rear += float(offsets.get("camber_rear", 0.0))

    bias_values = xp.asarray(
        [_value_or_nan(f"cphi_{suffix}_temp_delta") for suffix in WHEEL_SUFFIXES], dtype=xp.float64
    )
    bias_safe = xp.where(xp.isfinite(bias_values), bias_values, 0.0)

    base_pressures = xp.asarray(
        [pressure_front, pressure_front, pressure_rear, pressure_rear], dtype=xp.float64
    )
    per_wheel_array = xp.clip(
        base_pressures + bias_gain * bias_safe,
        -pressure_max_step,
        pressure_max_step,
    )
    per_wheel_numpy = np.asarray(per_wheel_array, dtype=float)
    per_wheel = {suffix: float(value) for suffix, value in zip(WHEEL_SUFFIXES, per_wheel_numpy)}

    return TyreBalanceControlOutput(
        pressure_delta_front=pressure_front,
        pressure_delta_rear=pressure_rear,
        camber_delta_front=camber_front,
        camber_delta_rear=camber_rear,
        per_wheel_pressure=per_wheel,
    )


__all__ = [
    "TyreBalanceControlOutput",
    "_STABILITY_COV_THRESHOLD",
    "_delta_integral_series",
    "_variance_payload",
    "mutation_operator",
    "recursive_filter_operator",
    "recursividad_operator",
    "tyre_balance_controller",
]

