"""Windowed telemetry metrics used by the HUD and setup planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable, Mapping, Sequence

from .contextual_delta import (
    ContextMatrix,
    apply_contextual_delta,
    load_context_matrix,
    resolve_context_from_bundle,
    resolve_context_from_record,
)
from .dissonance import compute_useful_dissonance_stats
from .epi import TelemetryRecord
from .epi_models import EPIBundle
from .spectrum import phase_alignment
from .resonance import estimate_excitation_frequency
from .structural_time import resolve_time_axis

__all__ = ["AeroCoherence", "WindowMetrics", "compute_window_metrics", "compute_aero_coherence"]


_FRONT_FEATURE_KEYS = {
    "mu_eff_front",
    "mu_eff_front_lateral",
    "mu_eff_front_longitudinal",
}
_REAR_FEATURE_KEYS = {
    "mu_eff_rear",
    "mu_eff_rear_lateral",
    "mu_eff_rear_longitudinal",
}


@dataclass(frozen=True)
class AeroCoherence:
    """Summarises aero balance deltas split by speed bins."""

    low_speed_front: float = 0.0
    low_speed_rear: float = 0.0
    low_speed_imbalance: float = 0.0
    low_speed_samples: int = 0
    high_speed_front: float = 0.0
    high_speed_rear: float = 0.0
    high_speed_imbalance: float = 0.0
    high_speed_samples: int = 0
    guidance: str = ""

    def dominant_axis(self, tolerance: float = 0.05) -> str | None:
        """Return the dominant axle when the imbalance exceeds ``tolerance``."""

        if abs(self.high_speed_imbalance) <= tolerance:
            return None
        return "front" if self.high_speed_imbalance < 0.0 else "rear"


@dataclass(frozen=True)
class WindowMetrics:
    """Aggregated metrics derived from a telemetry window."""

    si: float
    d_nfr_couple: float
    d_nfr_res: float
    d_nfr_flat: float
    nu_f: float
    nu_exc: float
    rho: float
    phase_lag: float
    phase_alignment: float
    useful_dissonance_ratio: float
    useful_dissonance_percentage: float
    aero_coherence: AeroCoherence = field(default_factory=AeroCoherence)


def compute_window_metrics(
    records: Sequence[TelemetryRecord],
    *,
    phase_indices: Sequence[int] | None = None,
    bundles: Sequence[EPIBundle] | None = None,
    fallback_to_chronological: bool = True,
) -> WindowMetrics:
    """Return averaged plan metrics for a telemetry window.

    Parameters
    ----------
    records:
        Ordered window of :class:`TelemetryRecord` samples.
    bundles:
        Optional precomputed :class:`~tnfr_lfs.core.epi_models.EPIBundle` series
        matching ``records``. When provided the ΔNFR derivative used for the
        Useful Dissonance Ratio (UDR) is computed from the smoothed bundle
        values.
    fallback_to_chronological:
        When ``True`` the metric computation gracefully falls back to the
        chronological timestamps if the structural axis is missing or
        non-monotonic.  Disabling the fallback raises a :class:`ValueError`
        whenever the structural axis cannot be resolved.
    """

    if not records:
        return WindowMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, AeroCoherence())

    si_value = mean(record.si for record in records)

    context_matrix = load_context_matrix()

    if phase_indices:
        selected = [
            records[index]
            for index in phase_indices
            if 0 <= index < len(records)
        ]
    else:
        selected = list(records)
    if len(selected) < 4:
        selected = list(records)
    freq, lag, alignment = phase_alignment(selected)
    nu_exc = estimate_excitation_frequency(records)
    rho = nu_exc / freq if freq > 1e-9 else 0.0

    couple, resonance, flatten = _segment_gradients(
        records, segments=3, fallback_to_chronological=fallback_to_chronological
    )

    if bundles:
        timestamps = resolve_time_axis(
            bundles, fallback_to_chronological=fallback_to_chronological
        )
        if timestamps is None:
            raise ValueError("Structural timeline unavailable and fallback disabled")
        bundle_context = [
            resolve_context_from_bundle(context_matrix, bundle) for bundle in bundles
        ]
        delta_series = [
            apply_contextual_delta(
                bundle.delta_nfr,
                factors,
                context_matrix=context_matrix,
            )
            for bundle, factors in zip(bundles, bundle_context)
        ]
        yaw_rates = [bundle.chassis.yaw_rate for bundle in bundles]
    else:
        timestamps = resolve_time_axis(
            records, fallback_to_chronological=fallback_to_chronological
        )
        if timestamps is None:
            raise ValueError("Structural timeline unavailable and fallback disabled")
        record_context = [
            resolve_context_from_record(context_matrix, record) for record in records
        ]
        delta_series = [
            apply_contextual_delta(
                getattr(record, "delta_nfr", record.nfr),
                factors,
                context_matrix=context_matrix,
            )
            for record, factors in zip(records, record_context)
        ]
        yaw_rates = [record.yaw_rate for record in records]
    _useful_samples, _high_yaw_samples, udr = compute_useful_dissonance_stats(
        timestamps,
        delta_series,
        yaw_rates,
    )

    aero = compute_aero_coherence(records, bundles)

    return WindowMetrics(
        si=si_value,
        d_nfr_couple=couple,
        d_nfr_res=resonance,
        d_nfr_flat=flatten,
        nu_f=freq,
        nu_exc=nu_exc,
        rho=rho,
        phase_lag=lag,
        phase_alignment=alignment,
        useful_dissonance_ratio=udr,
        useful_dissonance_percentage=udr * 100.0,
        aero_coherence=aero,
    )


def compute_aero_coherence(
    records: Sequence[TelemetryRecord],
    bundles: Sequence[EPIBundle] | None = None,
    *,
    low_speed_threshold: float = 35.0,
    high_speed_threshold: float = 50.0,
    imbalance_tolerance: float = 0.08,
) -> AeroCoherence:
    """Compute aero balance deltas at low and high speed.

    The helper inspects ΔNFR contributions attributed to μ_eff front/rear terms
    in the :class:`~tnfr_lfs.core.epi_models.EPIBundle.delta_breakdown` payload.
    When the optional ``bundles`` sequence is not provided or lacks breakdown
    data the function gracefully returns a neutral :class:`AeroCoherence`
    instance with zero samples.
    """

    if low_speed_threshold < 0.0 or high_speed_threshold < 0.0:
        raise ValueError("speed thresholds must be non-negative")
    if low_speed_threshold >= high_speed_threshold:
        raise ValueError("low speed threshold must be below high speed threshold")

    if not bundles:
        return AeroCoherence()

    def _aero_components(breakdown: Mapping[str, Mapping[str, float]] | None) -> tuple[float, float]:
        if not breakdown:
            return 0.0, 0.0
        front = 0.0
        rear = 0.0
        for features in breakdown.values():
            if not isinstance(features, Mapping):
                continue
            for key, value in features.items():
                if key in _FRONT_FEATURE_KEYS:
                    front += float(value)
                elif key in _REAR_FEATURE_KEYS:
                    rear += float(value)
        return front, rear

    def _resolve_speed(index: int) -> float | None:
        if 0 <= index < len(records):
            candidate = getattr(records[index], "speed", None)
            if isinstance(candidate, (int, float)):
                return float(candidate)
        bundle = bundles[index]
        transmission = getattr(bundle, "transmission", None)
        if transmission is not None:
            speed_value = getattr(transmission, "speed", None)
            if isinstance(speed_value, (int, float)):
                return float(speed_value)
        return None

    low_front = 0.0
    low_rear = 0.0
    low_samples = 0
    high_front = 0.0
    high_rear = 0.0
    high_samples = 0

    for index, bundle in enumerate(bundles):
        speed = _resolve_speed(index)
        if speed is None:
            continue
        front, rear = _aero_components(getattr(bundle, "delta_breakdown", {}))
        if front == 0.0 and rear == 0.0:
            continue
        if speed <= low_speed_threshold:
            low_front += front
            low_rear += rear
            low_samples += 1
        elif speed >= high_speed_threshold:
            high_front += front
            high_rear += rear
            high_samples += 1

    if low_samples:
        avg_low_front = low_front / low_samples
        avg_low_rear = low_rear / low_samples
    else:
        avg_low_front = 0.0
        avg_low_rear = 0.0
    if high_samples:
        avg_high_front = high_front / high_samples
        avg_high_rear = high_rear / high_samples
    else:
        avg_high_front = 0.0
        avg_high_rear = 0.0

    low_imbalance = avg_low_front - avg_low_rear
    high_imbalance = avg_high_front - avg_high_rear

    guidance: str
    if high_samples == 0:
        guidance = "Sin datos aero"
    elif abs(high_imbalance) <= imbalance_tolerance:
        guidance = "Aero alta velocidad equilibrado"
    elif high_imbalance > 0.0:
        guidance = "Alta velocidad → sube alerón trasero"
    else:
        guidance = "Alta velocidad → libera alerón trasero/refuerza delantero"

    if low_samples and abs(low_imbalance) > imbalance_tolerance:
        direction = "trasero" if low_imbalance > 0.0 else "delantero"
        guidance += f" · baja velocidad sesgo {direction}"

    return AeroCoherence(
        low_speed_front=avg_low_front,
        low_speed_rear=avg_low_rear,
        low_speed_imbalance=low_imbalance,
        low_speed_samples=low_samples,
        high_speed_front=avg_high_front,
        high_speed_rear=avg_high_rear,
        high_speed_imbalance=high_imbalance,
        high_speed_samples=high_samples,
        guidance=guidance,
    )


def _segment_gradients(
    records: Sequence[TelemetryRecord], *, segments: int, fallback_to_chronological: bool = True
) -> tuple[float, ...]:
    parts = _split_records(records, segments)
    context_matrix = load_context_matrix()
    return tuple(
        _gradient(
            part,
            fallback_to_chronological=fallback_to_chronological,
            context_matrix=context_matrix,
        )
        for part in parts
    )


def _split_records(
    records: Sequence[TelemetryRecord], segments: int
) -> list[Sequence[TelemetryRecord]]:
    length = len(records)
    if length == 0:
        return [tuple()] * segments

    base, remainder = divmod(length, segments)
    slices: list[Sequence[TelemetryRecord]] = []
    index = 0
    for segment_index in range(segments):
        size = base + (1 if segment_index < remainder else 0)
        if size <= 0:
            slices.append(records[index:index])
            continue
        next_index = index + size
        slices.append(records[index:next_index])
        index = next_index
    return slices


def _gradient(
    records: Iterable[TelemetryRecord],
    *,
    fallback_to_chronological: bool = True,
    context_matrix: ContextMatrix,
) -> float:
    iterator = list(records)
    if len(iterator) < 2:
        return 0.0
    axis = resolve_time_axis(
        iterator, fallback_to_chronological=fallback_to_chronological
    )
    if not axis or len(axis) < 2:
        return 0.0
    start = iterator[0]
    end = iterator[-1]
    dt = axis[-1] - axis[0]
    if dt <= 0.0:
        return 0.0
    start_delta = getattr(start, "delta_nfr", start.nfr)
    end_delta = getattr(end, "delta_nfr", end.nfr)
    start_factors = resolve_context_from_record(context_matrix, start)
    end_factors = resolve_context_from_record(context_matrix, end)
    start_value = apply_contextual_delta(
        start_delta, start_factors, context_matrix=context_matrix
    )
    end_value = apply_contextual_delta(
        end_delta, end_factors, context_matrix=context_matrix
    )
    return (end_value - start_value) / dt

