"""Windowed telemetry metrics used by the HUD and setup planner."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, pvariance
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

__all__ = [
    "AeroCoherence",
    "BrakeHeadroom",
    "WindowMetrics",
    "compute_window_metrics",
    "compute_aero_coherence",
    "resolve_aero_mechanical_coherence",
]


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


_BRAKE_DECEL_REFERENCE = 10.0
_PARTIAL_LOCK_LOWER = -0.35
_PARTIAL_LOCK_UPPER = -0.06
_SEVERE_LOCK_THRESHOLD = -0.45
_SUSTAINED_LOCK_ACTIVATION = 0.75
_WHEEL_SUFFIXES = ("fl", "fr", "rl", "rr")


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
class BrakeHeadroom:
    """Aggregated braking capacity metrics for the current window."""

    value: float = 0.0
    peak_decel: float = 0.0
    abs_activation_ratio: float = 0.0
    partial_locking_ratio: float = 0.0
    sustained_locking_ratio: float = 0.0


@dataclass(frozen=True)
class WindowMetrics:
    """Aggregated metrics derived from a telemetry window.

    The payload captures the usual ΔNFR gradients and phase alignment values
    while also exposing support efficiency information derived from the average
    vertical load and nodal ΔNFR contributions.  ``support_effective`` reflects
    the structurally-weighted ΔNFR absorbed by tyres and suspension, whereas
    ``load_support_ratio`` normalises that magnitude against the window's mean
    vertical load.  The structural expansion/contraction fields quantify how
    longitudinal and lateral ΔNFR components expand (positive) or contract
    (negative) the structural timeline when weighted by structural occupancy
    windows.
    """

    si: float
    si_variance: float
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
    coherence_index: float
    ackermann_parallel_index: float
    support_effective: float
    load_support_ratio: float
    structural_expansion_longitudinal: float
    structural_contraction_longitudinal: float
    structural_expansion_lateral: float
    structural_contraction_lateral: float
    bottoming_ratio_front: float
    bottoming_ratio_rear: float
    mu_usage_front_ratio: float
    mu_usage_rear_ratio: float
    phase_mu_usage_front_ratio: float
    phase_mu_usage_rear_ratio: float
    exit_gear_match: float
    shift_stability: float
    frequency_label: str
    aero_coherence: AeroCoherence = field(default_factory=AeroCoherence)
    aero_mechanical_coherence: float = 0.0
    epi_derivative_abs: float = 0.0
    brake_headroom: BrakeHeadroom = field(default_factory=BrakeHeadroom)


def compute_window_metrics(
    records: Sequence[TelemetryRecord],
    *,
    phase_indices: Sequence[int] | None = None,
    bundles: Sequence[EPIBundle] | None = None,
    fallback_to_chronological: bool = True,
    objectives: object | None = None,
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
        return WindowMetrics(
            si=0.0,
            si_variance=0.0,
            d_nfr_couple=0.0,
            d_nfr_res=0.0,
            d_nfr_flat=0.0,
            nu_f=0.0,
            nu_exc=0.0,
            rho=0.0,
            phase_lag=0.0,
            phase_alignment=1.0,
            useful_dissonance_ratio=0.0,
            useful_dissonance_percentage=0.0,
            coherence_index=0.0,
            ackermann_parallel_index=0.0,
            support_effective=0.0,
            load_support_ratio=0.0,
            structural_expansion_longitudinal=0.0,
            structural_contraction_longitudinal=0.0,
            structural_expansion_lateral=0.0,
            structural_contraction_lateral=0.0,
            bottoming_ratio_front=0.0,
            bottoming_ratio_rear=0.0,
            mu_usage_front_ratio=0.0,
            mu_usage_rear_ratio=0.0,
            phase_mu_usage_front_ratio=0.0,
            phase_mu_usage_rear_ratio=0.0,
            exit_gear_match=0.0,
            shift_stability=0.0,
            frequency_label="",
            aero_coherence=AeroCoherence(),
            aero_mechanical_coherence=0.0,
            epi_derivative_abs=0.0,
            brake_headroom=BrakeHeadroom(),
        )

    def _compute_brake_headroom(
        samples: Sequence[TelemetryRecord],
    ) -> BrakeHeadroom:
        decel_values: list[float] = []
        locking_values: list[float] = []
        partial_samples: list[float] = []
        severe_samples: list[float] = []
        for record in samples:
            try:
                long_accel = float(getattr(record, "longitudinal_accel", 0.0))
            except (TypeError, ValueError):
                long_accel = 0.0
            if math.isfinite(long_accel):
                decel_values.append(max(0.0, -long_accel))
            try:
                locking = float(getattr(record, "locking", 0.0))
            except (TypeError, ValueError):
                locking = 0.0
            if not math.isfinite(locking):
                locking = 0.0
            locking_clamped = max(0.0, min(1.0, locking))
            locking_values.append(locking_clamped)
            slip_values: list[float] = []
            for suffix in _WHEEL_SUFFIXES:
                attr = f"slip_ratio_{suffix}"
                try:
                    slip_value = float(getattr(record, attr))
                except (TypeError, ValueError, AttributeError):
                    continue
                if not math.isfinite(slip_value):
                    continue
                slip_values.append(slip_value)
            if slip_values:
                partial_count = sum(
                    1
                    for slip in slip_values
                    if _PARTIAL_LOCK_LOWER <= slip <= _PARTIAL_LOCK_UPPER
                )
                severe_count = sum(1 for slip in slip_values if slip <= _SEVERE_LOCK_THRESHOLD)
                wheel_total = len(slip_values)
                partial_samples.append(partial_count / wheel_total)
                severe_samples.append(severe_count / wheel_total)
        if not decel_values:
            return BrakeHeadroom()
        peak_decel = max(decel_values)
        normalized_peak = min(1.0, peak_decel / _BRAKE_DECEL_REFERENCE)
        if locking_values:
            abs_activation = mean(locking_values)
            sustained_from_abs = sum(
                1.0 for value in locking_values if value >= _SUSTAINED_LOCK_ACTIVATION
            ) / len(locking_values)
        else:
            abs_activation = 0.0
            sustained_from_abs = 0.0
        partial_locking = mean(partial_samples) if partial_samples else 0.0
        sustained_from_slip = mean(severe_samples) if severe_samples else 0.0
        sustained_ratio = max(sustained_from_abs, sustained_from_slip)
        stress = 0.7 * abs_activation + 0.25 * partial_locking + 0.15 * sustained_ratio
        stress = max(0.0, min(1.0, stress))
        value = (1.0 - normalized_peak) * (1.0 - stress)
        if value < 0.0:
            value = 0.0
        if value > 1.0:
            value = 1.0
        return BrakeHeadroom(
            value=value,
            peak_decel=peak_decel,
            abs_activation_ratio=abs_activation,
            partial_locking_ratio=partial_locking,
            sustained_locking_ratio=sustained_ratio,
        )

    def _objective(name: str, default: float) -> float:
        if objectives is None:
            return default
        if isinstance(objectives, Mapping):
            candidate = objectives.get(name)
        else:
            candidate = getattr(objectives, name, None)
        try:
            numeric = float(candidate)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(numeric):
            return default
        return numeric

    si_samples = [float(record.si) for record in records]
    si_value = mean(si_samples)
    si_variance = pvariance(si_samples) if len(si_samples) >= 2 else 0.0
    avg_vertical_load = mean(getattr(record, "vertical_load", 0.0) for record in records)
    support_samples: list[float] = []
    longitudinal_series: list[float] = []
    lateral_series: list[float] = []
    suspension_series: list[float] = []
    tyre_series: list[float] = []
    front_travel_series: list[float] = []
    rear_travel_series: list[float] = []
    front_mu_lat_series: list[float] = []
    front_mu_long_series: list[float] = []
    rear_mu_lat_series: list[float] = []
    rear_mu_long_series: list[float] = []

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

    epi_abs_derivative = 0.0
    ackermann_samples: list[float] = []

    if bundles:
        timestamps = resolve_time_axis(
            bundles, fallback_to_chronological=fallback_to_chronological
        )
        if timestamps is None:
            if not fallback_to_chronological:
                raise ValueError("Structural timeline unavailable and fallback disabled")
            timestamps = [float(index) for index in range(len(bundles))]
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
        epi_values = [abs(float(getattr(bundle, "dEPI_dt", 0.0))) for bundle in bundles]
        if epi_values:
            epi_abs_derivative = mean(epi_values)
        support_samples = [
            max(0.0, float(bundle.tyres.delta_nfr))
            + max(0.0, float(bundle.suspension.delta_nfr))
            for bundle in bundles
        ]
        suspension_series = [
            float(getattr(bundle.suspension, "delta_nfr", 0.0)) for bundle in bundles
        ]
        tyre_series = [float(getattr(bundle.tyres, "delta_nfr", 0.0)) for bundle in bundles]
        front_travel_series = [
            float(getattr(bundle.suspension, "travel_front", 0.0)) for bundle in bundles
        ]
        rear_travel_series = [
            float(getattr(bundle.suspension, "travel_rear", 0.0)) for bundle in bundles
        ]
        front_mu_lat_series = [
            float(getattr(bundle.tyres, "mu_eff_front_lateral", 0.0)) for bundle in bundles
        ]
        front_mu_long_series = [
            float(getattr(bundle.tyres, "mu_eff_front_longitudinal", 0.0))
            for bundle in bundles
        ]
        rear_mu_lat_series = [
            float(getattr(bundle.tyres, "mu_eff_rear_lateral", 0.0)) for bundle in bundles
        ]
        rear_mu_long_series = [
            float(getattr(bundle.tyres, "mu_eff_rear_longitudinal", 0.0)) for bundle in bundles
        ]
        longitudinal_series = [
            float(getattr(bundle, "delta_nfr_longitudinal", 0.0)) for bundle in bundles
        ]
        lateral_series = [
            float(getattr(bundle, "delta_nfr_lateral", 0.0)) for bundle in bundles
        ]
        if phase_indices:
            ackermann_samples = [
                float(bundles[index].ackermann_parallel_index)
                for index in phase_indices
                if 0 <= index < len(bundles)
            ]
        else:
            ackermann_samples = [
                float(getattr(bundle, "ackermann_parallel_index", 0.0))
                for bundle in bundles
            ]
    else:
        timestamps = resolve_time_axis(
            records, fallback_to_chronological=fallback_to_chronological
        )
        if timestamps is None:
            if not fallback_to_chronological:
                raise ValueError("Structural timeline unavailable and fallback disabled")
            timestamps = [float(index) for index in range(len(records))]
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
        longitudinal_series = [
            float(getattr(record, "delta_nfr_longitudinal", 0.0)) for record in records
        ]
        lateral_series = [
            float(getattr(record, "delta_nfr_lateral", 0.0)) for record in records
        ]
        front_travel_series = [
            float(getattr(record, "suspension_travel_front", 0.0)) for record in records
        ]
        rear_travel_series = [
            float(getattr(record, "suspension_travel_rear", 0.0)) for record in records
        ]
        front_mu_lat_series = [
            float(getattr(record, "mu_eff_front_lateral", 0.0)) for record in records
        ]
        front_mu_long_series = [
            float(getattr(record, "mu_eff_front_longitudinal", 0.0)) for record in records
        ]
        rear_mu_lat_series = [
            float(getattr(record, "mu_eff_rear_lateral", 0.0)) for record in records
        ]
        rear_mu_long_series = [
            float(getattr(record, "mu_eff_rear_longitudinal", 0.0)) for record in records
        ]
    _useful_samples, _high_yaw_samples, udr = compute_useful_dissonance_stats(
        timestamps,
        delta_series,
        yaw_rates,
    )

    windows: list[float] = []
    if timestamps:
        windows = [
            max(0.0, float(timestamps[index] - timestamps[index - 1]))
            for index in range(1, len(timestamps))
        ]

    def _weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
        if not values:
            return 0.0
        if not weights or len(values) <= 1 or len(values) - 1 != len(weights):
            return mean(values)
        total_weight = sum(weights)
        if total_weight <= 0.0:
            return mean(values)
        accumulator = 0.0
        for index, value in enumerate(values[1:], start=1):
            accumulator += float(value) * weights[index - 1]
        return accumulator / total_weight

    def _expansion_payload(
        series: Sequence[float], weights: Sequence[float]
    ) -> tuple[float, float]:
        if len(series) < 2:
            return 0.0, 0.0
        if not weights or len(series) - 1 != len(weights):
            expansion = sum(max(0.0, float(value)) for value in series[1:])
            contraction = sum(max(0.0, float(-value)) for value in series[1:])
            count = max(1, len(series) - 1)
            return expansion / count, contraction / count
        total_weight = sum(weights)
        if total_weight <= 0.0:
            total_weight = float(len(weights))
        expansion = 0.0
        contraction = 0.0
        for value, weight in zip(series[1:], weights):
            magnitude = abs(float(value)) * weight
            if value >= 0.0:
                expansion += magnitude
            else:
                contraction += magnitude
        if total_weight <= 0.0:
            return 0.0, 0.0
        return expansion / total_weight, contraction / total_weight

    bottoming_threshold_front = max(
        0.0, _objective("bottoming_travel_threshold_front", 0.015)
    )
    bottoming_threshold_rear = max(
        0.0, _objective("bottoming_travel_threshold_rear", 0.015)
    )
    delta_peak_threshold = abs(_objective("bottoming_delta_nfr_threshold", 0.4))

    def _bottoming_ratio(
        travel_series: Sequence[float], threshold: float
    ) -> float:
        if not travel_series or not longitudinal_series:
            return 0.0
        bottoming_indices = [
            index for index, travel in enumerate(travel_series) if travel <= threshold
        ]
        if not bottoming_indices:
            return 0.0
        overlap = 0
        for index in bottoming_indices:
            if index >= len(longitudinal_series):
                continue
            magnitude = abs(float(longitudinal_series[index]))
            if delta_peak_threshold <= 0.0 or magnitude >= delta_peak_threshold:
                overlap += 1
        if delta_peak_threshold <= 0.0:
            denominator = len(bottoming_indices)
        else:
            peak_count = sum(
                1 for value in longitudinal_series if abs(float(value)) >= delta_peak_threshold
            )
            denominator = peak_count if peak_count > 0 else len(longitudinal_series)
        if denominator <= 0:
            return 0.0
        ratio = overlap / float(denominator)
        if ratio < 0.0:
            return 0.0
        if ratio > 1.0:
            return 1.0
        return ratio

    def _mu_usage_ratio(
        lat_series: Sequence[float],
        long_series: Sequence[float],
        weights: Sequence[float],
        mu_max: float,
    ) -> float:
        if not lat_series or not long_series:
            return 0.0
        magnitudes = [
            math.hypot(float(lat), float(long))
            for lat, long in zip(lat_series, long_series)
        ]
        average = _weighted_average(magnitudes, weights)
        if mu_max <= 1e-9:
            return 0.0
        ratio = average / mu_max
        if ratio < 0.0:
            return 0.0
        if ratio > 1.0:
            return 1.0
        return ratio

    def _phase_mu_usage_ratio(
        lat_series: Sequence[float],
        long_series: Sequence[float],
        mu_max: float,
    ) -> float:
        if not phase_indices:
            return 0.0
        if not lat_series or not long_series:
            return 0.0
        valid_pairs = [
            (
                index,
                math.hypot(float(lat_series[index]), float(long_series[index])),
            )
            for index in phase_indices
            if 0 <= index < len(lat_series) and 0 <= index < len(long_series)
        ]
        if not valid_pairs:
            return 0.0
        sorted_pairs = sorted(valid_pairs, key=lambda item: item[0])
        sorted_indices = [index for index, _ in sorted_pairs]
        sorted_values = [value for _, value in sorted_pairs]
        weights: list[float] = []
        if timestamps and len(sorted_indices) > 1:
            for previous, current in zip(sorted_indices, sorted_indices[1:]):
                if 0 <= previous < len(timestamps) and 0 <= current < len(timestamps):
                    weights.append(
                        max(0.0, float(timestamps[current]) - float(timestamps[previous]))
                    )
        average = _weighted_average(sorted_values, weights)
        if mu_max <= 1e-9:
            return 0.0
        ratio = average / mu_max
        if ratio < 0.0:
            return 0.0
        if ratio > 1.0:
            return 1.0
        return ratio

    support_effective = _weighted_average(support_samples, windows)
    load_support_ratio = (
        support_effective / avg_vertical_load if avg_vertical_load > 1e-6 else 0.0
    )
    long_expansion, long_contraction = _expansion_payload(longitudinal_series, windows)
    lat_expansion, lat_contraction = _expansion_payload(lateral_series, windows)

    def _exit_shift_metrics() -> tuple[float, float]:
        if not records:
            return 0.0, 0.0
        apex_index = 0
        min_speed = float("inf")
        for index, record in enumerate(records):
            try:
                speed_value = float(getattr(record, "speed", 0.0))
            except (TypeError, ValueError):
                speed_value = 0.0
            if speed_value < min_speed:
                min_speed = speed_value
                apex_index = index
        exit_window = records[apex_index:]
        if len(exit_window) <= 1:
            return 0.0, 1.0

        def _gear_ratio(record: TelemetryRecord) -> float | None:
            try:
                rpm_value = float(getattr(record, "rpm", 0.0))
                speed_value = float(getattr(record, "speed", 0.0))
            except (TypeError, ValueError):
                return None
            if speed_value <= 1e-6 or rpm_value <= 0.0:
                return None
            ratio_value = rpm_value / speed_value
            if not math.isfinite(ratio_value):
                return None
            return ratio_value

        shift_events = 0
        for previous, current in zip(exit_window, exit_window[1:]):
            try:
                previous_gear = int(getattr(previous, "gear", 0))
                current_gear = int(getattr(current, "gear", 0))
            except (TypeError, ValueError):
                continue
            if current_gear != previous_gear:
                shift_events += 1
        exit_span = max(1, len(exit_window) - 1)
        stability = 1.0 - min(1.0, shift_events / exit_span)
        if stability < 0.0:
            stability = 0.0

        exit_record = exit_window[-1]
        exit_ratio = _gear_ratio(exit_record)
        if exit_ratio is None:
            return 0.0, stability
        try:
            exit_gear = int(getattr(exit_record, "gear", 0))
        except (TypeError, ValueError):
            exit_gear = 0
        same_gear_ratios: list[float] = []
        for record in records:
            try:
                record_gear = int(getattr(record, "gear", 0))
            except (TypeError, ValueError):
                continue
            if record_gear != exit_gear:
                continue
            ratio_value = _gear_ratio(record)
            if ratio_value is None:
                continue
            same_gear_ratios.append(ratio_value)
        if not same_gear_ratios:
            return 0.0, stability
        baseline_ratio = mean(same_gear_ratios)
        if baseline_ratio <= 1e-9 or not math.isfinite(baseline_ratio):
            return 0.0, stability
        mismatch = abs(exit_ratio - baseline_ratio) / baseline_ratio
        if not math.isfinite(mismatch):
            return 0.0, stability
        gear_match = 1.0 - min(1.0, mismatch)
        if gear_match < 0.0:
            gear_match = 0.0
        return gear_match, stability

    exit_gear_match, shift_stability = _exit_shift_metrics()

    bottoming_ratio_front = _bottoming_ratio(front_travel_series, bottoming_threshold_front)
    bottoming_ratio_rear = _bottoming_ratio(rear_travel_series, bottoming_threshold_rear)

    mu_max_front = max(1e-6, _objective("mu_max_front", 2.0))
    mu_max_rear = max(1e-6, _objective("mu_max_rear", 2.0))
    mu_usage_front_ratio = _mu_usage_ratio(
        front_mu_lat_series, front_mu_long_series, windows, mu_max_front
    )
    mu_usage_rear_ratio = _mu_usage_ratio(
        rear_mu_lat_series, rear_mu_long_series, windows, mu_max_rear
    )
    phase_mu_usage_front_ratio = _phase_mu_usage_ratio(
        front_mu_lat_series, front_mu_long_series, mu_max_front
    )
    phase_mu_usage_rear_ratio = _phase_mu_usage_ratio(
        rear_mu_lat_series, rear_mu_long_series, mu_max_rear
    )

    aero = compute_aero_coherence(records, bundles)
    coherence_values: list[float] = []
    frequency_label = ""
    if bundles:
        coherence_values = [float(getattr(bundle, "coherence_index", 0.0)) for bundle in bundles]
        if bundles:
            frequency_label = str(getattr(bundles[-1], "nu_f_label", ""))
    raw_coherence = mean(coherence_values) if coherence_values else 0.0
    ackermann_parallel = mean(ackermann_samples) if ackermann_samples else 0.0
    target_si = max(1e-6, min(1.0, _objective("target_sense_index", 0.75)))
    si_factor = si_value / target_si if target_si > 0 else 0.0
    normalised_coherence = max(0.0, min(1.0, raw_coherence * si_factor))

    target_delta_nfr = abs(_objective("target_delta_nfr", 0.0))
    target_mechanical_ratio = max(0.0, min(1.0, _objective("target_mechanical_ratio", 0.55)))
    target_aero_imbalance = float(_objective("target_aero_imbalance", 0.12))
    aero_mechanical = resolve_aero_mechanical_coherence(
        normalised_coherence,
        aero,
        suspension_deltas=suspension_series,
        tyre_deltas=tyre_series,
        target_delta_nfr=target_delta_nfr,
        target_mechanical_ratio=target_mechanical_ratio,
        target_aero_imbalance=target_aero_imbalance,
    )

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
        coherence_index=normalised_coherence,
        ackermann_parallel_index=ackermann_parallel,
        support_effective=support_effective,
        load_support_ratio=load_support_ratio,
        structural_expansion_longitudinal=long_expansion,
        structural_contraction_longitudinal=long_contraction,
        structural_expansion_lateral=lat_expansion,
        structural_contraction_lateral=lat_contraction,
        bottoming_ratio_front=bottoming_ratio_front,
        bottoming_ratio_rear=bottoming_ratio_rear,
        mu_usage_front_ratio=mu_usage_front_ratio,
        mu_usage_rear_ratio=mu_usage_rear_ratio,
        phase_mu_usage_front_ratio=phase_mu_usage_front_ratio,
        phase_mu_usage_rear_ratio=phase_mu_usage_rear_ratio,
        exit_gear_match=exit_gear_match,
        shift_stability=shift_stability,
        frequency_label=frequency_label,
        aero_coherence=aero,
        aero_mechanical_coherence=aero_mechanical,
        epi_derivative_abs=epi_abs_derivative,
        si_variance=si_variance,
        brake_headroom=_compute_brake_headroom(records),
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


def resolve_aero_mechanical_coherence(
    coherence_index: float,
    aero: AeroCoherence,
    *,
    suspension_deltas: Sequence[float] | None = None,
    tyre_deltas: Sequence[float] | None = None,
    target_delta_nfr: float = 0.0,
    target_mechanical_ratio: float = 0.55,
    target_aero_imbalance: float = 0.12,
) -> float:
    """Return a blended aero-mechanical coherence indicator in ``[0, 1]``."""

    coherence = max(0.0, min(1.0, float(coherence_index)))
    if coherence <= 0.0 or not math.isfinite(coherence):
        return 0.0

    suspension_samples = [abs(float(value)) for value in suspension_deltas or ()]
    tyre_samples = [abs(float(value)) for value in tyre_deltas or ()]
    suspension_average = mean(suspension_samples) if suspension_samples else 0.0
    tyre_average = mean(tyre_samples) if tyre_samples else 0.0

    total_samples = max(0, int(aero.high_speed_samples) + int(aero.low_speed_samples))
    if total_samples > 0:
        high_weight = float(aero.high_speed_samples) / total_samples
        low_weight = float(aero.low_speed_samples) / total_samples
    else:
        high_weight = 0.5
        low_weight = 0.5

    aero_magnitude = (
        (abs(float(aero.high_speed_front)) + abs(float(aero.high_speed_rear)))
        * 0.5
        * high_weight
        + (abs(float(aero.low_speed_front)) + abs(float(aero.low_speed_rear)))
        * 0.5
        * low_weight
    )

    mechanical_ratio_target = max(0.0, min(1.0, float(target_mechanical_ratio)))
    mechanical_component = suspension_average
    combined_mechanical = suspension_average + tyre_average
    combined_total = mechanical_component + aero_magnitude
    if combined_total <= 1e-9:
        mechanical_ratio = mechanical_ratio_target
    else:
        mechanical_ratio = mechanical_component / combined_total
    if mechanical_ratio >= mechanical_ratio_target:
        ratio_span = max(1e-6, mechanical_ratio_target)
    else:
        ratio_span = max(1e-6, 1.0 - mechanical_ratio_target)
    ratio_error = abs(mechanical_ratio - mechanical_ratio_target)
    ratio_factor = max(0.0, 1.0 - min(1.0, ratio_error / ratio_span))

    support_total = combined_mechanical + aero_magnitude
    target_delta = max(0.0, float(target_delta_nfr))
    if target_delta > 1e-6:
        coverage_factor = max(0.0, min(1.0, support_total / target_delta))
    else:
        coverage_factor = 1.0 if support_total > 0.0 else 0.0

    imbalance_target = max(0.05, abs(float(target_aero_imbalance)))
    weighted_imbalance = (
        abs(float(aero.high_speed_imbalance)) * high_weight
        + abs(float(aero.low_speed_imbalance)) * low_weight
    )
    aero_factor = max(0.0, 1.0 - min(1.0, weighted_imbalance / imbalance_target))
    if total_samples == 0:
        aero_factor *= 0.5

    composite = (ratio_factor + coverage_factor + aero_factor) / 3.0
    return max(0.0, min(1.0, coherence * composite))


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

