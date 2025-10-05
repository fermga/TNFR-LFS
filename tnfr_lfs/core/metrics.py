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
from .dissonance import YAW_ACCELERATION_THRESHOLD, compute_useful_dissonance_stats
from .epi import TelemetryRecord
from .epi_models import EPIBundle
from .spectrum import phase_alignment
from .resonance import estimate_excitation_frequency
from .structural_time import resolve_time_axis

__all__ = [
    "AeroCoherence",
    "BrakeHeadroom",
    "SlideCatchBudget",
    "BumpstopHistogram",
    "CPHIWheel",
    "WindowMetrics",
    "compute_window_metrics",
    "compute_aero_coherence",
    "resolve_aero_mechanical_coherence",
]


_BUMPSTOP_DEPTH_BINS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


def _zero_bump_bins() -> tuple[float, ...]:
    return tuple(0.0 for _ in _BUMPSTOP_DEPTH_BINS)


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

_FRONT_LATERAL_FEATURE_KEYS = {"mu_eff_front_lateral"}
_FRONT_LONGITUDINAL_FEATURE_KEYS = {"mu_eff_front_longitudinal"}
_REAR_LATERAL_FEATURE_KEYS = {"mu_eff_rear_lateral"}
_REAR_LONGITUDINAL_FEATURE_KEYS = {"mu_eff_rear_longitudinal"}


_BRAKE_DECEL_REFERENCE = 10.0
_PARTIAL_LOCK_LOWER = -0.35
_PARTIAL_LOCK_UPPER = -0.06
_SEVERE_LOCK_THRESHOLD = -0.45
_SUSTAINED_LOCK_ACTIVATION = 0.75
_WHEEL_SUFFIXES = ("fl", "fr", "rl", "rr")


_STEER_VELOCITY_THRESHOLD = 3.5
_ACKERMANN_OVERSHOOT_REFERENCE = 0.18


@dataclass(frozen=True)
class SlideCatchBudget:
    """Composite steering margin metric derived from yaw and steer activity."""

    value: float = 0.0
    yaw_acceleration_ratio: float = 0.0
    steer_velocity_ratio: float = 0.0
    overshoot_ratio: float = 0.0


@dataclass(frozen=True)
class AeroAxisCoherence:
    """Front/rear contribution pair for a given aerodynamic axis."""

    front: float = 0.0
    rear: float = 0.0

    @property
    def imbalance(self) -> float:
        """Signed imbalance favouring the front when negative."""

        return self.front - self.rear


@dataclass(frozen=True)
class AeroBandCoherence:
    """Per-speed-band aerodynamic coherence split by axes."""

    total: AeroAxisCoherence = field(default_factory=AeroAxisCoherence)
    lateral: AeroAxisCoherence = field(default_factory=AeroAxisCoherence)
    longitudinal: AeroAxisCoherence = field(default_factory=AeroAxisCoherence)
    samples: int = 0


@dataclass(frozen=True)
class AeroCoherence:
    """Summarises aero balance deltas split by speed bins and axes."""

    low_speed: AeroBandCoherence = field(default_factory=AeroBandCoherence)
    medium_speed: AeroBandCoherence = field(default_factory=AeroBandCoherence)
    high_speed: AeroBandCoherence = field(default_factory=AeroBandCoherence)
    guidance: str = ""

    def dominant_axis(self, tolerance: float = 0.05) -> str | None:
        """Return the dominant axle when the imbalance exceeds ``tolerance``."""

        imbalance = self.high_speed.total.imbalance
        if abs(imbalance) <= tolerance:
            return None
        return "front" if imbalance < 0.0 else "rear"

    @property
    def low_speed_front(self) -> float:
        return self.low_speed.total.front

    @property
    def low_speed_rear(self) -> float:
        return self.low_speed.total.rear

    @property
    def low_speed_imbalance(self) -> float:
        return self.low_speed.total.imbalance

    @property
    def low_speed_samples(self) -> int:
        return self.low_speed.samples

    @property
    def medium_speed_front(self) -> float:
        return self.medium_speed.total.front

    @property
    def medium_speed_rear(self) -> float:
        return self.medium_speed.total.rear

    @property
    def medium_speed_imbalance(self) -> float:
        return self.medium_speed.total.imbalance

    @property
    def medium_speed_samples(self) -> int:
        return self.medium_speed.samples

    @property
    def high_speed_front(self) -> float:
        return self.high_speed.total.front

    @property
    def high_speed_rear(self) -> float:
        return self.high_speed.total.rear

    @property
    def high_speed_imbalance(self) -> float:
        return self.high_speed.total.imbalance

    @property
    def high_speed_samples(self) -> int:
        return self.high_speed.samples


@dataclass(frozen=True)
class BrakeHeadroom:
    """Aggregated braking capacity metrics for the current window."""

    value: float = 0.0
    peak_decel: float = 0.0
    abs_activation_ratio: float = 0.0
    partial_locking_ratio: float = 0.0
    sustained_locking_ratio: float = 0.0


@dataclass(frozen=True)
class BumpstopHistogram:
    """Density and energy accumulated in the bump stop zone by axle."""

    depth_bins: tuple[float, ...] = _BUMPSTOP_DEPTH_BINS
    front_density: tuple[float, ...] = field(default_factory=_zero_bump_bins)
    rear_density: tuple[float, ...] = field(default_factory=_zero_bump_bins)
    front_energy: tuple[float, ...] = field(default_factory=_zero_bump_bins)
    rear_energy: tuple[float, ...] = field(default_factory=_zero_bump_bins)
    front_total_density: float = 0.0
    rear_total_density: float = 0.0
    front_total_energy: float = 0.0
    rear_total_energy: float = 0.0


@dataclass(frozen=True)
class CPHIWheel:
    """Contact Patch Health Index components for a single wheel."""

    value: float = 1.0
    temperature_component: float = 0.0
    gradient_component: float = 0.0
    mu_component: float = 0.0
    temperature_delta: float = 0.0
    gradient_rate: float = 0.0


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
    windows. The ``bumpstop_histogram`` field captures the occupancy density and
    ΔNFR energy accumulated when the suspension operates within the bump stop
    envelope for each axle.
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
    slide_catch_budget: SlideCatchBudget
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
    bumpstop_histogram: BumpstopHistogram = field(default_factory=BumpstopHistogram)
    cphi: Mapping[str, CPHIWheel] = field(default_factory=dict)
    phase_cphi: Mapping[str, Mapping[str, CPHIWheel]] = field(default_factory=dict)


def _compute_bumpstop_histogram(
    front_series: Sequence[float],
    rear_series: Sequence[float],
    *,
    front_threshold: float,
    rear_threshold: float,
    energy_series: Sequence[float],
) -> BumpstopHistogram:
    if not front_series and not rear_series:
        return BumpstopHistogram()

    front_values = list(front_series)
    rear_values = list(rear_series)
    energy_values = list(energy_series)
    bin_count = len(_BUMPSTOP_DEPTH_BINS)
    front_density_counts = [0.0] * bin_count
    rear_density_counts = [0.0] * bin_count
    front_energy_bins = [0.0] * bin_count
    rear_energy_bins = [0.0] * bin_count
    front_total_samples = float(len(front_values))
    rear_total_samples = float(len(rear_values))
    sample_count = max(len(front_values), len(rear_values), len(energy_values))
    if sample_count <= 0:
        return BumpstopHistogram()

    def _depth(value: float, threshold: float) -> float:
        if threshold <= 1e-12:
            return 0.0
        remaining = threshold - value
        if remaining <= 0.0:
            return 0.0
        ratio = remaining / threshold
        if ratio < 0.0:
            return 0.0
        if ratio > 1.0:
            return 1.0
        return ratio

    def _bin_index(depth_ratio: float) -> int | None:
        if depth_ratio <= 0.0:
            return None
        for index, boundary in enumerate(_BUMPSTOP_DEPTH_BINS):
            if depth_ratio <= boundary:
                return index
        return bin_count - 1 if bin_count else None

    for index in range(sample_count):
        front_present = index < len(front_values)
        rear_present = index < len(rear_values)
        front_value = float(front_values[index]) if front_present else 0.0
        rear_value = float(rear_values[index]) if rear_present else 0.0
        energy_value = (
            abs(float(energy_values[index])) if index < len(energy_values) else 0.0
        )
        front_depth = _depth(front_value, front_threshold) if front_present else 0.0
        rear_depth = _depth(rear_value, rear_threshold) if rear_present else 0.0
        total_depth = front_depth + rear_depth
        energy_front = energy_value * (front_depth / total_depth) if total_depth > 0.0 else 0.0
        energy_rear = energy_value * (rear_depth / total_depth) if total_depth > 0.0 else 0.0

        front_bin = _bin_index(front_depth)
        if front_bin is not None and front_present:
            front_density_counts[front_bin] += 1.0
            front_energy_bins[front_bin] += energy_front

        rear_bin = _bin_index(rear_depth)
        if rear_bin is not None and rear_present:
            rear_density_counts[rear_bin] += 1.0
            rear_energy_bins[rear_bin] += energy_rear

    def _normalise_counts(counts: Sequence[float], total: float) -> tuple[float, ...]:
        if total <= 0.0:
            return _zero_bump_bins()
        return tuple(count / total for count in counts)

    front_density = _normalise_counts(front_density_counts, front_total_samples)
    rear_density = _normalise_counts(rear_density_counts, rear_total_samples)
    front_total_density = sum(front_density)
    rear_total_density = sum(rear_density)
    front_total_energy = sum(front_energy_bins)
    rear_total_energy = sum(rear_energy_bins)

    return BumpstopHistogram(
        front_density=front_density,
        rear_density=rear_density,
        front_energy=tuple(front_energy_bins),
        rear_energy=tuple(rear_energy_bins),
        front_total_density=front_total_density,
        rear_total_density=rear_total_density,
        front_total_energy=front_total_energy,
        rear_total_energy=rear_total_energy,
    )


def compute_window_metrics(
    records: Sequence[TelemetryRecord],
    *,
    phase_indices: Sequence[int] | Mapping[str, Sequence[int]] | None = None,
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
            slide_catch_budget=SlideCatchBudget(),
            support_effective=0.0,
            load_support_ratio=0.0,
            structural_expansion_longitudinal=0.0,
            structural_contraction_longitudinal=0.0,
            structural_expansion_lateral=0.0,
            structural_contraction_lateral=0.0,
            bottoming_ratio_front=0.0,
            bottoming_ratio_rear=0.0,
            bumpstop_histogram=BumpstopHistogram(),
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
            cphi={},
            phase_cphi={},
        )

    if isinstance(phase_indices, Mapping):
        phase_windows: dict[str, tuple[int, ...]] = {}
        for key, window in phase_indices.items():
            if not isinstance(window, Sequence):
                continue
            indices = tuple(
                int(index)
                for index in window
                if isinstance(index, (int, float))
            )
            if indices:
                phase_windows[str(key)] = indices
    elif phase_indices is None:
        phase_windows = {}
    else:
        indices = tuple(
            int(index) for index in phase_indices if isinstance(index, (int, float))
        )
        phase_windows = {"active": indices} if indices else {}

    primary_phase_indices = next(iter(phase_windows.values()), ())

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
    wheel_temperature_series: dict[str, list[float]] = {
        suffix: [] for suffix in _WHEEL_SUFFIXES
    }

    context_matrix = load_context_matrix()

    if primary_phase_indices:
        selected = [
            records[index]
            for index in primary_phase_indices
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
    steer_series: list[float] = []

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
        steer_series = [float(getattr(bundle.driver, "steer", 0.0)) for bundle in bundles]
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
        for suffix in _WHEEL_SUFFIXES:
            attribute = f"tyre_temp_{suffix}"
            series = wheel_temperature_series[suffix]
            for bundle in bundles:
                try:
                    series.append(float(getattr(bundle.tyres, attribute, 0.0)))
                except (TypeError, ValueError):
                    series.append(0.0)
        longitudinal_series = [
            float(getattr(bundle, "delta_nfr_longitudinal", 0.0)) for bundle in bundles
        ]
        lateral_series = [
            float(getattr(bundle, "delta_nfr_lateral", 0.0)) for bundle in bundles
        ]
        if primary_phase_indices:
            ackermann_samples = [
                float(bundles[index].ackermann_parallel_index)
                for index in primary_phase_indices
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
        steer_series = [float(getattr(record, "steer", 0.0)) for record in records]
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
        for suffix in _WHEEL_SUFFIXES:
            attribute = f"tyre_temp_{suffix}"
            series = wheel_temperature_series[suffix]
            for record in records:
                try:
                    series.append(float(getattr(record, attribute, 0.0)))
                except (TypeError, ValueError):
                    series.append(0.0)
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
        indices: Sequence[int] | None,
    ) -> float:
        if not indices:
            return 0.0
        if not lat_series or not long_series:
            return 0.0
        valid_pairs = [
            (
                index,
                math.hypot(float(lat_series[index]), float(long_series[index])),
            )
            for index in indices
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
    bumpstop_histogram = _compute_bumpstop_histogram(
        front_travel_series,
        rear_travel_series,
        front_threshold=bottoming_threshold_front,
        rear_threshold=bottoming_threshold_rear,
        energy_series=suspension_series,
    )

    mu_max_front = max(1e-6, _objective("mu_max_front", 2.0))
    mu_max_rear = max(1e-6, _objective("mu_max_rear", 2.0))
    mu_usage_front_ratio = _mu_usage_ratio(
        front_mu_lat_series, front_mu_long_series, windows, mu_max_front
    )
    mu_usage_rear_ratio = _mu_usage_ratio(
        rear_mu_lat_series, rear_mu_long_series, windows, mu_max_rear
    )
    phase_mu_usage_front_ratio = _phase_mu_usage_ratio(
        front_mu_lat_series, front_mu_long_series, mu_max_front, primary_phase_indices
    )
    phase_mu_usage_rear_ratio = _phase_mu_usage_ratio(
        rear_mu_lat_series, rear_mu_long_series, mu_max_rear, primary_phase_indices
    )
    phase_mu_usage_map: dict[str, tuple[float, float]] = {}
    for phase_label, indices in phase_windows.items():
        phase_mu_usage_map[phase_label] = (
            _phase_mu_usage_ratio(
                front_mu_lat_series, front_mu_long_series, mu_max_front, indices
            ),
            _phase_mu_usage_ratio(
                rear_mu_lat_series, rear_mu_long_series, mu_max_rear, indices
            ),
        )

    def _rate_series(series: Sequence[float], stamps: Sequence[float]) -> list[float]:
        rates: list[float] = []
        length = min(len(series), len(stamps))
        for index in range(1, length):
            dt = float(stamps[index] - stamps[index - 1])
            if dt <= 0.0 or not math.isfinite(dt):
                continue
            delta = float(series[index] - series[index - 1])
            rate = delta / dt
            if math.isfinite(rate):
                rates.append(rate)
        return rates

    def _normalised_ratio(value: float, reference: float) -> float:
        if reference <= 1e-9:
            return 0.0
        ratio = value / reference
        if ratio < 0.0:
            return 0.0
        if ratio > 1.0:
            return 1.0
        return ratio

    target_front_temperature = float(_objective("target_front_temperature", 82.0))
    target_rear_temperature = float(_objective("target_rear_temperature", 80.0))
    temperature_tolerance = max(
        1.0, float(_objective("target_temperature_tolerance", 6.0))
    )
    gradient_reference = max(
        1e-3, float(_objective("target_temperature_gradient", 1.2))
    )
    target_mu_front = max(
        0.0, min(1.0, float(_objective("target_mu_usage_front", 0.88)))
    )
    target_mu_rear = max(
        0.0, min(1.0, float(_objective("target_mu_usage_rear", 0.85)))
    )
    cphi_weight_temperature = max(
        0.0, float(_objective("cphi_weight_temperature", 0.5))
    )
    cphi_weight_gradient = max(
        0.0, float(_objective("cphi_weight_gradient", 0.3))
    )
    cphi_weight_mu = max(0.0, float(_objective("cphi_weight_mu", 0.2)))
    cphi_weight_sum = max(
        1e-6, cphi_weight_temperature + cphi_weight_gradient + cphi_weight_mu
    )

    def _compute_cphi_for_indices(
        indices: Sequence[int] | None,
        front_ratio: float,
        rear_ratio: float,
    ) -> dict[str, CPHIWheel]:
        report: dict[str, CPHIWheel] = {}
        if indices:
            usable_indices = tuple(int(index) for index in indices)
        else:
            usable_indices = ()
        for suffix in _WHEEL_SUFFIXES:
            temps = wheel_temperature_series.get(suffix, [])
            sampled_values: list[float] = []
            sampled_times: list[float] = []
            if usable_indices:
                for index in usable_indices:
                    if 0 <= index < len(temps) and 0 <= index < len(timestamps):
                        sampled_values.append(float(temps[index]))
                        sampled_times.append(float(timestamps[index]))
            else:
                limit = min(len(temps), len(timestamps))
                for index in range(limit):
                    sampled_values.append(float(temps[index]))
                    sampled_times.append(float(timestamps[index]))
            if not sampled_values:
                sampled_values = [float(value) for value in temps[: len(timestamps)]]
                sampled_times = [float(stamp) for stamp in timestamps[: len(sampled_values)]]
            rates = _rate_series(sampled_values, sampled_times)
            gradient_mean = mean(rates) if rates else 0.0
            gradient_abs = mean(abs(rate) for rate in rates) if rates else 0.0
            if suffix in {"fl", "fr"}:
                target_temp = target_front_temperature
                mu_ratio = float(front_ratio)
                mu_target = target_mu_front
            else:
                target_temp = target_rear_temperature
                mu_ratio = float(rear_ratio)
                mu_target = target_mu_rear
            temp_delta = mean(sampled_values) - target_temp if sampled_values else -target_temp
            temp_penalty = min(1.0, abs(temp_delta) / temperature_tolerance)
            gradient_penalty = min(1.0, gradient_abs / gradient_reference)
            if mu_target >= 1.0:
                mu_penalty = 0.0
            elif mu_ratio <= mu_target:
                mu_penalty = 0.0
            else:
                mu_penalty = min(1.0, (mu_ratio - mu_target) / max(1e-6, 1.0 - mu_target))
            temp_component = (cphi_weight_temperature / cphi_weight_sum) * temp_penalty
            gradient_component = (cphi_weight_gradient / cphi_weight_sum) * gradient_penalty
            mu_component = (cphi_weight_mu / cphi_weight_sum) * mu_penalty
            value = max(0.0, 1.0 - (temp_component + gradient_component + mu_component))
            report[suffix] = CPHIWheel(
                value=value,
                temperature_component=temp_component,
                gradient_component=gradient_component,
                mu_component=mu_component,
                temperature_delta=temp_delta,
                gradient_rate=gradient_mean,
            )
        return report

    yaw_acceleration_series = _rate_series(yaw_rates, timestamps)
    steer_velocity_series = _rate_series(steer_series, timestamps)
    yaw_accel_average = mean(abs(sample) for sample in yaw_acceleration_series) if yaw_acceleration_series else 0.0
    steer_velocity_average = mean(abs(sample) for sample in steer_velocity_series) if steer_velocity_series else 0.0
    overshoot_average = mean(abs(sample) for sample in ackermann_samples) if ackermann_samples else 0.0

    yaw_ratio = _normalised_ratio(yaw_accel_average, YAW_ACCELERATION_THRESHOLD)
    steer_ratio = _normalised_ratio(steer_velocity_average, _STEER_VELOCITY_THRESHOLD)
    overshoot_ratio = _normalised_ratio(overshoot_average, _ACKERMANN_OVERSHOOT_REFERENCE)
    combined_load = 0.5 * yaw_ratio + 0.3 * steer_ratio + 0.2 * overshoot_ratio
    if combined_load > 1.0:
        combined_load = 1.0
    slide_catch_budget = SlideCatchBudget(
        value=max(0.0, 1.0 - combined_load),
        yaw_acceleration_ratio=yaw_ratio,
        steer_velocity_ratio=steer_ratio,
        overshoot_ratio=overshoot_ratio,
    )

    cphi_overall = _compute_cphi_for_indices(
        None, mu_usage_front_ratio, mu_usage_rear_ratio
    )
    phase_cphi: dict[str, Mapping[str, CPHIWheel]] = {}
    for phase_label, indices in phase_windows.items():
        front_ratio, rear_ratio = phase_mu_usage_map.get(
            phase_label, (mu_usage_front_ratio, mu_usage_rear_ratio)
        )
        phase_cphi[phase_label] = _compute_cphi_for_indices(
            indices, front_ratio, rear_ratio
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
        slide_catch_budget=slide_catch_budget,
        support_effective=support_effective,
        load_support_ratio=load_support_ratio,
        structural_expansion_longitudinal=long_expansion,
        structural_contraction_longitudinal=long_contraction,
        structural_expansion_lateral=lat_expansion,
        structural_contraction_lateral=lat_contraction,
        bottoming_ratio_front=bottoming_ratio_front,
        bottoming_ratio_rear=bottoming_ratio_rear,
        bumpstop_histogram=bumpstop_histogram,
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
        cphi=cphi_overall,
        phase_cphi=phase_cphi,
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

    axis_keys = {
        "total": (_FRONT_FEATURE_KEYS, _REAR_FEATURE_KEYS),
        "lateral": (_FRONT_LATERAL_FEATURE_KEYS, _REAR_LATERAL_FEATURE_KEYS),
        "longitudinal": (
            _FRONT_LONGITUDINAL_FEATURE_KEYS,
            _REAR_LONGITUDINAL_FEATURE_KEYS,
        ),
    }

    def _aero_components(
        breakdown: Mapping[str, Mapping[str, float]] | None,
    ) -> dict[str, tuple[float, float]]:
        totals = {axis: [0.0, 0.0] for axis in axis_keys}
        if not breakdown:
            return {axis: (0.0, 0.0) for axis in axis_keys}
        for features in breakdown.values():
            if not isinstance(features, Mapping):
                continue
            for key, value in features.items():
                try:
                    contribution = float(value)
                except (TypeError, ValueError):
                    continue
                for axis, (front_keys, rear_keys) in axis_keys.items():
                    if key in front_keys:
                        totals[axis][0] += contribution
                    if key in rear_keys:
                        totals[axis][1] += contribution
        return {axis: (front, rear) for axis, (front, rear) in totals.items()}

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

    bands: dict[str, dict[str, list[float] | int]] = {
        "low": {"total": [0.0, 0.0], "lateral": [0.0, 0.0], "longitudinal": [0.0, 0.0], "samples": 0},
        "medium": {
            "total": [0.0, 0.0],
            "lateral": [0.0, 0.0],
            "longitudinal": [0.0, 0.0],
            "samples": 0,
        },
        "high": {"total": [0.0, 0.0], "lateral": [0.0, 0.0], "longitudinal": [0.0, 0.0], "samples": 0},
    }

    for index, bundle in enumerate(bundles):
        speed = _resolve_speed(index)
        if speed is None:
            continue
        components = _aero_components(getattr(bundle, "delta_breakdown", {}))
        if all(front == 0.0 and rear == 0.0 for front, rear in components.values()):
            continue
        if speed <= low_speed_threshold:
            target = bands["low"]
        elif speed >= high_speed_threshold:
            target = bands["high"]
        else:
            target = bands["medium"]
        target["samples"] = int(target["samples"]) + 1
        for axis, (front, rear) in components.items():
            pair = target[axis]
            pair[0] += front
            pair[1] += rear

    def _average_band(payload: dict[str, list[float] | int]) -> AeroBandCoherence:
        samples = int(payload["samples"])

        def _average_pair(values: list[float]) -> tuple[float, float]:
            if samples:
                return values[0] / samples, values[1] / samples
            return 0.0, 0.0

        total_front, total_rear = _average_pair(payload["total"])
        lat_front, lat_rear = _average_pair(payload["lateral"])
        long_front, long_rear = _average_pair(payload["longitudinal"])
        return AeroBandCoherence(
            total=AeroAxisCoherence(total_front, total_rear),
            lateral=AeroAxisCoherence(lat_front, lat_rear),
            longitudinal=AeroAxisCoherence(long_front, long_rear),
            samples=samples,
        )

    low_band = _average_band(bands["low"])
    medium_band = _average_band(bands["medium"])
    high_band = _average_band(bands["high"])

    low_imbalance = low_band.total.imbalance
    medium_imbalance = medium_band.total.imbalance
    high_imbalance = high_band.total.imbalance

    guidance: str
    if high_band.samples == 0:
        guidance = "Sin datos aero"
    elif abs(high_imbalance) <= imbalance_tolerance:
        guidance = "Aero alta velocidad equilibrado"
    elif high_imbalance > 0.0:
        guidance = "Alta velocidad → sube alerón trasero"
    else:
        guidance = "Alta velocidad → libera alerón trasero/refuerza delantero"

    if low_band.samples and abs(low_imbalance) > imbalance_tolerance:
        direction = "trasero" if low_imbalance > 0.0 else "delantero"
        guidance += f" · baja velocidad sesgo {direction}"
    if medium_band.samples and abs(medium_imbalance) > imbalance_tolerance:
        direction = "trasero" if medium_imbalance > 0.0 else "delantero"
        guidance += f" · media velocidad sesgo {direction}"

    return AeroCoherence(
        low_speed=low_band,
        medium_speed=medium_band,
        high_speed=high_band,
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

    total_samples = max(
        0,
        int(aero.high_speed_samples)
        + int(aero.medium_speed_samples)
        + int(aero.low_speed_samples),
    )
    if total_samples > 0:
        high_weight = float(aero.high_speed_samples) / total_samples
        medium_weight = float(aero.medium_speed_samples) / total_samples
        low_weight = float(aero.low_speed_samples) / total_samples
    else:
        high_weight = medium_weight = low_weight = 1.0 / 3.0

    aero_magnitude = (
        (abs(float(aero.high_speed_front)) + abs(float(aero.high_speed_rear)))
        * 0.5
        * high_weight
        + (abs(float(aero.medium_speed_front)) + abs(float(aero.medium_speed_rear)))
        * 0.5
        * medium_weight
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
        + abs(float(aero.medium_speed_imbalance)) * medium_weight
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

