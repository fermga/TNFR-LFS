"""Windowed telemetry metrics used by the HUD and setup planner."""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from statistics import mean, pvariance, pstdev
from typing import Iterable, Iterator, Mapping, Sequence, Tuple

import numpy as np

import tnfr_core.equations.epi as _epi

from tnfr_core.equations.contextual_delta import (
    ContextMatrix,
    apply_contextual_delta,
    load_context_matrix,
    resolve_context_from_bundle,
    resolve_context_from_record,
)
from tnfr_core.equations.dissonance import YAW_ACCELERATION_THRESHOLD, compute_useful_dissonance_stats
from tnfr_core.runtime.shared import (
    SupportsChassisNode,
    SupportsContextBundle,
    SupportsContextRecord,
    SupportsDriverNode,
    SupportsEPIBundle,
    SupportsSuspensionNode,
    SupportsTelemetrySample,
    SupportsTyresNode,
    _HAS_JAX,
    jnp,
)
from tnfr_core.equations.phases import replicate_phase_aliases
from tnfr_core.metrics.spectrum import (
    PhaseCorrelation,
    motor_input_correlations,
    phase_alignment,
    phase_to_latency_ms,
)
from tnfr_core.metrics.resonance import estimate_excitation_frequency
from tnfr_core.operators.structural_time import resolve_time_axis
from tnfr_core.equations.utils import normalised_entropy

__all__ = [
    "AeroBalanceDrift",
    "AeroBalanceDriftBin",
    "AeroCoherence",
    "AeroAxisCoherence",
    "AeroBandCoherence",
    "BrakeHeadroom",
    "SlideCatchBudget",
    "LockingWindowScore",
    "BumpstopHistogram",
    "CPHIWheel",
    "CPHIWheelComponents",
    "CPHIThresholds",
    "CPHIReport",
    "WindowMetrics",
    "SuspensionVelocityBands",
    "compute_window_metrics",
    "compute_aero_coherence",
    "resolve_aero_mechanical_coherence",
    "phase_synchrony_index",
    "coherence_total",
    "psi_norm",
    "psi_support",
    "bifurcation_threshold",
    "mutation_threshold",
    "delta_nfr_by_node",
]


delta_nfr_by_node = _epi.delta_nfr_by_node


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
_FADE_PRESSURE_THRESHOLD = 0.6
_FADE_PRESSURE_VARIATION = 0.12
_FADE_MIN_DURATION = 0.35
_FADE_MAX_GAP = 0.6
_FADE_MIN_START_DECEL = 3.0
_FADE_MIN_DROP_RATIO = 0.05
_VENT_TEMP_WARNING = 600.0
_VENT_TEMP_CRITICAL = 720.0
_FADE_RATIO_WARNING = 0.12
_FADE_RATIO_CRITICAL = 0.22
_FADE_SLOPE_CRITICAL = 0.75
_WHEEL_SUFFIXES = ("fl", "fr", "rl", "rr")


_STEER_VELOCITY_THRESHOLD = 3.5
_ACKERMANN_OVERSHOOT_REFERENCE = math.radians(5.0)

_LOCKING_TRANSITION_DELTA = 0.15
_LOCKING_TRANSITION_LOW = 0.25
_LOCKING_TRANSITION_HIGH = 0.55
_LOCKING_YAW_REFERENCE = 1.0
_LOCKING_LONGITUDINAL_REFERENCE = 260.0
_LOCKING_THROTTLE_REFERENCE = 0.5


_CPHI_SLIP_RATIO_REFERENCE = 0.12
_CPHI_SLIP_ANGLE_REFERENCE = math.radians(7.0)
_CPHI_MU_REFERENCE = 1.2
_CPHI_LOAD_BIAS_REFERENCE = 0.25


def _safe_numeric(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _mean_optional(values: Sequence[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _normalised_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Return the normalised probability distribution for ``weights``."""

    positive: dict[str, float] = {}
    for key, raw_value in weights.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        if value <= 0.0:
            continue
        positive[str(key)] = positive.get(str(key), 0.0) + value
    total = sum(positive.values())
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in positive.items()}


def coherence_total(
    samples: Sequence[tuple[float, float]] | Mapping[float, float], *, initial: float = 0.0
) -> list[tuple[float, float]]:
    """Return the cumulative TNFR coherence ``C(t)`` profile.

    The HUD treats coherence as accumulated evidence that driver, chassis and
    aero signals are converging.  Under this interpretation the curve must be
    monotonic and any destabilising impulse is clipped rather than subtracted.

    Parameters
    ----------
    samples:
        Sequence or mapping of ``(time, delta)`` contributions.  ``time`` is
        converted to ``float`` and used for sorting.
    initial:
        Starting value of the cumulative signal.  Negative values are clipped to
        zero to preserve monotonicity.

    Returns
    -------
    list[tuple[float, float]]
        Sorted ``(time, cumulative)`` pairs representing ``C(t)``.

    Examples
    --------
    >>> coherence_total([(0.0, 0.1), (0.5, -0.2), (0.75, 0.4)])
    [(0.0, 0.1), (0.5, 0.1), (0.75, 0.5)]
    >>> coherence_total({0.5: 0.2, 0.0: 0.1})
    [(0.0, 0.1), (0.5, 0.30000000000000004)]
    """

    if isinstance(samples, MappingABC):
        iterator: Iterable[tuple[float, float]] = samples.items()
    else:
        iterator = samples

    cleaned: list[tuple[float, float]] = []
    for time, delta in iterator:
        try:
            timestamp = float(time)
            contribution = float(delta)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(timestamp) or not math.isfinite(contribution):
            continue
        cleaned.append((timestamp, contribution))

    cleaned.sort(key=lambda item: item[0])

    cumulative = max(0.0, float(initial))
    profile: list[tuple[float, float]] = []
    for timestamp, contribution in cleaned:
        if contribution > 0.0:
            cumulative += contribution
        profile.append((timestamp, cumulative))
    return profile


def psi_norm(values: Sequence[float], *, ord: float = 2.0) -> float:
    """Return the ℓₚ norm of a PSI vector.

    In TNFR the phase synchrony index (PSI) vector captures modal coupling
    between driver, chassis and tyre responses.  The Euclidean (``p=2``)
    norm therefore provides a reproducible aggregate synchrony score.

    Parameters
    ----------
    values:
        Iterable containing PSI components.
    ord:
        Order of the ℓₚ norm.  ``math.inf`` returns the maximum
        absolute component.

    Examples
    --------
    >>> psi_norm([0.4, 0.3])
    0.5
    >>> psi_norm([0.1, -0.8], ord=math.inf)
    0.8
    """

    cleaned = [float(value) for value in values if math.isfinite(float(value))]
    if not cleaned:
        return 0.0
    if ord is math.inf:
        return max(abs(value) for value in cleaned)
    if ord <= 0.0:
        raise ValueError("ord must be positive or math.inf")
    return sum(abs(value) ** ord for value in cleaned) ** (1.0 / ord)


def psi_support(values: Sequence[float], *, threshold: float = 1e-3) -> int:
    """Return the TNFR PSI support count.

    Support counts how many PSI components exceed a reproducible activation
    threshold (strictly greater than ``threshold``) and therefore contribute to
    the synchrony budget.

    Parameters
    ----------
    values:
        Iterable containing PSI components.
    threshold:
        Absolute magnitude below which PSI modes are ignored.

    Examples
    --------
    >>> psi_support([0.0, 0.002, 0.2])
    2
    >>> psi_support([0.0, 1e-5, -3e-4], threshold=1e-4)
    1
    """

    if threshold < 0.0:
        threshold = abs(threshold)
    cutoff = max(0.0, float(threshold))
    count = 0
    for value in values:
        try:
            magnitude = abs(float(value))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(magnitude):
            continue
        if magnitude > cutoff:
            count += 1
    return count


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    if q <= 0.0:
        return sorted_values[0]
    if q >= 1.0:
        return sorted_values[-1]
    position = q * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    if lower_index == upper_index:
        return lower_value
    weight = position - lower_index
    return lower_value * (1.0 - weight) + upper_value * weight


def bifurcation_threshold(
    samples: Sequence[float], *, quantile: float = 0.9
) -> float:
    """Return the TNFR bifurcation threshold τ.

    The threshold highlights coherence excursions large enough to trigger a
    behavioural branch (driver correction or chassis response).  Using a high
    quantile keeps the estimate reproducible while still reacting to genuine
    peaks.

    Examples
    --------
    >>> bifurcation_threshold([0.2, 0.35, 0.4, 0.5])
    0.47
    """

    cleaned = sorted(
        float(value)
        for value in samples
        if math.isfinite(float(value))
    )
    if not cleaned:
        return 0.0
    q = max(0.0, min(1.0, float(quantile)))
    return _quantile(cleaned, q)


def mutation_threshold(
    samples: Sequence[float], *, quantile: float = 0.75
) -> float:
    """Return the TNFR mutation threshold ξ.

    Mutations represent smaller disturbances that still warrant attention.  We
    capture them via a mid-to-high quantile so the estimator is sensitive to
    frequently repeating perturbations.

    Examples
    --------
    >>> mutation_threshold([0.2, 0.35, 0.4, 0.5])
    0.425
    """

    cleaned = sorted(
        float(value)
        for value in samples
        if math.isfinite(float(value))
    )
    if not cleaned:
        return 0.0
    q = max(0.0, min(1.0, float(quantile)))
    return _quantile(cleaned, q)


def _aggregate_node_delta(
    records: Sequence[SupportsTelemetrySample],
    indices: Sequence[int] | None = None,
) -> dict[str, float]:
    """Return absolute ΔNFR contributions aggregated per node."""

    weights: defaultdict[str, float] = defaultdict(float)
    if indices is None:
        iterator: Iterable[int] = range(len(records))
    else:
        iterator = indices
    for index in iterator:
        if not 0 <= index < len(records):
            continue
        distribution = _epi.delta_nfr_by_node(records[index])
        for node, contribution in distribution.items():
            try:
                magnitude = abs(float(contribution))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(magnitude) or magnitude <= 0.0:
                continue
            key = str(node)
            weights[key] += magnitude
    return dict(weights)


def _phase_delta_distribution(
    records: Sequence[SupportsTelemetrySample],
    phase_windows: Mapping[str, Sequence[int]] | None,
) -> tuple[dict[str, float], float]:
    """Return the phase-normalised ΔNFR profile and its entropy."""

    if not phase_windows:
        return {}, 0.0
    phase_totals: dict[str, float] = {}
    for label, indices in phase_windows.items():
        weights = _aggregate_node_delta(records, indices)
        total = sum(weights.values())
        phase_totals[str(label)] = total
    if not phase_totals:
        return {}, 0.0
    positive_total = sum(value for value in phase_totals.values() if value > 0.0)
    if positive_total <= 0.0:
        return {key: 0.0 for key in phase_totals}, 0.0
    probabilities = {
        key: (value / positive_total) if value > 0.0 else 0.0
        for key, value in phase_totals.items()
    }
    entropy = normalised_entropy(probabilities.values())
    return probabilities, entropy


def _phase_node_entropy_map(
    records: Sequence[SupportsTelemetrySample],
    phase_windows: Mapping[str, Sequence[int]] | None,
) -> dict[str, float]:
    """Return the per-phase Shannon entropy of nodal ΔNFR distributions."""

    if not phase_windows:
        return {}
    entropy_map: dict[str, float] = {}
    for label, indices in phase_windows.items():
        weights = _aggregate_node_delta(records, indices)
        probabilities = _normalised_weights(weights)
        entropy_map[str(label)] = normalised_entropy(probabilities.values())
    return entropy_map


def phase_synchrony_index(lag: float, alignment: float) -> float:
    """Return a composite synchrony index combining phase lag and alignment.

    The index normalises the absolute phase lag to a [0, 1] range where ``1``
    represents perfect synchronisation (zero lag) and ``0`` corresponds to a
    π radian mismatch.  The alignment cosine, naturally bounded in [-1, 1], is
    translated to the same [0, 1] interval.  The final score favours alignment
    while preserving the sensitivity to lag, providing a stable indicator for
    desynchronisation alerts.
    """

    if not math.isfinite(lag):
        lag = 0.0
    if not math.isfinite(alignment):
        alignment = 1.0
    lag_score = 1.0 - min(abs(lag) / math.pi, 1.0)
    lag_score = max(0.0, min(1.0, lag_score))
    alignment_score = (alignment + 1.0) * 0.5
    alignment_score = max(0.0, min(1.0, alignment_score))
    composite = 0.6 * alignment_score + 0.4 * lag_score
    return max(0.0, min(1.0, composite))


@dataclass(frozen=True)
class SlideCatchBudget:
    """Composite steering margin metric derived from yaw and steer activity."""

    value: float = 0.0
    yaw_acceleration_ratio: float = 0.0
    steer_velocity_ratio: float = 0.0
    overshoot_ratio: float = 0.0


@dataclass(frozen=True)
class LockingWindowScore:
    """Aggregated stability score around throttle locking transitions."""

    value: float = 1.0
    on_throttle: float = 1.0
    off_throttle: float = 1.0
    transition_samples: int = 0


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
class AeroBalanceDriftBin:
    """Aggregate rake/μ metrics for a given aerodynamic speed band."""

    speed_min: float = 0.0
    speed_max: float | None = None
    samples: int = 0
    rake_mean: float = 0.0
    rake_std: float = 0.0
    mu_front_mean: float = 0.0
    mu_rear_mean: float = 0.0
    mu_delta: float = 0.0
    mu_ratio: float = 1.0
    mu_balance: float = 0.0
    mu_symmetry_front: float = 0.0
    mu_symmetry_rear: float = 0.0
    mu_balance_slope: float = 0.0
    mu_balance_sign_change: bool = False

    @property
    def rake_deg(self) -> float:
        """Return the mean rake expressed in degrees."""

        return math.degrees(self.rake_mean)


@dataclass(frozen=True)
class AeroBalanceDrift:
    """Aerodynamic balance drift derived from pitch, travel and μ usage."""

    low_speed: AeroBalanceDriftBin = field(default_factory=AeroBalanceDriftBin)
    medium_speed: AeroBalanceDriftBin = field(default_factory=AeroBalanceDriftBin)
    high_speed: AeroBalanceDriftBin = field(default_factory=AeroBalanceDriftBin)
    mu_tolerance: float = 0.04
    guidance: str = ""

    def dominant_bin(
        self, tolerance: float | None = None
    ) -> tuple[str, str, AeroBalanceDriftBin] | None:
        """Return the dominant drift bin exceeding ``tolerance`` in μ Δ."""

        effective_tolerance = self.mu_tolerance if tolerance is None else float(tolerance)
        if effective_tolerance < 0.0:
            effective_tolerance = 0.0
        for label, payload in (
            ("high", self.high_speed),
            ("medium", self.medium_speed),
            ("low", self.low_speed),
        ):
            if payload.samples <= 0:
                continue
            if abs(payload.mu_delta) <= effective_tolerance:
                continue
            direction = "front axle" if payload.mu_delta > 0.0 else "rear axle"
            return label, direction, payload
        return None


@dataclass(frozen=True)
class BrakeHeadroom:
    """Aggregated braking capacity metrics for the current window."""

    value: float = 0.0
    peak_decel: float = 0.0
    abs_activation_ratio: float = 0.0
    partial_locking_ratio: float = 0.0
    sustained_locking_ratio: float = 0.0
    fade_slope: float = 0.0
    fade_ratio: float = 0.0
    temperature_peak: float = 0.0
    temperature_mean: float = 0.0
    ventilation_alert: str = ""
    ventilation_index: float = 0.0
    temperature_available: bool = True
    fade_available: bool = True


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
class CPHIWheelComponents:
    """Normalised contributions to the Contact Patch Health Index."""

    temperature: float = 0.0
    gradient: float = 0.0
    mu: float = 0.0


@dataclass(frozen=True)
class CPHIThresholds:
    """Traffic-light thresholds for the Contact Patch Health Index.

    Values below ``red`` demand immediate intervention, measurements between
    ``red`` and ``amber`` indicate marginal tyre health, and readings equal
    or above ``green`` describe an optimal contact patch ready for push laps.
    """

    red: float = 0.62
    amber: float = 0.78
    green: float = 0.9

    def classify(self, value: float) -> str:
        if not math.isfinite(value):
            return "unknown"
        if value < self.red:
            return "red"
        if value < self.amber:
            return "amber"
        return "green"

    def is_optimal(self, value: float) -> bool:
        return math.isfinite(value) and value >= self.green


@dataclass(frozen=True)
class CPHIWheel:
    """Contact Patch Health Index components for a single wheel."""

    value: float = 1.0
    components: CPHIWheelComponents = field(default_factory=CPHIWheelComponents)
    temperature_delta: float = 0.0
    gradient_rate: float = 0.0

    @property
    def temperature_component(self) -> float:
        return self.components.temperature

    @property
    def gradient_component(self) -> float:
        return self.components.gradient

    @property
    def mu_component(self) -> float:
        return self.components.mu

    def as_dict(self, *, thresholds: CPHIThresholds | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "value": self.value,
            "components": {
                "temperature": self.components.temperature,
                "gradient": self.components.gradient,
                "mu": self.components.mu,
            },
            "temperature_delta": self.temperature_delta,
            "gradient_rate": self.gradient_rate,
        }
        if thresholds is not None:
            payload["status"] = thresholds.classify(self.value)
            payload["optimal"] = thresholds.is_optimal(self.value)
        return payload


@dataclass(frozen=True)
class CPHIReport(MappingABC[str, CPHIWheel]):
    """Aggregate CPHI values and expose shared thresholds.

    The mapping behaves like ``{suffix: CPHIWheel}`` while ``as_legacy_mapping``
    preserves the historical flat keys consumed by exporters and rule engines.
    """

    wheels: Mapping[str, CPHIWheel] = field(default_factory=dict)
    thresholds: CPHIThresholds = field(default_factory=CPHIThresholds)

    def __post_init__(self) -> None:
        object.__setattr__(self, "wheels", {str(key): value for key, value in self.wheels.items()})

    def __iter__(self) -> Iterator[str]:
        return iter(self.wheels)

    def __len__(self) -> int:
        return len(self.wheels)

    def __getitem__(self, key: str) -> CPHIWheel:
        return self.wheels[key]

    def classification(self, value: float) -> str:
        return self.thresholds.classify(value)

    def classification_for(self, suffix: str) -> str:
        wheel = self.wheels.get(suffix)
        return self.thresholds.classify(wheel.value) if wheel is not None else "unknown"

    def is_optimal(self, value: float) -> bool:
        return self.thresholds.is_optimal(value)

    def is_optimal_for(self, suffix: str) -> bool:
        wheel = self.wheels.get(suffix)
        return self.thresholds.is_optimal(wheel.value) if wheel is not None else False

    def as_dict(
        self, *, include_thresholds: bool = True, include_status: bool = True
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "wheels": {
                suffix: wheel.as_dict(
                    thresholds=self.thresholds if include_status else None
                )
                for suffix, wheel in self.wheels.items()
            }
        }
        if include_thresholds:
            payload["thresholds"] = {
                "red": self.thresholds.red,
                "amber": self.thresholds.amber,
                "green": self.thresholds.green,
            }
        return payload

    def as_legacy_mapping(self) -> dict[str, float]:
        warnings.warn(
            "CPHIReport.as_legacy_mapping() is deprecated and will be removed in a future release",
            DeprecationWarning,
            stacklevel=2,
        )
        legacy: dict[str, float] = {}
        for suffix, wheel in self.wheels.items():
            legacy[f"cphi_{suffix}"] = wheel.value
            legacy[f"cphi_{suffix}_temperature"] = wheel.temperature_component
            legacy[f"cphi_{suffix}_gradient"] = wheel.gradient_component
            legacy[f"cphi_{suffix}_mu"] = wheel.mu_component
            legacy[f"cphi_{suffix}_temp_delta"] = wheel.temperature_delta
            legacy[f"cphi_{suffix}_gradient_rate"] = wheel.gradient_rate
        return legacy


def _cphi_wheel_from_samples(
    samples: Sequence[tuple[float, float, float, float, float, float, float]]
) -> CPHIWheel:
    if not samples:
        sentinel = math.nan
        components = CPHIWheelComponents(
            temperature=sentinel,
            gradient=sentinel,
            mu=sentinel,
        )
        return CPHIWheel(
            value=sentinel,
            components=components,
            temperature_delta=sentinel,
            gradient_rate=sentinel,
        )

    slip_components: list[float] = []
    angle_components: list[float] = []
    mu_components: list[float] = []
    bias_components: list[float] = []
    bias_values: list[float] = []
    gradient_rates: list[float] = []

    prev_angle: float | None = None
    prev_time: float | None = None

    for slip_ratio, slip_angle, lat_force, long_force, load, bias, timestamp in samples:
        if math.isfinite(slip_ratio):
            ratio_component = abs(slip_ratio) / max(_CPHI_SLIP_RATIO_REFERENCE, 1e-6)
            slip_components.append(min(1.0, ratio_component))
        if math.isfinite(slip_angle):
            angle_component = abs(slip_angle) / max(_CPHI_SLIP_ANGLE_REFERENCE, 1e-6)
            angle_components.append(min(1.0, angle_component))
        if (
            math.isfinite(lat_force)
            and math.isfinite(long_force)
            and math.isfinite(load)
            and abs(load) > 1e-6
        ):
            resultant = math.hypot(lat_force, long_force)
            mu_usage = resultant / abs(load)
            mu_components.append(min(1.0, mu_usage / max(_CPHI_MU_REFERENCE, 1e-6)))
        if math.isfinite(bias):
            bias_components.append(min(1.0, abs(bias) / max(_CPHI_LOAD_BIAS_REFERENCE, 1e-6)))
            bias_values.append(bias)
        if (
            math.isfinite(slip_angle)
            and prev_angle is not None
            and math.isfinite(timestamp)
            and prev_time is not None
        ):
            dt = timestamp - prev_time
            if dt > 1e-3:
                gradient = abs(slip_angle - prev_angle) / dt
                if math.isfinite(gradient):
                    gradient_rates.append(gradient)
        if math.isfinite(slip_angle):
            prev_angle = slip_angle
        if math.isfinite(timestamp):
            prev_time = timestamp

    has_slip_or_force_samples = bool(slip_components or angle_components or mu_components)
    if not has_slip_or_force_samples:
        sentinel = math.nan
        components = CPHIWheelComponents(
            temperature=sentinel,
            gradient=sentinel,
            mu=sentinel,
        )
        return CPHIWheel(
            value=sentinel,
            components=components,
            temperature_delta=sentinel,
            gradient_rate=sentinel,
        )

    def _average(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    temperature_component = _average(slip_components)
    gradient_component = _average(angle_components)
    mu_component = _average(mu_components)
    bias_component = _average(bias_components)
    gradient_rate = _average(gradient_rates)
    temperature_delta = _average(bias_values)

    combined = (
        0.4 * temperature_component
        + 0.3 * gradient_component
        + 0.2 * mu_component
        + 0.1 * bias_component
    )
    combined = max(0.0, min(1.0, combined))
    value = max(0.0, 1.0 - combined)

    components = CPHIWheelComponents(
        temperature=temperature_component,
        gradient=gradient_component,
        mu=mu_component,
    )
    return CPHIWheel(
        value=value,
        components=components,
        temperature_delta=temperature_delta,
        gradient_rate=gradient_rate,
    )


def _cphi_from_samples(
    samples: Sequence[Mapping[str, tuple[float, float, float, float, float, float, float]]],
) -> CPHIReport:
    per_wheel: dict[str, list[tuple[float, float, float, float, float, float, float]]] = {
        suffix: [] for suffix in _WHEEL_SUFFIXES
    }
    for sample in samples:
        for suffix in _WHEEL_SUFFIXES:
            payload = sample.get(suffix)
            if payload is None:
                continue
            per_wheel.setdefault(suffix, []).append(payload)
    wheels = {
        suffix: _cphi_wheel_from_samples(series)
        for suffix, series in per_wheel.items()
    }
    return CPHIReport(wheels=wheels)


@dataclass(frozen=True)
class SuspensionVelocityBands:
    """Distribution of suspension velocities split by direction and band."""

    compression_low_ratio: float = 0.0
    compression_medium_ratio: float = 0.0
    compression_high_ratio: float = 0.0
    rebound_low_ratio: float = 0.0
    rebound_medium_ratio: float = 0.0
    rebound_high_ratio: float = 0.0
    ar_index: float = 0.0

    @property
    def compression_high_speed_percentage(self) -> float:
        return self.compression_high_ratio * 100.0

    @property
    def rebound_high_speed_percentage(self) -> float:
        return self.rebound_high_ratio * 100.0


def _suspension_velocity_bands_from_series(
    series: Sequence[float],
    *,
    low_threshold: float,
    high_threshold: float,
) -> SuspensionVelocityBands:
    if not series:
        return SuspensionVelocityBands()

    def _ratios(values: Sequence[float]) -> tuple[float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0
        total = float(len(values))
        low_count = 0
        medium_count = 0
        high_count = 0
        for value in values:
            magnitude = abs(float(value))
            if magnitude < low_threshold:
                low_count += 1
            elif magnitude < high_threshold:
                medium_count += 1
            else:
                high_count += 1
        return low_count / total, medium_count / total, high_count / total

    compression_samples = [float(value) for value in series if float(value) >= 0.0]
    rebound_samples = [float(value) for value in series if float(value) < 0.0]

    comp_low, comp_med, comp_high = _ratios(compression_samples)
    reb_low, reb_med, reb_high = _ratios(rebound_samples)

    compression_mean = (
        mean(abs(sample) for sample in compression_samples) if compression_samples else 0.0
    )
    rebound_mean = (
        mean(abs(sample) for sample in rebound_samples) if rebound_samples else 0.0
    )
    denominator = rebound_mean if rebound_mean > 1e-9 else 1e-9
    ar_index = compression_mean / denominator if denominator > 0.0 else 0.0

    if not math.isfinite(ar_index):
        ar_index = 0.0

    return SuspensionVelocityBands(
        compression_low_ratio=max(0.0, min(1.0, comp_low)),
        compression_medium_ratio=max(0.0, min(1.0, comp_med)),
        compression_high_ratio=max(0.0, min(1.0, comp_high)),
        rebound_low_ratio=max(0.0, min(1.0, reb_low)),
        rebound_medium_ratio=max(0.0, min(1.0, reb_med)),
        rebound_high_ratio=max(0.0, min(1.0, reb_high)),
        ar_index=max(0.0, min(10.0, ar_index)),
    )


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
    envelope for each axle.  ``phase_synchrony_index`` blends the normalised
    phase lag with the phase alignment cosine to produce a stable, unitless
    indicator for desynchronisation events.
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
    phase_synchrony_index: float
    motor_latency_ms: float
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
    phase_motor_latency_ms: Mapping[str, float] = field(default_factory=dict)
    mu_balance: float = 0.0
    mu_symmetry: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    delta_nfr_std: float = 0.0
    nodal_delta_nfr_std: float = 0.0
    delta_nfr_entropy: float = 0.0
    node_entropy: float = 0.0
    exit_gear_match: float = 0.0
    shift_stability: float = 0.0
    frequency_label: str = ""
    locking_window_score: LockingWindowScore = field(default_factory=LockingWindowScore)
    aero_coherence: AeroCoherence = field(default_factory=AeroCoherence)
    aero_mechanical_coherence: float = 0.0
    epi_derivative_abs: float = 0.0
    brake_headroom: BrakeHeadroom = field(default_factory=BrakeHeadroom)
    bumpstop_histogram: BumpstopHistogram = field(default_factory=BumpstopHistogram)
    cphi: CPHIReport = field(default_factory=CPHIReport)
    phase_cphi: Mapping[str, CPHIReport] = field(default_factory=dict)
    suspension_velocity_front: SuspensionVelocityBands = field(
        default_factory=SuspensionVelocityBands
    )
    suspension_velocity_rear: SuspensionVelocityBands = field(
        default_factory=SuspensionVelocityBands
    )
    aero_balance_drift: AeroBalanceDrift = field(default_factory=AeroBalanceDrift)
    phase_delta_nfr_std: Mapping[str, float] = field(default_factory=dict)
    phase_nodal_delta_nfr_std: Mapping[str, float] = field(default_factory=dict)
    phase_delta_nfr_entropy: Mapping[str, float] = field(default_factory=dict)
    phase_node_entropy: Mapping[str, float] = field(default_factory=dict)
    brake_longitudinal_correlation: float = 0.0
    throttle_longitudinal_correlation: float = 0.0
    phase_brake_longitudinal_correlation: Mapping[str, float] = field(
        default_factory=dict
    )
    phase_throttle_longitudinal_correlation: Mapping[str, float] = field(
        default_factory=dict
    )


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
    records: Sequence[SupportsTelemetrySample],
    *,
    phase_indices: Sequence[int] | Mapping[str, Sequence[int]] | None = None,
    bundles: Sequence[SupportsEPIBundle] | None = None,
    fallback_to_chronological: bool = True,
    objectives: object | None = None,
) -> WindowMetrics:
    """Return averaged plan metrics for a telemetry window.

    Parameters
    ----------
    records:
        Ordered window of telemetry samples implementing
        :class:`~tnfr_core.runtime.shared.SupportsTelemetrySample`. Entries
        must also satisfy :class:`~tnfr_core.runtime.shared.SupportsContextRecord`
        when contextual weighting is applied.
    bundles:
        Optional precomputed insight series implementing
        :class:`~tnfr_core.runtime.shared.SupportsEPIBundle` and matching
        ``records``. Each bundle must adhere to
        :class:`~tnfr_core.runtime.shared.SupportsContextBundle` so the node
        metrics remain accessible to the contextual helpers.
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
            phase_synchrony_index=1.0,
            motor_latency_ms=0.0,
            phase_motor_latency_ms={},
            useful_dissonance_ratio=0.0,
            useful_dissonance_percentage=0.0,
            coherence_index=0.0,
            ackermann_parallel_index=0.0,
            slide_catch_budget=SlideCatchBudget(),
            locking_window_score=LockingWindowScore(),
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
            mu_balance=0.0,
            mu_symmetry={},
            delta_nfr_std=0.0,
            nodal_delta_nfr_std=0.0,
            delta_nfr_entropy=0.0,
            node_entropy=0.0,
            exit_gear_match=0.0,
            shift_stability=0.0,
            frequency_label="",
            aero_coherence=AeroCoherence(),
            aero_mechanical_coherence=0.0,
            epi_derivative_abs=0.0,
            brake_headroom=BrakeHeadroom(),
            cphi=CPHIReport(),
            phase_cphi={},
            aero_balance_drift=AeroBalanceDrift(),
            phase_delta_nfr_std={},
            phase_nodal_delta_nfr_std={},
            phase_delta_nfr_entropy={},
            phase_node_entropy={},
            brake_longitudinal_correlation=0.0,
            throttle_longitudinal_correlation=0.0,
            phase_brake_longitudinal_correlation={},
            phase_throttle_longitudinal_correlation={},
        )

    if records and not isinstance(records[0], SupportsContextRecord):
        raise TypeError("records must expose lateral/vertical/longitudinal signals")
    if bundles and not isinstance(bundles[0], SupportsContextBundle):
        raise TypeError("bundles must expose chassis, tyres and transmission nodes")

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

    wheel_samples: list[
        dict[str, tuple[float, float, float, float, float, float, float]]
    ] = []

    def _compute_brake_headroom(
        samples: Sequence[SupportsTelemetrySample],
    ) -> BrakeHeadroom:
        decel_values: list[float] = []
        locking_values: list[float] = []
        partial_samples: list[float] = []
        severe_samples: list[float] = []
        temperature_samples: list[float] = []
        segment_temperatures: list[float] = []
        temperature_peak = 0.0
        temperature_available = False
        fade_slopes: list[float] = []
        fade_ratios: list[float] = []
        current_segment: list[tuple[float, float, float, float]] = []
        last_timestamp: float | None = None

        def _flush_segment() -> None:
            nonlocal current_segment
            if len(current_segment) < 3:
                current_segment.clear()
                return
            times = [entry[0] for entry in current_segment]
            duration = times[-1] - times[0]
            if duration < _FADE_MIN_DURATION:
                current_segment.clear()
                return
            pressures = [entry[1] for entry in current_segment]
            if max(pressures) - min(pressures) > _FADE_PRESSURE_VARIATION:
                current_segment.clear()
                return
            decels = [entry[2] for entry in current_segment]
            start_decel = decels[0]
            end_decel = decels[-1]
            if start_decel <= _FADE_MIN_START_DECEL:
                current_segment.clear()
                return
            drop = max(0.0, start_decel - end_decel)
            if drop <= 0.0:
                current_segment.clear()
                return
            ratio = drop / max(start_decel, 1e-6)
            if ratio < _FADE_MIN_DROP_RATIO:
                current_segment.clear()
                return
            slope = drop / max(duration, 1e-6)
            fade_slopes.append(slope)
            fade_ratios.append(ratio)
            temps = [entry[3] for entry in current_segment if math.isfinite(entry[3]) and entry[3] > 0.0]
            if temps:
                segment_temperatures.append(max(temps))
            current_segment.clear()

        for record in samples:
            try:
                long_accel = float(getattr(record, "longitudinal_accel", 0.0))
            except (TypeError, ValueError):
                long_accel = 0.0
            if math.isfinite(long_accel):
                decel = max(0.0, -long_accel)
                decel_values.append(decel)
            else:
                decel = 0.0
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
            brake_temps: list[float] = []
            for suffix in _WHEEL_SUFFIXES:
                attr = f"brake_temp_{suffix}"
                try:
                    temp_value = float(getattr(record, attr))
                except (TypeError, ValueError, AttributeError):
                    continue
                if not math.isfinite(temp_value) or temp_value <= 0.0:
                    continue
                brake_temps.append(temp_value)
            avg_temp = 0.0
            if brake_temps:
                avg_temp = sum(brake_temps) / len(brake_temps)
                temperature_samples.append(avg_temp)
                temperature_peak = max(temperature_peak, max(brake_temps))
                temperature_available = True
            try:
                timestamp = float(getattr(record, "timestamp", 0.0))
            except (TypeError, ValueError):
                timestamp = 0.0
            if not math.isfinite(timestamp):
                timestamp = 0.0
            try:
                pressure = float(getattr(record, "brake_pressure", 0.0))
            except (TypeError, ValueError):
                pressure = 0.0
            if not math.isfinite(pressure):
                pressure = 0.0
            pressure = max(0.0, min(1.0, pressure))
            if pressure >= _FADE_PRESSURE_THRESHOLD:
                if (
                    current_segment
                    and last_timestamp is not None
                    and timestamp - last_timestamp > _FADE_MAX_GAP
                ):
                    _flush_segment()
                current_segment.append((timestamp, pressure, decel, avg_temp))
            elif current_segment:
                _flush_segment()
            last_timestamp = timestamp
        if current_segment:
            _flush_segment()
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
        if segment_temperatures:
            temperature_available = True
        if temperature_available:
            temperature_mean = mean(temperature_samples) if temperature_samples else 0.0
            segment_peak = max(segment_temperatures) if segment_temperatures else 0.0
            temperature_peak = max(temperature_peak, segment_peak)
        else:
            temperature_mean = math.nan
            temperature_peak = math.nan
        fade_available = temperature_available
        if fade_available:
            fade_slope = max(fade_slopes) if fade_slopes else 0.0
            fade_ratio = max(fade_ratios) if fade_ratios else 0.0
        else:
            fade_slope = math.nan
            fade_ratio = math.nan
        ventilation_alert = ""
        if temperature_available:
            ventilation_index = 0.0
            temp_peak_value = (
                temperature_peak if math.isfinite(temperature_peak) else 0.0
            )
            fade_ratio_value = (
                fade_ratio if fade_available and math.isfinite(fade_ratio) else 0.0
            )
            fade_slope_value = (
                fade_slope if fade_available and math.isfinite(fade_slope) else 0.0
            )
            if (
                temp_peak_value > 0.0
                or fade_ratio_value > 0.0
                or fade_slope_value > 0.0
            ):
                temp_component = 0.0
                if temp_peak_value > _VENT_TEMP_WARNING:
                    temp_component = (temp_peak_value - _VENT_TEMP_WARNING) / max(
                        _VENT_TEMP_CRITICAL - _VENT_TEMP_WARNING, 1e-6
                    )
                ratio_component = fade_ratio_value / max(_FADE_RATIO_CRITICAL, 1e-6)
                slope_component = fade_slope_value / max(_FADE_SLOPE_CRITICAL, 1e-6)
                ventilation_index = max(
                    temp_component, ratio_component, slope_component
                )
                ventilation_index = max(0.0, min(1.0, ventilation_index))
                if (
                    temp_peak_value >= _VENT_TEMP_CRITICAL
                    or fade_ratio_value >= _FADE_RATIO_CRITICAL
                    or fade_slope_value >= _FADE_SLOPE_CRITICAL
                    or ventilation_index >= 0.9
                ):
                    ventilation_alert = "critical"
                elif (
                    temp_peak_value >= _VENT_TEMP_WARNING
                    or fade_ratio_value >= _FADE_RATIO_WARNING
                    or ventilation_index >= 0.45
                ):
                    ventilation_alert = "attention"
        else:
            ventilation_index = math.nan
        value = (1.0 - normalized_peak) * (1.0 - stress)
        fade_penalty = 0.0
        if fade_available and math.isfinite(fade_ratio):
            fade_penalty = min(0.6, max(0.0, fade_ratio))
        value *= max(0.0, 1.0 - fade_penalty)
        if temperature_available and math.isfinite(ventilation_index) and ventilation_index > 0.0:
            value *= max(0.0, 1.0 - 0.5 * ventilation_index)
        value = max(0.0, min(1.0, value))
        return BrakeHeadroom(
            value=value,
            peak_decel=peak_decel,
            abs_activation_ratio=abs_activation,
            partial_locking_ratio=partial_locking,
            sustained_locking_ratio=sustained_ratio,
            fade_slope=fade_slope,
            fade_ratio=fade_ratio,
            temperature_peak=temperature_peak,
            temperature_mean=temperature_mean,
            ventilation_alert=ventilation_alert,
            ventilation_index=ventilation_index,
            temperature_available=temperature_available,
            fade_available=fade_available,
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
    front_velocity_series: list[float] = []
    rear_velocity_series: list[float] = []
    front_mu_lat_series: list[float] = []
    front_mu_long_series: list[float] = []
    rear_mu_lat_series: list[float] = []
    rear_mu_long_series: list[float] = []
    throttle_series: list[float] = []
    locking_series: list[float] = []
    brake_series: list[float] = []
    longitudinal_accel_series: list[float] = []
    for index, record in enumerate(records):
        try:
            throttle_value = float(getattr(record, "throttle", 0.0))
        except (TypeError, ValueError):
            throttle_value = 0.0
        if not math.isfinite(throttle_value):
            throttle_value = 0.0
        throttle_series.append(max(0.0, min(1.0, throttle_value)))
        try:
            locking_value = float(getattr(record, "locking", 0.0))
        except (TypeError, ValueError):
            locking_value = 0.0
        if not math.isfinite(locking_value):
            locking_value = 0.0
        locking_series.append(max(0.0, min(1.0, locking_value)))
        try:
            brake_value = float(getattr(record, "brake_pressure", 0.0))
        except (TypeError, ValueError):
            brake_value = 0.0
        if not math.isfinite(brake_value):
            brake_value = 0.0
        brake_series.append(max(0.0, min(1.0, brake_value)))
        try:
            accel_value = float(getattr(record, "longitudinal_accel", 0.0))
        except (TypeError, ValueError):
            accel_value = 0.0
        if not math.isfinite(accel_value):
            accel_value = 0.0
        longitudinal_accel_series.append(accel_value)

        per_wheel_payload: dict[str, tuple[float, float, float, float, float, float, float]] = {}
        wheel_loads: dict[str, float | None] = {}
        for suffix in _WHEEL_SUFFIXES:
            attr = f"wheel_load_{suffix}"
            wheel_loads[suffix] = _safe_numeric(getattr(record, attr, None))
        front_mean = _mean_optional([wheel_loads.get("fl"), wheel_loads.get("fr")])
        rear_mean = _mean_optional([wheel_loads.get("rl"), wheel_loads.get("rr")])
        timestamp_value = _safe_numeric(getattr(record, "timestamp", index))
        if timestamp_value is None:
            timestamp_value = float(index)
        for suffix in _WHEEL_SUFFIXES:
            slip_ratio = _safe_numeric(getattr(record, f"slip_ratio_{suffix}", None))
            if slip_ratio is None:
                slip_ratio = _safe_numeric(getattr(record, "slip_ratio", 0.0))
            slip_angle = _safe_numeric(getattr(record, f"slip_angle_{suffix}", None))
            if slip_angle is None:
                slip_angle = _safe_numeric(getattr(record, "slip_angle", 0.0))
            lateral_force = _safe_numeric(getattr(record, f"wheel_lateral_force_{suffix}", None))
            longitudinal_force = _safe_numeric(
                getattr(record, f"wheel_longitudinal_force_{suffix}", None)
            )
            load = wheel_loads.get(suffix)
            if load is None:
                load = 0.0
            axle_mean = front_mean if suffix in {"fl", "fr"} else rear_mean
            if axle_mean is None or axle_mean <= 1e-6:
                bias = 0.0
            else:
                bias = (load - axle_mean) / axle_mean
            payload = (
                slip_ratio if slip_ratio is not None else float("nan"),
                slip_angle if slip_angle is not None else float("nan"),
                lateral_force if lateral_force is not None else float("nan"),
                longitudinal_force if longitudinal_force is not None else float("nan"),
                float(load),
                bias,
                float(timestamp_value),
            )
            per_wheel_payload[suffix] = payload
        wheel_samples.append(per_wheel_payload)

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
    synchrony = phase_synchrony_index(lag, alignment)

    correlations = motor_input_correlations(selected)

    def _select_correlation(
        mapping: Mapping[str | Tuple[str, str], PhaseCorrelation] | None,
    ) -> PhaseCorrelation | None:
        best: PhaseCorrelation | None = None
        if not mapping:
            return None
        for payload in mapping.values():
            if not isinstance(payload, PhaseCorrelation):
                continue
            if best is None or payload.magnitude > best.magnitude:
                best = payload
        return best

    best_correlation = _select_correlation(correlations)
    motor_latency_ms = (
        abs(best_correlation.latency_ms)
        if best_correlation is not None
        else abs(phase_to_latency_ms(freq, lag))
    )

    phase_latency_map: dict[str, float] = {}
    for phase_label, indices in phase_windows.items():
        subset = [
            records[index]
            for index in indices
            if 0 <= index < len(records)
        ]
        if len(subset) < 3:
            continue
        phase_correlations = motor_input_correlations(subset)
        phase_best = _select_correlation(phase_correlations)
        if phase_best is not None:
            phase_latency_map[str(phase_label)] = abs(phase_best.latency_ms)
            continue
        phase_freq, phase_lag, _ = phase_alignment(subset)
        if phase_freq > 0.0:
            phase_latency_map[str(phase_label)] = abs(
                phase_to_latency_ms(phase_freq, phase_lag)
            )

    phase_latency_map = replicate_phase_aliases(phase_latency_map)

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
        tyre_nodes: list[SupportsTyresNode] = [bundle.tyres for bundle in bundles]
        suspension_nodes: list[SupportsSuspensionNode] = [
            bundle.suspension for bundle in bundles
        ]
        chassis_nodes: list[SupportsChassisNode] = [bundle.chassis for bundle in bundles]
        driver_nodes: list[SupportsDriverNode] = [bundle.driver for bundle in bundles]

        yaw_rates = [node.yaw_rate for node in chassis_nodes]
        steer_series = [float(node.steer) for node in driver_nodes]
        epi_values = [abs(float(bundle.dEPI_dt)) for bundle in bundles]
        if epi_values:
            epi_abs_derivative = mean(epi_values)
        support_samples = [
            max(0.0, float(tyre.delta_nfr)) + max(0.0, float(suspension.delta_nfr))
            for tyre, suspension in zip(tyre_nodes, suspension_nodes)
        ]
        suspension_series = [float(node.delta_nfr) for node in suspension_nodes]
        tyre_series = [float(node.delta_nfr) for node in tyre_nodes]
        front_travel_series = [
            float(node.travel_front) for node in suspension_nodes
        ]
        rear_travel_series = [
            float(node.travel_rear) for node in suspension_nodes
        ]
        front_velocity_series = [
            float(node.velocity_front) for node in suspension_nodes
        ]
        rear_velocity_series = [
            float(node.velocity_rear) for node in suspension_nodes
        ]
        front_mu_lat_series = [
            float(node.mu_eff_front_lateral) for node in tyre_nodes
        ]
        front_mu_long_series = [
            float(node.mu_eff_front_longitudinal) for node in tyre_nodes
        ]
        rear_mu_lat_series = [
            float(node.mu_eff_rear_lateral) for node in tyre_nodes
        ]
        rear_mu_long_series = [
            float(node.mu_eff_rear_longitudinal) for node in tyre_nodes
        ]
        longitudinal_series = [
            float(bundle.delta_nfr_proj_longitudinal) for bundle in bundles
        ]
        lateral_series = [
            float(bundle.delta_nfr_proj_lateral) for bundle in bundles
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
            float(getattr(record, "delta_nfr_proj_longitudinal", 0.0)) for record in records
        ]
        lateral_series = [
            float(getattr(record, "delta_nfr_proj_lateral", 0.0)) for record in records
        ]
        front_travel_series = [
            float(getattr(record, "suspension_travel_front", 0.0)) for record in records
        ]
        rear_travel_series = [
            float(getattr(record, "suspension_travel_rear", 0.0)) for record in records
        ]
        front_velocity_series = [
            float(getattr(record, "suspension_velocity_front", 0.0)) for record in records
        ]
        rear_velocity_series = [
            float(getattr(record, "suspension_velocity_rear", 0.0)) for record in records
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

    def _standard_deviation(values: Sequence[float]) -> float:
        if len(values) < 2:
            return 0.0
        clean = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        if len(clean) < 2:
            return 0.0
        return pstdev(clean)

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

    def _pearson_correlation(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b) or len(a) < 2:
            return 0.0
        mean_a = mean(a)
        mean_b = mean(b)
        num = 0.0
        sum_sq_a = 0.0
        sum_sq_b = 0.0
        for value_a, value_b in zip(a, b):
            da = float(value_a) - mean_a
            db = float(value_b) - mean_b
            num += da * db
            sum_sq_a += da * da
            sum_sq_b += db * db
        denom = math.sqrt(sum_sq_a * sum_sq_b)
        if denom <= 1e-12:
            return 0.0
        coeff = num / denom
        if coeff > 1.0:
            return 1.0
        if coeff < -1.0:
            return -1.0
        return coeff

    def _normalise_series(values: Sequence[float]) -> list[float]:
        cleaned: list[float] = []
        minimum = float("inf")
        maximum = float("-inf")
        for raw in values:
            try:
                numeric = float(raw)
            except (TypeError, ValueError):
                numeric = 0.0
            if not math.isfinite(numeric):
                numeric = 0.0
            cleaned.append(numeric)
            if numeric < minimum:
                minimum = numeric
            if numeric > maximum:
                maximum = numeric
        if not cleaned:
            return []
        span = maximum - minimum
        if not math.isfinite(span) or span <= 1e-12:
            return [0.0 for _ in cleaned]
        scale = 1.0 / span
        return [(value - minimum) * scale for value in cleaned]

    def _select_series(series: Sequence[float], indices: Sequence[int] | None) -> list[float]:
        if not indices:
            return [float(value) for value in series]
        selected: list[float] = []
        for index in indices:
            if 0 <= index < len(series):
                try:
                    value = float(series[index])
                except (TypeError, ValueError):
                    value = 0.0
                if not math.isfinite(value):
                    value = 0.0
                selected.append(value)
        return selected

    def _phase_correlation_map(
        series_a: Sequence[float], series_b: Sequence[float]
    ) -> dict[str, float]:
        if not phase_windows:
            return {}
        phase_map: dict[str, float] = {}
        for phase_label, indices in phase_windows.items():
            samples_a = _select_series(series_a, indices)
            samples_b = _select_series(series_b, indices)
            if len(samples_a) != len(samples_b):
                length = min(len(samples_a), len(samples_b))
                samples_a = samples_a[:length]
                samples_b = samples_b[:length]
            phase_map[str(phase_label)] = _pearson_correlation(samples_a, samples_b)
        return phase_map

    def _phase_standard_deviation_map(
        series: Sequence[float],
    ) -> dict[str, float]:
        if not phase_windows:
            return {}
        phase_map: dict[str, float] = {}
        for phase_label, indices in phase_windows.items():
            samples = _select_series(series, indices)
            phase_map[phase_label] = _standard_deviation(samples)
        return phase_map

    delta_std = _standard_deviation(delta_series)
    nodal_std = _standard_deviation(support_samples)
    phase_delta_std_map = replicate_phase_aliases(
        _phase_standard_deviation_map(delta_series)
    )
    phase_nodal_std_map = replicate_phase_aliases(
        _phase_standard_deviation_map(support_samples)
    )
    phase_delta_profile, phase_entropy_value = _phase_delta_distribution(
        records,
        phase_windows,
    )
    phase_delta_entropy_map = replicate_phase_aliases(phase_delta_profile)
    phase_node_entropy_map = replicate_phase_aliases(
        _phase_node_entropy_map(records, phase_windows)
    )
    node_profile = _normalised_weights(_aggregate_node_delta(records))
    node_entropy_value = normalised_entropy(node_profile.values())

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

        def _gear_ratio(record: SupportsTelemetrySample) -> float | None:
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

    def _mean_subset(series: Sequence[float], indices: Sequence[int] | None) -> float:
        if indices is None:
            values = series
        else:
            values = [series[index] for index in indices if 0 <= index < len(series)]
        cleaned = [float(value) for value in values if math.isfinite(float(value))]
        return mean(cleaned) if cleaned else 0.0

    def _axis_means(indices: Sequence[int] | None) -> tuple[float, float, float, float]:
        return (
            _mean_subset(front_mu_lat_series, indices),
            _mean_subset(front_mu_long_series, indices),
            _mean_subset(rear_mu_lat_series, indices),
            _mean_subset(rear_mu_long_series, indices),
        )

    def _normalised_symmetry(lat_mean: float, long_mean: float) -> float:
        denominator = abs(lat_mean) + abs(long_mean)
        if denominator <= 1e-9:
            return 0.0
        return (lat_mean - long_mean) / denominator

    def _normalised_balance(
        front_lat_mean: float,
        front_long_mean: float,
        rear_lat_mean: float,
        rear_long_mean: float,
    ) -> float:
        front_total = front_lat_mean + front_long_mean
        rear_total = rear_lat_mean + rear_long_mean
        denominator = abs(front_total) + abs(rear_total)
        if denominator <= 1e-9:
            return 0.0
        return (front_total - rear_total) / denominator

    window_front_lat, window_front_long, window_rear_lat, window_rear_long = _axis_means(None)
    mu_balance = _normalised_balance(
        window_front_lat,
        window_front_long,
        window_rear_lat,
        window_rear_long,
    )
    mu_symmetry: dict[str, dict[str, float]] = {
        "window": {
            "front": _normalised_symmetry(window_front_lat, window_front_long),
            "rear": _normalised_symmetry(window_rear_lat, window_rear_long),
        }
    }
    for phase_label, indices in phase_windows.items():
        front_lat_mean, front_long_mean, rear_lat_mean, rear_long_mean = _axis_means(indices)
        mu_symmetry[phase_label] = {
            "front": _normalised_symmetry(front_lat_mean, front_long_mean),
            "rear": _normalised_symmetry(rear_lat_mean, rear_long_mean),
        }

    drift_low_threshold = max(
        0.0, _objective("aero_drift_low_speed_threshold", 35.0)
    )
    drift_high_candidate = max(
        0.0, _objective("aero_drift_high_speed_threshold", 50.0)
    )
    if drift_high_candidate <= drift_low_threshold:
        drift_high_threshold = drift_low_threshold + max(
            1.0, drift_low_threshold * 0.1 + 1e-6
        )
    else:
        drift_high_threshold = drift_high_candidate
    drift_mu_tolerance = max(
        0.0, _objective("aero_drift_mu_tolerance", 0.04)
    )
    drift_bins: dict[str, dict[str, list[float]]] = {
        "low": {
            "rake": [],
            "mu_front": [],
            "mu_rear": [],
            "mu_front_lat": [],
            "mu_front_long": [],
            "mu_rear_lat": [],
            "mu_rear_long": [],
            "mu_balance_samples": [],
        },
        "medium": {
            "rake": [],
            "mu_front": [],
            "mu_rear": [],
            "mu_front_lat": [],
            "mu_front_long": [],
            "mu_rear_lat": [],
            "mu_rear_long": [],
            "mu_balance_samples": [],
        },
        "high": {
            "rake": [],
            "mu_front": [],
            "mu_rear": [],
            "mu_front_lat": [],
            "mu_front_long": [],
            "mu_rear_lat": [],
            "mu_rear_long": [],
            "mu_balance_samples": [],
        },
    }

    def _drift_bin_key(speed_value: float) -> str:
        if speed_value <= drift_low_threshold:
            return "low"
        if speed_value <= drift_high_threshold:
            return "medium"
        return "high"

    for record in records:
        try:
            speed_value = float(getattr(record, "speed", 0.0))
        except (TypeError, ValueError):
            speed_value = 0.0
        if not math.isfinite(speed_value):
            speed_value = 0.0
        bin_key = _drift_bin_key(speed_value)
        try:
            pitch_value = float(getattr(record, "pitch", 0.0))
        except (TypeError, ValueError):
            pitch_value = 0.0
        if not math.isfinite(pitch_value):
            pitch_value = 0.0
        try:
            front_travel = float(getattr(record, "suspension_travel_front", 0.0))
        except (TypeError, ValueError):
            front_travel = 0.0
        if not math.isfinite(front_travel):
            front_travel = 0.0
        try:
            rear_travel = float(getattr(record, "suspension_travel_rear", 0.0))
        except (TypeError, ValueError):
            rear_travel = 0.0
        if not math.isfinite(rear_travel):
            rear_travel = 0.0
        travel_delta = rear_travel - front_travel
        if abs(front_travel) > 1e-9:
            travel_ratio = rear_travel / front_travel
        elif abs(rear_travel) > 1e-9:
            travel_ratio = math.copysign(10.0, rear_travel)
        else:
            travel_ratio = 1.0
        if not math.isfinite(travel_ratio):
            travel_ratio = 1.0
        travel_ratio = max(-10.0, min(10.0, travel_ratio))
        rake_correction = math.atan2(travel_delta, travel_ratio)
        rake_value = pitch_value + rake_correction
        try:
            mu_front = float(getattr(record, "mu_eff_front", 0.0))
        except (TypeError, ValueError):
            mu_front = 0.0
        if not math.isfinite(mu_front):
            mu_front = 0.0
        try:
            mu_rear = float(getattr(record, "mu_eff_rear", 0.0))
        except (TypeError, ValueError):
            mu_rear = 0.0
        if not math.isfinite(mu_rear):
            mu_rear = 0.0
        try:
            mu_front_lat = float(getattr(record, "mu_eff_front_lateral", 0.0))
        except (TypeError, ValueError):
            mu_front_lat = 0.0
        if not math.isfinite(mu_front_lat):
            mu_front_lat = 0.0
        try:
            mu_front_long = float(getattr(record, "mu_eff_front_longitudinal", 0.0))
        except (TypeError, ValueError):
            mu_front_long = 0.0
        if not math.isfinite(mu_front_long):
            mu_front_long = 0.0
        try:
            mu_rear_lat = float(getattr(record, "mu_eff_rear_lateral", 0.0))
        except (TypeError, ValueError):
            mu_rear_lat = 0.0
        if not math.isfinite(mu_rear_lat):
            mu_rear_lat = 0.0
        try:
            mu_rear_long = float(getattr(record, "mu_eff_rear_longitudinal", 0.0))
        except (TypeError, ValueError):
            mu_rear_long = 0.0
        if not math.isfinite(mu_rear_long):
            mu_rear_long = 0.0
        bin_payload = drift_bins[bin_key]
        if math.isfinite(rake_value):
            front_total_sample = mu_front_lat + mu_front_long
            rear_total_sample = mu_rear_lat + mu_rear_long
            balance_denominator = abs(front_total_sample) + abs(rear_total_sample)
            if balance_denominator <= 1e-9:
                mu_balance_sample = 0.0
            else:
                mu_balance_sample = (front_total_sample - rear_total_sample) / balance_denominator
            if not math.isfinite(mu_balance_sample):
                mu_balance_sample = 0.0
            bin_payload["rake"].append(rake_value)
            bin_payload["mu_front"].append(mu_front)
            bin_payload["mu_rear"].append(mu_rear)
            bin_payload["mu_front_lat"].append(mu_front_lat)
            bin_payload["mu_front_long"].append(mu_front_long)
            bin_payload["mu_rear_lat"].append(mu_rear_lat)
            bin_payload["mu_rear_long"].append(mu_rear_long)
            bin_payload["mu_balance_samples"].append(mu_balance_sample)

    def _build_drift_bin(
        key: str, lower: float, upper: float | None
    ) -> AeroBalanceDriftBin:
        payload = drift_bins[key]
        rakes = payload["rake"]
        mu_front_values = payload["mu_front"]
        mu_rear_values = payload["mu_rear"]
        mu_front_lat_values = payload["mu_front_lat"]
        mu_front_long_values = payload["mu_front_long"]
        mu_rear_lat_values = payload["mu_rear_lat"]
        mu_rear_long_values = payload["mu_rear_long"]
        mu_balance_samples = payload["mu_balance_samples"]
        samples = len(rakes)
        rake_mean = mean(rakes) if rakes else 0.0
        rake_std = math.sqrt(pvariance(rakes)) if len(rakes) >= 2 else 0.0
        mu_front_mean = mean(mu_front_values) if mu_front_values else 0.0
        mu_rear_mean = mean(mu_rear_values) if mu_rear_values else 0.0
        mu_delta_value = mu_front_mean - mu_rear_mean
        if abs(mu_rear_mean) > 1e-9:
            mu_ratio = mu_front_mean / mu_rear_mean
        else:
            mu_ratio = 1.0 if abs(mu_front_mean) <= 1e-9 else math.copysign(10.0, mu_front_mean)
        if not math.isfinite(mu_ratio):
            mu_ratio = 1.0
        mu_front_lat_mean = mean(mu_front_lat_values) if mu_front_lat_values else 0.0
        mu_front_long_mean = mean(mu_front_long_values) if mu_front_long_values else 0.0
        mu_rear_lat_mean = mean(mu_rear_lat_values) if mu_rear_lat_values else 0.0
        mu_rear_long_mean = mean(mu_rear_long_values) if mu_rear_long_values else 0.0
        front_total = mu_front_lat_mean + mu_front_long_mean
        rear_total = mu_rear_lat_mean + mu_rear_long_mean
        balance_denominator = abs(front_total) + abs(rear_total)
        mu_delta_total = front_total - rear_total
        if balance_denominator <= 1e-9:
            mu_balance = 0.0
        else:
            mu_balance = mu_delta_total / balance_denominator
        front_symmetry_denominator = abs(mu_front_lat_mean) + abs(mu_front_long_mean)
        if front_symmetry_denominator <= 1e-9:
            mu_symmetry_front = 0.0
        else:
            mu_symmetry_front = (mu_front_lat_mean - mu_front_long_mean) / front_symmetry_denominator
        rear_symmetry_denominator = abs(mu_rear_lat_mean) + abs(mu_rear_long_mean)
        if rear_symmetry_denominator <= 1e-9:
            mu_symmetry_rear = 0.0
        else:
            mu_symmetry_rear = (mu_rear_lat_mean - mu_rear_long_mean) / rear_symmetry_denominator
        mu_balance_slope = 0.0
        if len(rakes) >= 2 and len(mu_balance_samples) == len(rakes):
            rake_mean_value = mean(rakes)
            mu_balance_mean = mean(mu_balance_samples)
            rake_variance = sum((value - rake_mean_value) ** 2 for value in rakes)
            if rake_variance > 1e-12:
                covariance = sum(
                    (rake_value - rake_mean_value) * (mu_value - mu_balance_mean)
                    for rake_value, mu_value in zip(rakes, mu_balance_samples)
                )
                mu_balance_slope = covariance / rake_variance
        if not math.isfinite(mu_balance_slope):
            mu_balance_slope = 0.0
        mu_balance_sign_change = False
        if mu_balance_samples:
            positive_balance = any(value > 1e-6 for value in mu_balance_samples)
            negative_balance = any(value < -1e-6 for value in mu_balance_samples)
            mu_balance_sign_change = positive_balance and negative_balance
        return AeroBalanceDriftBin(
            speed_min=lower,
            speed_max=upper,
            samples=samples,
            rake_mean=rake_mean,
            rake_std=rake_std,
            mu_front_mean=mu_front_mean,
            mu_rear_mean=mu_rear_mean,
            mu_delta=mu_delta_value,
            mu_ratio=mu_ratio,
            mu_balance=mu_balance,
            mu_symmetry_front=mu_symmetry_front,
            mu_symmetry_rear=mu_symmetry_rear,
            mu_balance_slope=mu_balance_slope,
            mu_balance_sign_change=mu_balance_sign_change,
        )

    low_drift = _build_drift_bin("low", 0.0, drift_low_threshold)
    medium_drift = _build_drift_bin("medium", drift_low_threshold, drift_high_threshold)
    high_drift = _build_drift_bin("high", drift_high_threshold, None)
    drift_labels = {"low": "low", "medium": "medium", "high": "high"}
    drift_guidance = ""
    for key, payload in (("high", high_drift), ("medium", medium_drift), ("low", low_drift)):
        if payload.samples <= 0:
            continue
        if abs(payload.mu_delta) <= drift_mu_tolerance:
            continue
        direction = "front axle" if payload.mu_delta > 0.0 else "rear axle"
        drift_guidance = (
            f"{drift_labels[key]} μΔ {payload.mu_delta:+.2f} "
            f"rake {payload.rake_deg:+.2f}° load {direction}"
        )
        guidance_details: list[str] = []
        if abs(payload.mu_balance_slope) > 1e-6 and math.isfinite(payload.mu_balance_slope):
            guidance_details.append(f"μβ sensitivity {payload.mu_balance_slope:+.3f}/rad")
        if payload.mu_balance_sign_change:
            guidance_details.append("μβ sign change")
        if guidance_details:
            drift_guidance = f"{drift_guidance} ({'; '.join(guidance_details)})"
        break
    aero_balance_drift = AeroBalanceDrift(
        low_speed=low_drift,
        medium_speed=medium_drift,
        high_speed=high_drift,
        mu_tolerance=drift_mu_tolerance,
        guidance=drift_guidance,
    )

    velocity_low_threshold = max(
        0.0, _objective("suspension_velocity_low_threshold", _SUSPENSION_LOW_SPEED_THRESHOLD)
    )
    velocity_high_candidate = max(
        0.0, _objective("suspension_velocity_high_threshold", _SUSPENSION_HIGH_SPEED_THRESHOLD)
    )
    if velocity_high_candidate <= velocity_low_threshold:
        velocity_high_threshold = velocity_low_threshold + max(0.01, velocity_low_threshold * 0.5 + 1e-6)
    else:
        velocity_high_threshold = velocity_high_candidate

    front_velocity_profile = _suspension_velocity_bands_from_series(
        front_velocity_series,
        low_threshold=velocity_low_threshold,
        high_threshold=velocity_high_threshold,
    )
    rear_velocity_profile = _suspension_velocity_bands_from_series(
        rear_velocity_series,
        low_threshold=velocity_low_threshold,
        high_threshold=velocity_high_threshold,
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

    def _locking_window_score_from_series(
        throttle: Sequence[float],
        locking: Sequence[float],
        yaw: Sequence[float],
        longitudinal: Sequence[float],
    ) -> LockingWindowScore:
        sample_count = min(len(throttle), len(locking))
        if sample_count < 2:
            return LockingWindowScore()
        transitions: list[tuple[str, float]] = []
        yaw_count = len(yaw)
        long_count = len(longitudinal)
        for index in range(1, sample_count):
            prev = float(throttle[index - 1])
            curr = float(throttle[index])
            if not (math.isfinite(prev) and math.isfinite(curr)):
                continue
            delta = curr - prev
            transition: str | None = None
            if delta >= _LOCKING_TRANSITION_DELTA or (
                prev <= _LOCKING_TRANSITION_LOW and curr >= _LOCKING_TRANSITION_HIGH
            ):
                if delta > 0.0:
                    transition = "on"
            elif delta <= -_LOCKING_TRANSITION_DELTA or (
                prev >= _LOCKING_TRANSITION_HIGH and curr <= _LOCKING_TRANSITION_LOW
            ):
                if delta < 0.0:
                    transition = "off"
            if transition is None:
                continue
            lock_value = locking[index] if index < len(locking) else locking[-1]
            yaw_value = 0.0
            if yaw_count:
                yaw_index = index if index < yaw_count else yaw_count - 1
                yaw_value = float(yaw[yaw_index])
            long_value = 0.0
            if long_count:
                long_index = index if index < long_count else long_count - 1
                long_value = float(longitudinal[long_index])
            throttle_change = abs(delta)
            locking_component = max(0.0, min(1.0, float(lock_value)))
            yaw_component = _normalised_ratio(abs(yaw_value), _LOCKING_YAW_REFERENCE)
            longitudinal_component = _normalised_ratio(
                abs(long_value), _LOCKING_LONGITUDINAL_REFERENCE
            )
            throttle_component = _normalised_ratio(
                throttle_change, _LOCKING_THROTTLE_REFERENCE
            )
            penalty = (
                0.4 * locking_component
                + 0.3 * yaw_component
                + 0.2 * longitudinal_component
                + 0.1 * throttle_component
            )
            if penalty > 1.0:
                penalty = 1.0
            transitions.append((transition, penalty))
        if not transitions:
            return LockingWindowScore()

        def _score(values: Sequence[float]) -> float:
            if not values:
                return 1.0
            mean_penalty = sum(values) / len(values)
            if not math.isfinite(mean_penalty):
                mean_penalty = 1.0
            return max(0.0, 1.0 - mean_penalty)

        on_penalties = [value for label, value in transitions if label == "on"]
        off_penalties = [value for label, value in transitions if label == "off"]
        overall_penalties = [value for _, value in transitions]
        return LockingWindowScore(
            value=_score(overall_penalties),
            on_throttle=_score(on_penalties),
            off_throttle=_score(off_penalties),
            transition_samples=len(transitions),
        )

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

    locking_window_score = _locking_window_score_from_series(
        throttle_series,
        locking_series,
        yaw_rates,
        longitudinal_series,
    )

    normalised_brake = _normalise_series(brake_series)
    normalised_throttle = _normalise_series(throttle_series)
    normalised_accel = _normalise_series(longitudinal_accel_series)
    normalised_decel = _normalise_series([-value for value in longitudinal_accel_series])

    brake_longitudinal_correlation = _pearson_correlation(
        normalised_brake,
        normalised_decel,
    )
    throttle_longitudinal_correlation = _pearson_correlation(
        normalised_throttle,
        normalised_accel,
    )

    phase_brake_correlation_map = replicate_phase_aliases(
        _phase_correlation_map(normalised_brake, normalised_decel)
    )
    phase_throttle_correlation_map = replicate_phase_aliases(
        _phase_correlation_map(normalised_throttle, normalised_accel)
    )

    if wheel_samples:
        cphi_overall = _cphi_from_samples(wheel_samples)
        base_thresholds = cphi_overall.thresholds
    else:
        base_thresholds = CPHIThresholds()
        cphi_overall = CPHIReport(
            wheels={suffix: CPHIWheel() for suffix in _WHEEL_SUFFIXES},
            thresholds=base_thresholds,
        )
    phase_cphi: dict[str, CPHIReport] = {}
    for phase_label, indices in phase_windows.items():
        selected = [
            wheel_samples[index]
            for index in indices
            if 0 <= index < len(wheel_samples)
        ]
        if selected:
            phase_report = _cphi_from_samples(selected)
            phase_cphi[phase_label] = CPHIReport(
                wheels=dict(phase_report.items()),
                thresholds=base_thresholds,
            )
        else:
            phase_cphi[phase_label] = CPHIReport(
                wheels={suffix: CPHIWheel() for suffix in _WHEEL_SUFFIXES},
                thresholds=base_thresholds,
            )

    phase_cphi = replicate_phase_aliases(phase_cphi)

    aero = compute_aero_coherence(records, bundles)
    coherence_values: list[float] = []
    frequency_label = ""
    if bundles:
        coherence_values = [float(getattr(bundle, "coherence_index", 0.0)) for bundle in bundles]
        if bundles:
            frequency_label = str(getattr(bundles[-1], "nu_f_label", ""))
    raw_coherence = mean(coherence_values) if coherence_values else 0.0
    ackermann_parallel = mean(ackermann_samples) if ackermann_samples else 0.0
    ackermann_sample_count = sum(1 for sample in ackermann_samples if math.isfinite(sample))
    rake_velocity_profile = [
        (aero_balance_drift.low_speed.rake_mean, int(aero_balance_drift.low_speed.samples)),
        (
            aero_balance_drift.medium_speed.rake_mean,
            int(aero_balance_drift.medium_speed.samples),
        ),
        (aero_balance_drift.high_speed.rake_mean, int(aero_balance_drift.high_speed.samples)),
    ]
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
        rake_velocity_profile=rake_velocity_profile,
        ackermann_parallel_index=ackermann_parallel,
        ackermann_parallel_samples=ackermann_sample_count,
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
        phase_synchrony_index=synchrony,
        useful_dissonance_ratio=udr,
        useful_dissonance_percentage=udr * 100.0,
        coherence_index=normalised_coherence,
        ackermann_parallel_index=ackermann_parallel,
        slide_catch_budget=slide_catch_budget,
        locking_window_score=locking_window_score,
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
        motor_latency_ms=motor_latency_ms,
        phase_motor_latency_ms=phase_latency_map,
        mu_balance=mu_balance,
        mu_symmetry=mu_symmetry,
        delta_nfr_std=delta_std,
        nodal_delta_nfr_std=nodal_std,
        delta_nfr_entropy=phase_entropy_value,
        node_entropy=node_entropy_value,
        exit_gear_match=exit_gear_match,
        shift_stability=shift_stability,
        suspension_velocity_front=front_velocity_profile,
        suspension_velocity_rear=rear_velocity_profile,
        frequency_label=frequency_label,
        aero_coherence=aero,
        aero_mechanical_coherence=aero_mechanical,
        epi_derivative_abs=epi_abs_derivative,
        si_variance=si_variance,
        brake_headroom=_compute_brake_headroom(records),
        cphi=cphi_overall,
        phase_cphi=phase_cphi,
        aero_balance_drift=aero_balance_drift,
        phase_delta_nfr_std=phase_delta_std_map,
        phase_nodal_delta_nfr_std=phase_nodal_std_map,
        phase_delta_nfr_entropy=phase_delta_entropy_map,
        phase_node_entropy=phase_node_entropy_map,
        brake_longitudinal_correlation=brake_longitudinal_correlation,
        throttle_longitudinal_correlation=throttle_longitudinal_correlation,
        phase_brake_longitudinal_correlation=phase_brake_correlation_map,
        phase_throttle_longitudinal_correlation=phase_throttle_correlation_map,
    )


def compute_aero_coherence(
    records: Sequence[SupportsTelemetrySample],
    bundles: Sequence[SupportsEPIBundle] | None = None,
    *,
    low_speed_threshold: float = 35.0,
    high_speed_threshold: float = 50.0,
    imbalance_tolerance: float = 0.08,
) -> AeroCoherence:
    """Compute aero balance deltas at low and high speed.

    The helper inspects ΔNFR contributions attributed to μ_eff front/rear terms
    in the :attr:`~tnfr_core.runtime.shared.SupportsEPIBundle.delta_breakdown`
    payload.
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

    axis_order = ("total", "lateral", "longitudinal")
    front_components: list[list[float]] = []
    rear_components: list[list[float]] = []
    speed_samples: list[float] = []

    for index, bundle in enumerate(bundles):
        speed = _resolve_speed(index)
        if speed is None:
            continue
        components = _aero_components(getattr(bundle, "delta_breakdown", {}))
        if all(front == 0.0 and rear == 0.0 for front, rear in components.values()):
            continue
        speed_samples.append(speed)
        front_components.append([components[axis][0] for axis in axis_order])
        rear_components.append([components[axis][1] for axis in axis_order])

    axis_count = len(axis_order)
    front_array = (
        xp.asarray(front_components, dtype=float)
        if front_components
        else xp.zeros((0, axis_count), dtype=float)
    )
    rear_array = (
        xp.asarray(rear_components, dtype=float)
        if rear_components
        else xp.zeros((0, axis_count), dtype=float)
    )
    speed_array = xp.asarray(speed_samples if speed_samples else (), dtype=float)

    low_mask = speed_array <= low_speed_threshold
    high_mask = speed_array >= high_speed_threshold
    medium_mask = xp.logical_and(speed_array > low_speed_threshold, speed_array < high_speed_threshold)

    def _scalar_to_int(value: object) -> int:
        scalar = value.item() if hasattr(value, "item") else value
        return int(scalar)

    def _band_from_mask(mask: object) -> AeroBandCoherence:
        mask_bool = xp.asarray(mask, dtype=bool)
        samples = _scalar_to_int(xp.sum(mask_bool))
        if samples > 0:
            mask_float = mask_bool.astype(front_array.dtype)
            front_sum = xp.sum(front_array * mask_float[:, None], axis=0)
            rear_sum = xp.sum(rear_array * mask_float[:, None], axis=0)
            front_avg = front_sum / samples
            rear_avg = rear_sum / samples
        else:
            front_avg = xp.zeros((axis_count,), dtype=float)
            rear_avg = xp.zeros((axis_count,), dtype=float)

        front_values = np.asarray(front_avg, dtype=float)
        rear_values = np.asarray(rear_avg, dtype=float)

        return AeroBandCoherence(
            total=AeroAxisCoherence(front_values[0], rear_values[0]),
            lateral=AeroAxisCoherence(front_values[1], rear_values[1]),
            longitudinal=AeroAxisCoherence(front_values[2], rear_values[2]),
            samples=samples,
        )

    low_band = _band_from_mask(low_mask)
    medium_band = _band_from_mask(medium_mask)
    high_band = _band_from_mask(high_mask)

    low_imbalance = low_band.total.imbalance
    medium_imbalance = medium_band.total.imbalance
    high_imbalance = high_band.total.imbalance

    guidance: str
    if high_band.samples == 0:
        guidance = "No aero data"
    elif abs(high_imbalance) <= imbalance_tolerance:
        guidance = "High speed aero balance is neutral"
    elif high_imbalance > 0.0:
        guidance = "High speed → add rear wing"
    else:
        guidance = "High speed → trim rear wing / add front wing"

    if low_band.samples and abs(low_imbalance) > imbalance_tolerance:
        direction = "rear" if low_imbalance > 0.0 else "front"
        guidance += f" · low speed bias {direction}"
    if medium_band.samples and abs(medium_imbalance) > imbalance_tolerance:
        direction = "rear" if medium_imbalance > 0.0 else "front"
        guidance += f" · medium speed bias {direction}"

    return AeroCoherence(
        low_speed=low_band,
        medium_speed=medium_band,
        high_speed=high_band,
        guidance=guidance,
    )


xp = jnp if _HAS_JAX and jnp is not None else np


def resolve_aero_mechanical_coherence(
    coherence_index: float,
    aero: AeroCoherence,
    *,
    suspension_deltas: Sequence[float] | None = None,
    tyre_deltas: Sequence[float] | None = None,
    target_delta_nfr: float = 0.0,
    target_mechanical_ratio: float = 0.55,
    target_aero_imbalance: float = 0.12,
    rake_velocity_profile: Sequence[tuple[float, int]] | None = None,
    ackermann_parallel_index: float | None = None,
    ackermann_parallel_samples: int | None = None,
) -> float:
    """Return a blended aero-mechanical coherence indicator in ``[0, 1]``."""

    coherence = max(0.0, min(1.0, float(coherence_index)))
    if coherence <= 0.0 or not math.isfinite(coherence):
        return 0.0

    suspension_array = xp.asarray(
        suspension_deltas if suspension_deltas is not None else (), dtype=float
    )
    tyre_array = xp.asarray(tyre_deltas if tyre_deltas is not None else (), dtype=float)

    suspension_average = (
        float(xp.mean(xp.abs(suspension_array))) if suspension_array.size else 0.0
    )
    tyre_average = float(xp.mean(xp.abs(tyre_array))) if tyre_array.size else 0.0

    sample_counts = xp.asarray(
        (
            float(aero.high_speed_samples),
            float(aero.medium_speed_samples),
            float(aero.low_speed_samples),
        ),
        dtype=float,
    )
    sample_total = float(xp.sum(sample_counts))
    total_samples = int(sample_total)
    if sample_total > 0.0:
        weights = sample_counts / sample_total
    else:
        weights = xp.asarray((1.0 / 3.0,) * 3, dtype=float)

    front_values = xp.asarray(
        (
            float(aero.high_speed_front),
            float(aero.medium_speed_front),
            float(aero.low_speed_front),
        ),
        dtype=float,
    )
    rear_values = xp.asarray(
        (
            float(aero.high_speed_rear),
            float(aero.medium_speed_rear),
            float(aero.low_speed_rear),
        ),
        dtype=float,
    )
    magnitude_pairs = xp.abs(xp.stack((front_values, rear_values), axis=-1))
    per_band_magnitude = xp.mean(magnitude_pairs, axis=-1)
    aero_magnitude = float(xp.average(per_band_magnitude, weights=weights))

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
    imbalance_values = xp.abs(
        xp.asarray(
            (
                float(aero.high_speed_imbalance),
                float(aero.medium_speed_imbalance),
                float(aero.low_speed_imbalance),
            ),
            dtype=float,
        )
    )
    weighted_imbalance = float(xp.average(imbalance_values, weights=weights))
    aero_factor = max(0.0, 1.0 - min(1.0, weighted_imbalance / imbalance_target))
    if total_samples == 0:
        aero_factor *= 0.5

    def _body_factor(profile: Sequence[tuple[float, int]] | None) -> float | None:
        if not profile:
            return None
        magnitudes: list[float] = []
        counts: list[float] = []
        for rake_mean, samples in profile:
            count = int(samples)
            if count <= 0:
                continue
            try:
                rake_value = float(rake_mean)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(rake_value):
                continue
            magnitudes.append(abs(rake_value))
            counts.append(count)
        if not counts:
            return None
        counts_array = xp.asarray(counts, dtype=float)
        total_weight = float(xp.sum(counts_array))
        if total_weight <= 0.0:
            return None
        magnitudes_array = xp.asarray(magnitudes, dtype=float)
        tolerance = math.radians(2.5)
        if tolerance <= 1e-9:
            return None
        weighted_rake = float(xp.average(magnitudes_array, weights=counts_array))
        ratio = weighted_rake / tolerance
        return max(0.0, 1.0 - min(1.0, ratio))

    def _steering_factor(
        index: float | None, samples: int | None = None
    ) -> float | None:
        if index is None:
            return None
        try:
            value = float(index)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        tolerance = 0.25
        magnitude = float(xp.abs(xp.asarray(value, dtype=float)))
        ratio = magnitude / tolerance if tolerance > 1e-9 else 0.0
        factor = max(0.0, 1.0 - min(1.0, ratio))
        sample_value = 0.0
        if samples is not None:
            try:
                sample_value = float(samples)
            except (TypeError, ValueError):
                sample_value = 0.0
        sample_count = int(float(xp.asarray(sample_value, dtype=float)))
        if sample_count <= 0:
            return factor * 0.5
        return factor

    def _aero_velocity_factor(bands: AeroCoherence) -> float | None:
        entries: list[tuple[float, float, float]] = []
        for band in (
            bands.low_speed,
            bands.medium_speed,
            bands.high_speed,
        ):
            count = int(getattr(band, "samples", 0))
            if count <= 0:
                continue
            front_value = float(getattr(band.total, "front", 0.0))
            rear_value = float(getattr(band.total, "rear", 0.0))
            if not math.isfinite(front_value) or not math.isfinite(rear_value):
                continue
            front_abs = abs(front_value)
            rear_abs = abs(rear_value)
            total = front_abs + rear_abs
            if total <= 1e-9:
                continue
            entries.append((count, front_abs, rear_abs))
        if not entries:
            return None
        counts = xp.asarray([entry[0] for entry in entries], dtype=float)
        total_weight = float(xp.sum(counts))
        if total_weight <= 0.0:
            return None
        front = xp.asarray([entry[1] for entry in entries], dtype=float)
        rear = xp.asarray([entry[2] for entry in entries], dtype=float)
        totals = front + rear
        imbalance_ratio = xp.minimum(1.0, xp.abs(front - rear) / totals)
        balance = 1.0 - imbalance_ratio
        weighted = xp.average(balance, weights=counts)
        return float(xp.clip(weighted, 0.0, 1.0))

    body_factor = _body_factor(rake_velocity_profile)
    steering_factor = _steering_factor(
        ackermann_parallel_index, ackermann_parallel_samples
    )
    aero_velocity_factor = _aero_velocity_factor(aero)

    components = [ratio_factor, coverage_factor, aero_factor]
    for candidate in (body_factor, steering_factor, aero_velocity_factor):
        if candidate is not None:
            components.append(candidate)
    if not components:
        return 0.0
    composite = sum(components) / len(components)
    return float(max(0.0, min(1.0, coherence * composite)))


def _segment_gradients(
    records: Sequence[SupportsTelemetrySample], *, segments: int, fallback_to_chronological: bool = True
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
    records: Sequence[SupportsTelemetrySample], segments: int
) -> list[Sequence[SupportsTelemetrySample]]:
    length = len(records)
    if length == 0:
        return [tuple()] * segments

    base, remainder = divmod(length, segments)
    slices: list[Sequence[SupportsTelemetrySample]] = []
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
    records: Iterable[SupportsTelemetrySample],
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

_SUSPENSION_LOW_SPEED_THRESHOLD = 0.05
_SUSPENSION_HIGH_SPEED_THRESHOLD = 0.2



