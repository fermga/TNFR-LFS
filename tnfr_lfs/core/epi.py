"""EPI extraction and ΔNFR/ΔSi computations."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from collections.abc import Mapping as MappingABC
from statistics import mean
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .coherence_calibration import CoherenceCalibrationStore
    from ..cache_settings import CacheOptions

from .cache import (
    cached_delta_nfr_map,
    cached_dynamic_multipliers,
    invalidate_dynamic_record,
    delta_cache_enabled,
    dynamic_cache_enabled,
)
from .coherence import compute_node_delta_nfr, sense_index
from .delta_utils import distribute_weighted_delta
from .epi_models import (
    BrakesNode,
    ChassisNode,
    DriverNode,
    EPIBundle,
    SuspensionNode,
    TrackNode,
    TransmissionNode,
    TyresNode,
)
from .phases import expand_phase_alias, phase_family

# Natural frequencies for each subsystem (Hz) extracted from the
# Activation-model tables in the manual.  They provide the base
# rate at which ΔNFR fluctuations modulate the EPI dynamics.
NU_F_NODE_DEFAULTS: Mapping[str, float] = {
    "tyres": 0.18,
    "suspension": 0.14,
    "chassis": 0.12,
    "brakes": 0.16,
    "transmission": 0.11,
    "track": 0.08,
    "driver": 0.05,
}

DEFAULT_PHASE_WEIGHTS: Mapping[str, float] = {"__default__": 1.0}

_AVERAGE_NU_F = sum(NU_F_NODE_DEFAULTS.values()) / float(len(NU_F_NODE_DEFAULTS))
_MISSING_FLOAT = float("nan")


@dataclass(frozen=True)
class NaturalFrequencySettings:
    """Configuration options for the natural frequency estimator."""

    min_window_seconds: float = 2.0
    max_window_seconds: float = 5.0
    bandpass_low_hz: float = 0.1
    bandpass_high_hz: float = 6.0
    min_multiplier: float = 0.5
    max_multiplier: float = 2.5
    smoothing_alpha: float = 0.25
    vehicle_frequency: Mapping[str, float] = field(
        default_factory=lambda: {"__default__": _AVERAGE_NU_F}
    )
    default_category: str = "generic"
    frequency_bands: Mapping[str, Tuple[float, float]] = field(
        default_factory=lambda: {"generic": (1.6, 2.4)}
    )
    car_categories: Mapping[str, str] = field(default_factory=dict)
    structural_density_window: int = 6
    structural_density_normaliser: float = 5.0

    def resolve_vehicle_frequency(self, car_model: str | None) -> float:
        if car_model and car_model in self.vehicle_frequency:
            return max(1e-6, float(self.vehicle_frequency[car_model]))
        if "__default__" in self.vehicle_frequency:
            return max(1e-6, float(self.vehicle_frequency["__default__"]))
        return max(1e-6, _AVERAGE_NU_F)

    def resolve_category(self, car_model: str | None) -> str:
        if car_model and car_model in self.car_categories:
            return str(self.car_categories[car_model])
        return str(self.default_category)

    def resolve_frequency_band(self, car_model: str | None) -> Tuple[float, float]:
        category = self.resolve_category(car_model)
        if category in self.frequency_bands:
            low, high = self.frequency_bands[category]
        elif "__default__" in self.frequency_bands:
            low, high = self.frequency_bands["__default__"]
        else:
            low, high = (1.6, 2.4)
        low = max(0.0, float(low))
        high = max(low, float(high))
        return (low, high)


@dataclass(frozen=True)
class NaturalFrequencySnapshot:
    """Result of a natural frequency analysis step."""

    by_node: Mapping[str, float]
    dominant_frequency: float
    classification: str
    category: str
    target_band: Tuple[float, float]
    coherence_index: float

    @property
    def frequency_label(self) -> str:
        if self.dominant_frequency <= 0.0:
            return "ν_f no data"
        low, high = self.target_band
        return (
            f"ν_f {self.classification} {self.dominant_frequency:.2f}Hz "
            f"(obj {low:.2f}-{high:.2f}Hz)"
        )


class NaturalFrequencyAnalyzer:
    """Track dominant frequencies over a sliding telemetry window."""

    def __init__(
        self,
        settings: NaturalFrequencySettings | None = None,
        cache_options: "CacheOptions" | None = None,
    ) -> None:
        self.settings = settings or NaturalFrequencySettings()
        self._history: Deque[TelemetryRecord] = deque()
        self._smoothed: Dict[str, float] = {}
        self._last_car: str | None = None
        self._last_snapshot: NaturalFrequencySnapshot | None = None
        self._cache_options = cache_options

    def _dynamic_cache_active(self) -> bool:
        if self._cache_options is not None:
            return self._cache_options.nu_f_cache_size > 0
        return dynamic_cache_enabled()

    def reset(self) -> None:
        if self._dynamic_cache_active():
            for sample in list(self._history):
                invalidate_dynamic_record(sample)
        self._history.clear()
        self._smoothed.clear()
        self._last_car = None
        self._last_snapshot = None

    def update(
        self,
        record: TelemetryRecord,
        base_map: Mapping[str, float],
        *,
        car_model: str | None = None,
    ) -> NaturalFrequencySnapshot:
        self._append_record(record)
        mapping, dominant_frequency = self._resolve(base_map, car_model)
        return self._build_snapshot(mapping, dominant_frequency, car_model)

    def compute_from_history(
        self,
        history: Sequence[TelemetryRecord],
        base_map: Mapping[str, float],
        *,
        record: TelemetryRecord | None = None,
        car_model: str | None = None,
    ) -> NaturalFrequencySnapshot:
        if self._dynamic_cache_active():
            for sample in list(self._history):
                invalidate_dynamic_record(sample)
        self._history.clear()
        for sample in history:
            self._append_record(sample)
        if record is not None and (not self._history or self._history[-1] is not record):
            self._append_record(record)
        mapping, dominant_frequency = self._resolve(base_map, car_model)
        return self._build_snapshot(mapping, dominant_frequency, car_model)

    def _append_record(self, record: TelemetryRecord) -> None:
        self._history.append(record)
        max_window = max(self.settings.max_window_seconds, self.settings.min_window_seconds)
        while (
            len(self._history) >= 2
            and (self._history[-1].timestamp - self._history[0].timestamp) > max_window
        ):
            removed = self._history.popleft()
            if self._dynamic_cache_active():
                invalidate_dynamic_record(removed)

    def _resolve(
        self, base_map: Mapping[str, float], car_model: str | None
    ) -> Tuple[Dict[str, float], float]:
        if car_model != self._last_car:
            self._smoothed.clear()
            self._last_car = car_model

        dynamic, dominant_frequency = self._dynamic_multipliers(car_model)
        if not dynamic:
            # No dynamic adjustment available, fall back to smoothing defaults.
            mapping = self._apply_smoothing(dict(base_map))
            return mapping, dominant_frequency

        adjusted = {}
        for node, value in base_map.items():
            multiplier = dynamic.get(node, 1.0)
            adjusted[node] = value * multiplier
        mapping = self._apply_smoothing(adjusted)
        return mapping, dominant_frequency

    def _dynamic_multipliers(
        self, car_model: str | None
    ) -> Tuple[Dict[str, float], float]:
        history = tuple(self._history)
        if len(history) < 2:
            return {}, 0.0
        if not self._dynamic_cache_active():
            return self._compute_dynamic_multipliers_raw(history, car_model)
        return cached_dynamic_multipliers(
            car_model,
            history,
            lambda: self._compute_dynamic_multipliers_raw(history, car_model),
        )

    def _compute_dynamic_multipliers_raw(
        self,
        history: Sequence[TelemetryRecord],
        car_model: str | None,
    ) -> Tuple[Dict[str, float], float]:
        history = tuple(history)
        if len(history) < 2:
            return {}, 0.0

        duration = history[-1].timestamp - history[0].timestamp
        if duration < max(0.0, self.settings.min_window_seconds - 1e-6):
            return {}, 0.0

        from .spectrum import cross_spectrum, power_spectrum, estimate_sample_rate

        sample_rate = estimate_sample_rate(history)
        if sample_rate <= 0.0:
            return {}, 0.0

        min_samples = max(4, int(self.settings.min_window_seconds * sample_rate))
        if len(history) < min_samples:
            return {}, 0.0

        sample_count = len(history)
        channel_count = 5
        channel_matrix = np.fromiter(
            (
                float(value)
                for record in history
                for value in (
                    record.steer,
                    record.throttle,
                    record.brake_pressure,
                    record.suspension_velocity_front,
                    record.suspension_velocity_rear,
                )
            ),
            dtype=float,
            count=sample_count * channel_count,
        ).reshape(sample_count, channel_count)

        steer_series = channel_matrix[:, 0]
        throttle_series = channel_matrix[:, 1]
        brake_series = channel_matrix[:, 2]
        suspension_combined = np.mean(channel_matrix[:, 3:5], axis=1)

        low = max(0.0, self.settings.bandpass_low_hz)
        high = max(low, self.settings.bandpass_high_hz)

        def _dominant_frequency(series: np.ndarray) -> float:
            if series.size == 0:
                return 0.0
            spectrum = power_spectrum(series, sample_rate)
            if not spectrum:
                return 0.0
            spectrum_array = np.asarray(spectrum, dtype=float)
            frequencies = spectrum_array[:, 0]
            energies = spectrum_array[:, 1]
            mask = (frequencies >= low) & (frequencies <= high)
            if not np.any(mask):
                return 0.0
            masked_energies = energies[mask]
            peak_index = int(np.argmax(masked_energies))
            peak_energy = masked_energies[peak_index]
            if peak_energy <= 1e-9:
                return 0.0
            masked_frequencies = frequencies[mask]
            return float(masked_frequencies[peak_index])

        def _dominant_cross(x_series: np.ndarray, y_series: np.ndarray) -> float:
            if x_series.size == 0 or y_series.size == 0:
                return 0.0
            spectrum = cross_spectrum(x_series, y_series, sample_rate)
            if not spectrum:
                return 0.0
            spectrum_array = np.asarray(spectrum, dtype=float)
            frequencies = spectrum_array[:, 0]
            reals = spectrum_array[:, 1]
            imags = spectrum_array[:, 2]
            mask = (frequencies >= low) & (frequencies <= high)
            if not np.any(mask):
                return 0.0
            masked_reals = reals[mask]
            masked_imags = imags[mask]
            magnitudes = np.hypot(masked_reals, masked_imags)
            peak_index = int(np.argmax(magnitudes))
            if magnitudes[peak_index] <= 1e-9:
                return 0.0
            masked_frequencies = frequencies[mask]
            return float(masked_frequencies[peak_index])

        steer_freq = _dominant_frequency(steer_series)
        throttle_freq = _dominant_frequency(throttle_series)
        brake_freq = _dominant_frequency(brake_series)
        suspension_freq = _dominant_frequency(suspension_combined)
        tyre_freq = _dominant_cross(steer_series, suspension_combined)

        vehicle_frequency = self.settings.resolve_vehicle_frequency(car_model)

        def _normalise(frequency: float) -> float:
            if frequency <= 0.0:
                return 1.0
            ratio = frequency / vehicle_frequency
            ratio = max(self.settings.min_multiplier, min(self.settings.max_multiplier, ratio))
            return ratio

        dominant_frequency = 0.0
        multipliers: Dict[str, float] = {}
        if steer_freq > 0.0:
            multipliers["driver"] = _normalise(steer_freq)
            dominant_frequency = steer_freq
        if throttle_freq > 0.0:
            multipliers["transmission"] = _normalise(throttle_freq)
            if throttle_freq > dominant_frequency:
                dominant_frequency = throttle_freq
        if brake_freq > 0.0:
            multipliers["brakes"] = _normalise(brake_freq)
            if brake_freq > dominant_frequency:
                dominant_frequency = brake_freq
        if suspension_freq > 0.0:
            value = _normalise(suspension_freq)
            multipliers["suspension"] = value
            multipliers.setdefault("chassis", value)
            if suspension_freq > dominant_frequency:
                dominant_frequency = suspension_freq
        if tyre_freq > 0.0:
            value = _normalise(tyre_freq)
            multipliers["tyres"] = value
            multipliers["chassis"] = value
            if tyre_freq > dominant_frequency:
                dominant_frequency = tyre_freq
        elif suspension_freq > 0.0 and steer_freq > 0.0:
            blended = (multipliers["suspension"] + multipliers["driver"]) * 0.5
            multipliers["tyres"] = blended

        return multipliers, dominant_frequency

    def _apply_smoothing(self, mapping: Dict[str, float]) -> Dict[str, float]:
        alpha = self.settings.smoothing_alpha
        if not (0.0 < alpha < 1.0):
            # Smoothing disabled, return mapping as-is.
            for node, value in mapping.items():
                self._smoothed[node] = value
            return mapping

        for node, value in mapping.items():
            previous = self._smoothed.get(node)
            if previous is None:
                smoothed = value
            else:
                smoothed = (alpha * value) + ((1.0 - alpha) * previous)
            self._smoothed[node] = smoothed
            mapping[node] = smoothed
        return mapping

    def _coherence_index(self) -> float:
        history = list(self._history)
        if len(history) < 2:
            return 0.0
        densities: List[float] = []
        for previous, current in zip(history[:-1], history[1:]):
            chrono_dt = float(current.timestamp) - float(previous.timestamp)
            if chrono_dt <= 1e-9:
                continue
            structural_curr = getattr(current, "structural_timestamp", None)
            structural_prev = getattr(previous, "structural_timestamp", None)
            if structural_curr is None or structural_prev is None:
                continue
            structural_dt = float(structural_curr) - float(structural_prev)
            if structural_dt <= 0.0:
                continue
            ratio = structural_dt / chrono_dt
            densities.append(max(0.0, ratio - 1.0))
        if not densities:
            return 0.0
        window = max(1, int(self.settings.structural_density_window))
        tail = densities[-window:]
        average_density = sum(tail) / len(tail)
        normaliser = max(1e-6, float(self.settings.structural_density_normaliser))
        return max(0.0, min(1.0, average_density / normaliser))

    def _build_snapshot(
        self,
        mapping: Mapping[str, float],
        dominant_frequency: float,
        car_model: str | None,
    ) -> NaturalFrequencySnapshot:
        category = self.settings.resolve_category(car_model)
        band = self.settings.resolve_frequency_band(car_model)
        classification = self._classify_frequency(dominant_frequency, band)
        snapshot = NaturalFrequencySnapshot(
            by_node=dict(mapping),
            dominant_frequency=dominant_frequency,
            classification=classification,
            category=category,
            target_band=band,
            coherence_index=self._coherence_index(),
        )
        self._last_snapshot = snapshot
        return snapshot

    @staticmethod
    def _classify_frequency(frequency: float, band: Tuple[float, float]) -> str:
        low, high = band
        if frequency <= 0.0:
            return "no data"
        if frequency < low * 0.9:
            return "very low"
        if frequency < low:
            return "low"
        if frequency <= high:
            return "optimal"
        if frequency <= high * 1.1:
            return "high"
        return "very high"

AXIS_FEATURE_MAP: Mapping[str, Mapping[str, str]] = {
    "tyres": {
        "slip_ratio": "longitudinal",
        "locking": "longitudinal",
        "slip_angle": "lateral",
        "mu_eff_front": "both",
        "mu_eff_rear": "both",
    },
    "suspension": {
        "travel_front": "both",
        "travel_rear": "both",
        "velocity_front": "longitudinal",
        "velocity_rear": "longitudinal",
        "load_front": "longitudinal",
        "load_rear": "longitudinal",
    },
    "chassis": {
        "yaw_rate": "lateral",
        "lateral_accel": "lateral",
        "roll": "lateral",
        "pitch": "longitudinal",
    },
    "brakes": {
        "pressure": "longitudinal",
        "locking": "longitudinal",
        "longitudinal_decel": "longitudinal",
        "load_front": "longitudinal",
    },
    "transmission": {
        "throttle": "longitudinal",
        "longitudinal_accel": "longitudinal",
        "slip_ratio": "longitudinal",
        "gear": "longitudinal",
        "speed": "longitudinal",
    },
    "track": {
        "mu_eff_front": "lateral",
        "mu_eff_rear": "lateral",
        "axle_load_balance": "lateral",
        "axle_velocity_balance": "lateral",
        "yaw": "lateral",
        "vertical_load": "longitudinal",
    },
    "driver": {
        "style_index": "lateral",
        "steer": "lateral",
        "throttle": "longitudinal",
        "yaw_rate": "lateral",
    },
}


@dataclass(frozen=True)
class TelemetryRecord:
    """Single telemetry sample emitted by the acquisition backend."""

    timestamp: float
    vertical_load: float
    slip_ratio: float
    lateral_accel: float
    longitudinal_accel: float
    yaw: float
    pitch: float
    roll: float
    brake_pressure: float
    locking: float
    nfr: float
    si: float
    speed: float
    yaw_rate: float
    slip_angle: float
    steer: float
    throttle: float
    gear: int
    vertical_load_front: float
    vertical_load_rear: float
    mu_eff_front: float
    mu_eff_rear: float
    mu_eff_front_lateral: float
    mu_eff_front_longitudinal: float
    mu_eff_rear_lateral: float
    mu_eff_rear_longitudinal: float
    suspension_travel_front: float
    suspension_travel_rear: float
    suspension_velocity_front: float
    suspension_velocity_rear: float
    slip_ratio_fl: float = _MISSING_FLOAT
    slip_ratio_fr: float = _MISSING_FLOAT
    slip_ratio_rl: float = _MISSING_FLOAT
    slip_ratio_rr: float = _MISSING_FLOAT
    slip_angle_fl: float = _MISSING_FLOAT
    slip_angle_fr: float = _MISSING_FLOAT
    slip_angle_rl: float = _MISSING_FLOAT
    slip_angle_rr: float = _MISSING_FLOAT
    brake_input: float = _MISSING_FLOAT
    clutch_input: float = _MISSING_FLOAT
    handbrake_input: float = _MISSING_FLOAT
    steer_input: float = _MISSING_FLOAT
    wheel_load_fl: float = _MISSING_FLOAT
    wheel_load_fr: float = _MISSING_FLOAT
    wheel_load_rl: float = _MISSING_FLOAT
    wheel_load_rr: float = _MISSING_FLOAT
    wheel_lateral_force_fl: float = _MISSING_FLOAT
    wheel_lateral_force_fr: float = _MISSING_FLOAT
    wheel_lateral_force_rl: float = _MISSING_FLOAT
    wheel_lateral_force_rr: float = _MISSING_FLOAT
    wheel_longitudinal_force_fl: float = _MISSING_FLOAT
    wheel_longitudinal_force_fr: float = _MISSING_FLOAT
    wheel_longitudinal_force_rl: float = _MISSING_FLOAT
    wheel_longitudinal_force_rr: float = _MISSING_FLOAT
    suspension_deflection_fl: float = _MISSING_FLOAT
    suspension_deflection_fr: float = _MISSING_FLOAT
    suspension_deflection_rl: float = _MISSING_FLOAT
    suspension_deflection_rr: float = _MISSING_FLOAT
    structural_timestamp: float | None = None
    tyre_temp_fl: float = _MISSING_FLOAT
    tyre_temp_fr: float = _MISSING_FLOAT
    tyre_temp_rl: float = _MISSING_FLOAT
    tyre_temp_rr: float = _MISSING_FLOAT
    tyre_temp_fl_inner: float = _MISSING_FLOAT
    tyre_temp_fr_inner: float = _MISSING_FLOAT
    tyre_temp_rl_inner: float = _MISSING_FLOAT
    tyre_temp_rr_inner: float = _MISSING_FLOAT
    tyre_temp_fl_middle: float = _MISSING_FLOAT
    tyre_temp_fr_middle: float = _MISSING_FLOAT
    tyre_temp_rl_middle: float = _MISSING_FLOAT
    tyre_temp_rr_middle: float = _MISSING_FLOAT
    tyre_temp_fl_outer: float = _MISSING_FLOAT
    tyre_temp_fr_outer: float = _MISSING_FLOAT
    tyre_temp_rl_outer: float = _MISSING_FLOAT
    tyre_temp_rr_outer: float = _MISSING_FLOAT
    tyre_pressure_fl: float = _MISSING_FLOAT
    tyre_pressure_fr: float = _MISSING_FLOAT
    tyre_pressure_rl: float = _MISSING_FLOAT
    tyre_pressure_rr: float = _MISSING_FLOAT
    brake_temp_fl: float = _MISSING_FLOAT
    brake_temp_fr: float = _MISSING_FLOAT
    brake_temp_rl: float = _MISSING_FLOAT
    brake_temp_rr: float = _MISSING_FLOAT
    rpm: float = _MISSING_FLOAT
    line_deviation: float = _MISSING_FLOAT
    instantaneous_radius: float = _MISSING_FLOAT
    front_track_width: float = _MISSING_FLOAT
    wheelbase: float = _MISSING_FLOAT
    lap: int | str | None = None
    reference: Optional["TelemetryRecord"] = None
    car_model: str | None = None
    track_name: str | None = None
    tyre_compound: str | None = None


def _angle_difference(value: float, reference: float) -> float:
    """Return the wrapped angular delta between ``value`` and ``reference``."""

    delta = value - reference
    if not math.isfinite(delta):
        return 0.0
    wrapped = (delta + math.pi) % (2.0 * math.pi)
    return wrapped - math.pi


def _abs_delta(value: float, base: float) -> float:
    delta_value = value - base
    if not math.isfinite(delta_value):
        return 0.0
    return abs(delta_value)


def _resolve_track_gradient(record: TelemetryRecord) -> float:
    """Return the track gradient associated with ``record``."""

    for attribute in ("track_gradient", "gradient", "slope", "track_slope"):
        try:
            candidate = getattr(record, attribute)
        except AttributeError:
            continue
        if candidate is None:
            continue
        try:
            gradient = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(gradient):
            return gradient

    pitch_value = getattr(record, "pitch", 0.0)
    try:
        pitch = float(pitch_value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(pitch):
        return 0.0
    return math.tan(pitch)


def _ackermann_parallel_delta(record: TelemetryRecord, baseline: TelemetryRecord) -> float:
    """Return the parallel steer delta derived from wheel slip angles."""

    def _turn_direction(sample: TelemetryRecord, fallback: int = 0) -> int:
        try:
            yaw_rate = float(getattr(sample, "yaw_rate", 0.0))
        except (TypeError, ValueError):
            yaw_rate = 0.0
        if not math.isfinite(yaw_rate) or abs(yaw_rate) <= 1e-6:
            return fallback
        return 1 if yaw_rate > 0.0 else -1

    def _wheel_slip(sample: TelemetryRecord, direction: int) -> tuple[float, float]:
        default_angle = float(getattr(sample, "slip_angle", 0.0))

        def _clean(value: float) -> float:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return default_angle
            if not math.isfinite(numeric):
                return default_angle
            return numeric

        if direction >= 0:
            inner_attr, outer_attr = "slip_angle_fl", "slip_angle_fr"
        else:
            inner_attr, outer_attr = "slip_angle_fr", "slip_angle_fl"
        inner = _clean(getattr(sample, inner_attr, default_angle))
        outer = _clean(getattr(sample, outer_attr, default_angle))
        return inner, outer

    direction = _turn_direction(record)
    baseline_direction = _turn_direction(baseline, direction)
    effective_direction = direction or baseline_direction
    if effective_direction == 0:
        return 0.0

    observed_inner, observed_outer = _wheel_slip(record, effective_direction)
    baseline_inner, baseline_outer = _wheel_slip(
        baseline, baseline_direction or effective_direction
    )
    observed_delta = observed_inner - observed_outer
    baseline_delta = baseline_inner - baseline_outer
    if not math.isfinite(observed_delta) or not math.isfinite(baseline_delta):
        return 0.0
    return observed_delta - baseline_delta


def _node_feature_contributions(
    record: TelemetryRecord, baseline: TelemetryRecord
) -> Dict[str, Dict[str, float]]:
    slip_delta = _abs_delta(record.slip_ratio, baseline.slip_ratio)
    slip_angle_delta = abs(_angle_difference(record.slip_angle, baseline.slip_angle))
    lat_delta = _abs_delta(record.lateral_accel, baseline.lateral_accel)
    long_delta = _abs_delta(record.longitudinal_accel, baseline.longitudinal_accel)
    yaw_delta = abs(_angle_difference(record.yaw, baseline.yaw))
    yaw_rate_delta = _abs_delta(record.yaw_rate, baseline.yaw_rate)
    pitch_delta = _abs_delta(record.pitch, baseline.pitch)
    roll_delta = _abs_delta(record.roll, baseline.roll)
    brake_delta = _abs_delta(record.brake_pressure, baseline.brake_pressure)
    locking_delta = _abs_delta(record.locking, baseline.locking)
    si_delta = _abs_delta(record.si, baseline.si)
    throttle_delta = _abs_delta(record.throttle, baseline.throttle)
    speed_delta = _abs_delta(record.speed, baseline.speed)
    steer_delta = _abs_delta(record.steer, baseline.steer)
    gear_delta = abs(record.gear - baseline.gear)
    load_delta = _abs_delta(record.vertical_load, baseline.vertical_load)
    load_front_delta = _abs_delta(record.vertical_load_front, baseline.vertical_load_front)
    load_rear_delta = _abs_delta(record.vertical_load_rear, baseline.vertical_load_rear)
    mu_front_delta = _abs_delta(record.mu_eff_front, baseline.mu_eff_front)
    mu_rear_delta = _abs_delta(record.mu_eff_rear, baseline.mu_eff_rear)
    travel_front_delta = _abs_delta(
        record.suspension_travel_front, baseline.suspension_travel_front
    )
    travel_rear_delta = _abs_delta(
        record.suspension_travel_rear, baseline.suspension_travel_rear
    )
    velocity_front_delta = _abs_delta(
        record.suspension_velocity_front, baseline.suspension_velocity_front
    )
    velocity_rear_delta = _abs_delta(
        record.suspension_velocity_rear, baseline.suspension_velocity_rear
    )

    axle_balance = (record.vertical_load_front - record.vertical_load_rear) - (
        baseline.vertical_load_front - baseline.vertical_load_rear
    )
    axle_velocity_balance = (
        record.suspension_velocity_front - record.suspension_velocity_rear
    ) - (
        baseline.suspension_velocity_front - baseline.suspension_velocity_rear
    )

    brake_longitudinal_delta = max(0.0, baseline.longitudinal_accel - record.longitudinal_accel)
    drive_longitudinal_delta = max(0.0, record.longitudinal_accel - baseline.longitudinal_accel)

    return {
        "tyres": {
            "slip_ratio": slip_delta * 0.35,
            "slip_angle": slip_angle_delta * 0.25,
            "mu_eff_front": mu_front_delta * 0.2,
            "mu_eff_rear": mu_rear_delta * 0.2,
            "locking": locking_delta * 0.2,
        },
        "suspension": {
            "travel_front": travel_front_delta * 0.35,
            "travel_rear": travel_rear_delta * 0.35,
            "velocity_front": velocity_front_delta * 0.4,
            "velocity_rear": velocity_rear_delta * 0.4,
            "load_front": load_front_delta * 0.25,
            "load_rear": load_rear_delta * 0.25,
        },
        "chassis": {
            "yaw_rate": yaw_rate_delta * 0.4,
            "lateral_accel": lat_delta * 0.35,
            "roll": roll_delta * 0.15,
            "pitch": pitch_delta * 0.1,
        },
        "brakes": {
            "pressure": brake_delta * 0.4,
            "locking": locking_delta * 0.25,
            "longitudinal_decel": brake_longitudinal_delta * 0.2,
            "load_front": load_front_delta * 0.15,
        },
        "transmission": {
            "throttle": throttle_delta * 0.3,
            "longitudinal_accel": drive_longitudinal_delta * 0.25,
            "slip_ratio": slip_delta * 0.2,
            "gear": gear_delta * 0.15,
            "speed": speed_delta * 0.1,
        },
        "track": {
            "mu_eff_front": mu_front_delta * 0.3,
            "mu_eff_rear": mu_rear_delta * 0.3,
            "axle_load_balance": abs(axle_balance) * 0.25,
            "axle_velocity_balance": abs(axle_velocity_balance) * 0.2,
            "yaw": yaw_delta * 0.15,
            "vertical_load": load_delta * 0.1,
        },
        "driver": {
            "style_index": si_delta * 0.35,
            "steer": steer_delta * 0.25,
            "throttle": throttle_delta * 0.2,
            "yaw_rate": yaw_rate_delta * 0.2,
        },
    }


def _axis_signal_components(
    record: TelemetryRecord,
    baseline: TelemetryRecord,
    feature_contributions: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    """Partition feature signals into longitudinal and lateral components."""

    axis_signals = {"longitudinal": 0.0, "lateral": 0.0}
    for node, contributions in feature_contributions.items():
        feature_map = AXIS_FEATURE_MAP.get(node, {})
        for feature, value in contributions.items():
            signal = abs(value)
            axis = feature_map.get(feature)
            if axis == "longitudinal":
                axis_signals["longitudinal"] += signal
            elif axis == "lateral":
                axis_signals["lateral"] += signal
            elif axis == "both":
                axis_signals["longitudinal"] += signal * 0.5
                axis_signals["lateral"] += signal * 0.5
    if axis_signals["longitudinal"] <= 1e-9 and axis_signals["lateral"] <= 1e-9:
        long_proxy = (
            _abs_delta(record.longitudinal_accel, baseline.longitudinal_accel)
            + _abs_delta(record.brake_pressure, baseline.brake_pressure)
            + _abs_delta(record.throttle, baseline.throttle)
        )
        lat_proxy = (
            _abs_delta(record.lateral_accel, baseline.lateral_accel)
            + _abs_delta(record.steer, baseline.steer)
            + _abs_delta(record.yaw_rate, baseline.yaw_rate)
        )
        axis_signals["longitudinal"] = long_proxy
        axis_signals["lateral"] = lat_proxy
    return axis_signals


def _distribute_node_delta(
    delta_nfr: float, node_signals: Mapping[str, float]
) -> Dict[str, float]:
    return distribute_weighted_delta(delta_nfr, node_signals)


def _delta_nfr_by_node_uncached(record: TelemetryRecord) -> Dict[str, float]:
    baseline = record.reference or record
    delta_nfr = record.nfr - baseline.nfr
    feature_contributions = _node_feature_contributions(record, baseline)
    node_signals = {
        node: sum(values.values()) for node, values in feature_contributions.items()
    }
    return _distribute_node_delta(delta_nfr, node_signals)


def delta_nfr_by_node(
    record: TelemetryRecord,
    *,
    cache_options: "CacheOptions" | None = None,
) -> Mapping[str, float]:
    """Compute ΔNFR contributions for each subsystem.

    The function expects ``record`` to optionally provide a ``reference``
    sample, typically the baseline derived from the telemetry stint.  When a
    reference is available the signal strength for every subsystem is
    measured relative to it, otherwise the calculation degenerates into a
    uniform distribution.
    """

    if cache_options is not None:
        use_cache = cache_options.enable_delta_cache
    else:
        use_cache = delta_cache_enabled()
    if not use_cache:
        return _delta_nfr_by_node_uncached(record)
    return cached_delta_nfr_map(record, lambda: _delta_nfr_by_node_uncached(record))


def _phase_weight(
    node: str,
    phase: str | None,
    phase_weights: Mapping[str, Mapping[str, float] | float] | None,
) -> float:
    if not phase or not phase_weights or not isinstance(phase_weights, MappingABC):
        return 1.0
    profile: Mapping[str, float] | float | None = None
    for candidate in (*expand_phase_alias(phase), phase_family(phase), phase):
        if candidate is None:
            continue
        profile = phase_weights.get(candidate)
        if profile is not None:
            break
    if profile is None:
        profile = phase_weights.get("__default__")
    if profile is None:
        return 1.0
    if isinstance(profile, MappingABC):
        if node in profile:
            return float(profile[node])
        if "__default__" in profile:
            return float(profile["__default__"])
        return 1.0
    if isinstance(profile, (int, float)):
        return float(profile)
    return 1.0


def _base_nu_f_map(
    record: TelemetryRecord,
    *,
    phase: str | None = None,
    phase_weights: Mapping[str, Mapping[str, float] | float] | None = None,
) -> Dict[str, float]:
    slip_modifier = 1.0 + min(abs(record.slip_ratio), 1.0) * 0.2
    load_deviation = (record.vertical_load - 5000.0) / 4000.0
    load_modifier = 1.0 + max(-0.5, min(0.5, load_deviation))
    sense_modifier = 1.0 + max(-0.3, min(0.3, record.si - 0.8))
    mapping: Dict[str, float] = {}
    for node, base_value in NU_F_NODE_DEFAULTS.items():
        if node == "tyres":
            value = base_value * slip_modifier
        elif node == "suspension":
            value = base_value * load_modifier
        elif node == "driver":
            value = base_value * sense_modifier
        else:
            value = base_value
        phase_modifier = max(0.5, min(3.0, _phase_weight(node, phase, phase_weights)))
        mapping[node] = value * phase_modifier
    return mapping


def resolve_nu_f_by_node(
    record: TelemetryRecord,
    *,
    phase: str | None = None,
    phase_weights: Mapping[str, Mapping[str, float] | float] | None = None,
    history: Sequence[TelemetryRecord] | None = None,
    car_model: str | None = None,
    analyzer: NaturalFrequencyAnalyzer | None = None,
    settings: NaturalFrequencySettings | None = None,
    cache_options: "CacheOptions" | None = None,
) -> NaturalFrequencySnapshot:
    """Return the natural frequency snapshot for a telemetry sample.

    When the optional ``analyzer`` or ``history`` arguments are provided the
    result incorporates sliding-window spectral analysis to align the
    instantaneous ν_f values with the dominant excitation frequencies observed
    in the steering, pedal and suspension signals.
    """

    base_map = _base_nu_f_map(record, phase=phase, phase_weights=phase_weights)

    if analyzer is None:
        analyzer = NaturalFrequencyAnalyzer(settings, cache_options=cache_options)

    if history is not None:
        return analyzer.compute_from_history(
            history,
            base_map,
            record=record,
            car_model=car_model,
        )
    return analyzer.update(record, base_map, car_model=car_model)


class _RunningBaseline:
    """Incrementally compute the baseline record statistics."""

    _AVERAGE_FIELDS = (
        "vertical_load",
        "slip_ratio",
        "lateral_accel",
        "longitudinal_accel",
        "yaw",
        "pitch",
        "roll",
        "brake_pressure",
        "locking",
        "nfr",
        "si",
        "speed",
        "yaw_rate",
        "slip_angle",
        "steer",
        "throttle",
        "vertical_load_front",
        "vertical_load_rear",
        "mu_eff_front",
        "mu_eff_rear",
        "mu_eff_front_lateral",
        "mu_eff_front_longitudinal",
        "mu_eff_rear_lateral",
        "mu_eff_rear_longitudinal",
        "suspension_travel_front",
        "suspension_travel_rear",
        "suspension_velocity_front",
        "suspension_velocity_rear",
        "slip_ratio_fl",
        "slip_ratio_fr",
        "slip_ratio_rl",
        "slip_ratio_rr",
        "slip_angle_fl",
        "slip_angle_fr",
        "slip_angle_rl",
        "slip_angle_rr",
    )
    _AVERAGE_INT_FIELDS = ("gear",)

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._count = 0
        self._sums: Dict[str, float] = {field: 0.0 for field in self._AVERAGE_FIELDS}
        for field in self._AVERAGE_INT_FIELDS:
            self._sums[field] = 0.0
        self._first_timestamp: float | None = None
        self._first_structural: float | None = None
        self._car_model: str | None = None
        self._track_name: str | None = None
        self._tyre_compound: str | None = None

    @property
    def count(self) -> int:
        return self._count

    def add(
        self,
        record: TelemetryRecord,
        *,
        car_model: str | None = None,
        track_name: str | None = None,
        tyre_compound: str | None = None,
    ) -> None:
        if self._count == 0:
            self._first_timestamp = float(record.timestamp)
            structural = getattr(record, "structural_timestamp", None)
            if structural is not None and math.isfinite(structural):
                self._first_structural = float(structural)
            else:
                self._first_structural = self._first_timestamp
            self._car_model = car_model or record.car_model
            self._track_name = track_name or record.track_name
            self._tyre_compound = tyre_compound or record.tyre_compound
        else:
            if car_model:
                self._car_model = car_model
            elif record.car_model:
                self._car_model = record.car_model
            if track_name:
                self._track_name = track_name
            elif record.track_name:
                self._track_name = record.track_name
            if tyre_compound:
                self._tyre_compound = tyre_compound
            elif record.tyre_compound:
                self._tyre_compound = record.tyre_compound

        self._count += 1
        for field in self._AVERAGE_FIELDS:
            value = float(getattr(record, field))
            self._accumulate(field, value)
        for field in self._AVERAGE_INT_FIELDS:
            value = float(getattr(record, field))
            self._accumulate(field, value)

    def build(self) -> TelemetryRecord:
        if self._count == 0 or self._first_timestamp is None:
            raise RuntimeError("Cannot build baseline without samples")

        def _average(field: str) -> float:
            total = self._sums.get(field, 0.0)
            if math.isnan(total):
                return math.nan
            return total / float(self._count)

        gear_total = self._sums.get("gear", 0.0)
        if math.isnan(gear_total):
            gear_value = 0
        else:
            gear_value = int(round(gear_total / float(self._count)))

        baseline = TelemetryRecord(
            timestamp=self._first_timestamp,
            structural_timestamp=self._first_structural,
            vertical_load=_average("vertical_load"),
            slip_ratio=_average("slip_ratio"),
            lateral_accel=_average("lateral_accel"),
            longitudinal_accel=_average("longitudinal_accel"),
            yaw=_average("yaw"),
            pitch=_average("pitch"),
            roll=_average("roll"),
            brake_pressure=_average("brake_pressure"),
            locking=_average("locking"),
            nfr=_average("nfr"),
            si=_average("si"),
            speed=_average("speed"),
            yaw_rate=_average("yaw_rate"),
            slip_angle=_average("slip_angle"),
            steer=_average("steer"),
            throttle=_average("throttle"),
            gear=gear_value,
            vertical_load_front=_average("vertical_load_front"),
            vertical_load_rear=_average("vertical_load_rear"),
            mu_eff_front=_average("mu_eff_front"),
            mu_eff_rear=_average("mu_eff_rear"),
            mu_eff_front_lateral=_average("mu_eff_front_lateral"),
            mu_eff_front_longitudinal=_average("mu_eff_front_longitudinal"),
            mu_eff_rear_lateral=_average("mu_eff_rear_lateral"),
            mu_eff_rear_longitudinal=_average("mu_eff_rear_longitudinal"),
            suspension_travel_front=_average("suspension_travel_front"),
            suspension_travel_rear=_average("suspension_travel_rear"),
            suspension_velocity_front=_average("suspension_velocity_front"),
            suspension_velocity_rear=_average("suspension_velocity_rear"),
            slip_ratio_fl=_average("slip_ratio_fl"),
            slip_ratio_fr=_average("slip_ratio_fr"),
            slip_ratio_rl=_average("slip_ratio_rl"),
            slip_ratio_rr=_average("slip_ratio_rr"),
            slip_angle_fl=_average("slip_angle_fl"),
            slip_angle_fr=_average("slip_angle_fr"),
            slip_angle_rl=_average("slip_angle_rl"),
            slip_angle_rr=_average("slip_angle_rr"),
            car_model=self._car_model,
            track_name=self._track_name,
            tyre_compound=self._tyre_compound,
        )
        return baseline

    def _accumulate(self, field: str, value: float) -> None:
        if not math.isfinite(value):
            self._sums[field] = math.nan
            return
        current = self._sums.get(field, 0.0)
        if math.isnan(current):
            return
        self._sums[field] = current + value


class EPIExtractor:
    """Compute EPI bundles for a stream of telemetry records."""

    def __init__(
        self,
        load_weight: float = 0.6,
        slip_weight: float = 0.4,
        *,
        natural_frequency_settings: NaturalFrequencySettings | None = None,
        cache_options: "CacheOptions" | None = None,
    ) -> None:
        if not 0 <= load_weight <= 1:
            raise ValueError("load_weight must be in the 0..1 range")
        if not 0 <= slip_weight <= 1:
            raise ValueError("slip_weight must be in the 0..1 range")
        self.load_weight = load_weight
        self.slip_weight = slip_weight
        self.natural_frequency_settings = (
            natural_frequency_settings or NaturalFrequencySettings()
        )
        self._cache_options = cache_options
        self._nu_f_analyzer = NaturalFrequencyAnalyzer(
            self.natural_frequency_settings, cache_options=cache_options
        )
        self._baseline = _RunningBaseline()
        self._baseline_record: TelemetryRecord | None = None
        self._prev_integrated_epi: Optional[float] = None
        self._prev_timestamp: float | None = None
        self._prev_structural: float | None = None
        self._batching_calibration: bool = False
        self._batched_calibration: tuple[
            "CoherenceCalibrationStore",
            str,
            str,
        ] | None = None
        self._pending_baseline: TelemetryRecord | None = None

    def reset(self) -> None:
        """Clear the incremental state."""

        self._nu_f_analyzer.reset()
        self._baseline.reset()
        self._baseline_record = None
        self._prev_integrated_epi = None
        self._prev_timestamp = None
        self._prev_structural = None
        self._batched_calibration = None
        self._pending_baseline = None
        self._batching_calibration = False

    def update(
        self,
        record: TelemetryRecord,
        *,
        calibration: "CoherenceCalibrationStore" | None = None,
        player_name: str | None = None,
        car_model: str | None = None,
        track_name: str | None = None,
        tyre_compound: str | None = None,
    ) -> EPIBundle:
        """Process a single telemetry record and return the resulting bundle."""

        self._baseline.add(
            record,
            car_model=car_model,
            track_name=track_name,
            tyre_compound=tyre_compound,
        )
        baseline = self._resolve_baseline(
            calibration=calibration,
            player_name=player_name,
            car_model=car_model,
        )

        epi_value = self._compute_epi(record)
        structural_ts = getattr(record, "structural_timestamp", None)

        if self._prev_timestamp is None:
            dt = 0.0
        else:
            if (
                structural_ts is not None
                and self._prev_structural is not None
                and math.isfinite(structural_ts)
                and math.isfinite(self._prev_structural)
            ):
                dt = max(0.0, structural_ts - self._prev_structural)
            else:
                dt = max(0.0, record.timestamp - self._prev_timestamp)

        nu_snapshot = resolve_nu_f_by_node(
            record,
            analyzer=self._nu_f_analyzer,
            car_model=car_model or record.car_model,
            cache_options=self._cache_options,
        )
        bundle = DeltaCalculator.compute_bundle(
            record,
            baseline,
            epi_value,
            prev_integrated_epi=self._prev_integrated_epi,
            dt=dt,
            nu_f_by_node=nu_snapshot.by_node,
            nu_f_snapshot=nu_snapshot,
        )

        self._baseline_record = baseline
        self._prev_integrated_epi = bundle.integrated_epi
        self._prev_timestamp = record.timestamp
        if structural_ts is not None and math.isfinite(structural_ts):
            self._prev_structural = structural_ts

        return bundle

    def extract(
        self,
        records: Sequence[TelemetryRecord],
        *,
        calibration: "CoherenceCalibrationStore" | None = None,
        player_name: str | None = None,
        car_model: str | None = None,
    ) -> List[EPIBundle]:
        if not records:
            return []
        self.reset()
        bundles: List[EPIBundle] = []
        batching = calibration is not None and player_name is not None
        if batching:
            self._batching_calibration = True
            self._batched_calibration = None
            self._pending_baseline = None
        try:
            for record in records:
                bundle = self.update(
                    record,
                    calibration=calibration,
                    player_name=player_name,
                    car_model=car_model or record.car_model,
                    track_name=record.track_name,
                    tyre_compound=record.tyre_compound,
                )
                bundles.append(bundle)
        finally:
            if batching and self._batched_calibration and self._pending_baseline:
                calibration.observe_baseline(
                    self._batched_calibration[1],
                    self._batched_calibration[2],
                    self._pending_baseline,
                )
            self._batching_calibration = False
            self._batched_calibration = None
            self._pending_baseline = None
        return bundles

    def _compute_epi(self, record: TelemetryRecord) -> float:
        # Normalise vertical load between 0 and 10 kN which is a typical
        # race car range.  Slip ratio is expected in -1..1.
        load_component = min(max(record.vertical_load / 10000.0, 0.0), 1.0)
        slip_component = min(max((record.slip_ratio + 1.0) / 2.0, 0.0), 1.0)
        return (load_component * self.load_weight) + (slip_component * self.slip_weight)

    def _resolve_baseline(
        self,
        *,
        calibration: "CoherenceCalibrationStore" | None,
        player_name: str | None,
        car_model: str | None,
    ) -> TelemetryRecord:
        raw_baseline = self._baseline.build()
        if calibration is None or not player_name or not (car_model or raw_baseline.car_model):
            self._baseline_record = raw_baseline
            return raw_baseline
        resolved_car = car_model or raw_baseline.car_model
        assert resolved_car is not None  # for type checkers
        resolved = calibration.baseline_for(player_name, resolved_car, raw_baseline)
        if self._batching_calibration:
            self._batched_calibration = (calibration, player_name, resolved_car)
            self._pending_baseline = raw_baseline
        else:
            calibration.observe_baseline(player_name, resolved_car, raw_baseline)
        self._baseline_record = resolved
        return resolved


class DeltaCalculator:
    """Compute delta metrics relative to a baseline."""

    @staticmethod
    def derive_baseline(records: Sequence[TelemetryRecord]) -> TelemetryRecord:
        """Return a synthetic baseline record representing the average state."""

        # Collate the continuous telemetry channels into a single NumPy array so
        # the baseline statistics can be obtained with a single vectorised
        # ``np.mean`` call.  ``tests/test_epi.py::test_vectorised_baseline``
        # asserts these aggregated means remain compatible with the historical
        # scalar behaviour.
        float_fields = (
            "vertical_load",
            "slip_ratio",
            "slip_ratio_fl",
            "slip_ratio_fr",
            "slip_ratio_rl",
            "slip_ratio_rr",
            "lateral_accel",
            "longitudinal_accel",
            "yaw",
            "pitch",
            "roll",
            "brake_pressure",
            "locking",
            "nfr",
            "si",
            "speed",
            "yaw_rate",
            "slip_angle",
            "slip_angle_fl",
            "slip_angle_fr",
            "slip_angle_rl",
            "slip_angle_rr",
            "steer",
            "throttle",
            "vertical_load_front",
            "vertical_load_rear",
            "mu_eff_front",
            "mu_eff_rear",
            "mu_eff_front_lateral",
            "mu_eff_front_longitudinal",
            "mu_eff_rear_lateral",
            "mu_eff_rear_longitudinal",
            "suspension_travel_front",
            "suspension_travel_rear",
            "suspension_velocity_front",
            "suspension_velocity_rear",
        )
        float_samples = np.array(
            [[getattr(record, field) for field in float_fields] for record in records],
            dtype=float,
        )
        float_means = np.mean(float_samples, axis=0)

        baseline_kwargs = dict(zip(float_fields, float_means))
        gear_values = np.fromiter((record.gear for record in records), dtype=float)
        baseline_kwargs["gear"] = int(np.rint(np.mean(gear_values, axis=0)))

        return TelemetryRecord(
            timestamp=records[0].timestamp,
            structural_timestamp=(
                records[0].structural_timestamp
                if getattr(records[0], "structural_timestamp", None) is not None
                else records[0].timestamp
            ),
            car_model=getattr(records[0], "car_model", None),
            track_name=getattr(records[0], "track_name", None),
            tyre_compound=getattr(records[0], "tyre_compound", None),
            **baseline_kwargs,
        )

    @staticmethod
    def resolve_baseline(
        records: Sequence[TelemetryRecord],
        *,
        calibration: "CoherenceCalibrationStore" | None = None,
        player_name: str | None = None,
        car_model: str | None = None,
    ) -> TelemetryRecord:
        baseline = DeltaCalculator.derive_baseline(records)
        if calibration is None or not player_name or not car_model:
            return baseline
        resolved = calibration.baseline_for(player_name, car_model, baseline)
        calibration.observe_baseline(player_name, car_model, baseline)
        return resolved

    @staticmethod
    def compute_bundle(
        record: TelemetryRecord,
        baseline: TelemetryRecord,
        epi_value: float,
        *,
        prev_integrated_epi: Optional[float] = None,
        dt: float = 0.0,
        nu_f_by_node: Optional[Mapping[str, float]] = None,
        nu_f_snapshot: NaturalFrequencySnapshot | None = None,
        phase: str = "entry",
        phase_weights: Optional[Mapping[str, Mapping[str, float] | float]] = None,
        phase_target_nu_f: Mapping[str, Mapping[str, float] | float]
        | Mapping[str, float]
        | float
        | None = None,
    ) -> EPIBundle:
        structural_timestamp = getattr(record, "structural_timestamp", None)
        delta_nfr = record.nfr - baseline.nfr
        feature_contributions = _node_feature_contributions(record, baseline)
        node_signals = {
            node: sum(values.values()) for node, values in feature_contributions.items()
        }
        axis_signals = _axis_signal_components(record, baseline, feature_contributions)
        node_deltas = _distribute_node_delta(delta_nfr, node_signals)
        if nu_f_snapshot is None:
            nu_f_snapshot = resolve_nu_f_by_node(record)
        nu_f_map = dict(nu_f_by_node or nu_f_snapshot.by_node)
        phase_weight_map = phase_weights or DEFAULT_PHASE_WEIGHTS
        global_si = sense_index(
            delta_nfr,
            node_deltas,
            baseline.nfr,
            nu_f_by_node=nu_f_map,
            active_phase=phase,
            w_phase=phase_weight_map,
            nu_f_targets=phase_target_nu_f,
        )
        previous_state = epi_value if prev_integrated_epi is None else prev_integrated_epi
        try:
            from .operators import evolve_epi
        except ImportError:  # pragma: no cover - defensive fallback during circular import
            def evolve_epi(prev_epi: float, delta_map: Mapping[str, float], dt: float, nu_map: Mapping[str, float]):
                nodal: Dict[str, tuple[float, float]] = {}
                derivative = 0.0
                for node in set(delta_map) | set(nu_map):
                    node_derivative = nu_map.get(node, 0.0) * delta_map.get(node, 0.0)
                    nodal[node] = (node_derivative * dt, node_derivative)
                    derivative += node_derivative
                return prev_epi + (derivative * dt), derivative, nodal

        integrated_epi, derivative, nodal_evolution = evolve_epi(previous_state, node_deltas, dt, nu_f_map)
        delta_breakdown = {
            node: compute_node_delta_nfr(node, node_deltas.get(node, 0.0), features, prefix=False)
            for node, features in feature_contributions.items()
        }
        axis_total = axis_signals["longitudinal"] + axis_signals["lateral"]
        if axis_total > 1e-9 and math.isfinite(axis_total):
            longitudinal_delta = delta_nfr * (axis_signals["longitudinal"] / axis_total)
            lateral_delta = delta_nfr * (axis_signals["lateral"] / axis_total)
        else:
            longitudinal_delta = 0.0
            lateral_delta = 0.0
        nodes = DeltaCalculator._build_nodes(
            record, node_deltas, delta_nfr, nu_f_map, nodal_evolution
        )
        ackermann_delta = _ackermann_parallel_delta(record, baseline)
        return EPIBundle(
            timestamp=record.timestamp,
            structural_timestamp=(
                float(structural_timestamp)
                if structural_timestamp is not None
                else float(record.timestamp)
            ),
            epi=epi_value,
            delta_nfr=delta_nfr,
            delta_nfr_proj_longitudinal=longitudinal_delta,
            delta_nfr_proj_lateral=lateral_delta,
            sense_index=global_si,
            delta_breakdown=delta_breakdown,
            dEPI_dt=derivative,
            integrated_epi=integrated_epi,
            node_evolution=dict(nodal_evolution),
            tyres=nodes["tyres"],
            suspension=nodes["suspension"],
            chassis=nodes["chassis"],
            brakes=nodes["brakes"],
            transmission=nodes["transmission"],
            track=nodes["track"],
            driver=nodes["driver"],
            nu_f_classification=nu_f_snapshot.classification,
            nu_f_category=nu_f_snapshot.category,
            nu_f_label=nu_f_snapshot.frequency_label,
            nu_f_dominant=nu_f_snapshot.dominant_frequency,
            coherence_index=nu_f_snapshot.coherence_index,
            ackermann_parallel_index=ackermann_delta,
        )

    @staticmethod
    def _build_nodes(
        record: TelemetryRecord,
        node_deltas: Mapping[str, float],
        delta_nfr: float,
        nu_f_by_node: Mapping[str, float],
        node_evolution: Mapping[str, tuple[float, float]] | None,
    ) -> Dict[str, object]:
        node_evolution = node_evolution or {}

        def node_si(node_delta: float) -> float:
            if abs(delta_nfr) < 1e-9:
                return 1.0
            ratio = min(1.0, abs(node_delta) / (abs(delta_nfr) + 1e-9))
            return max(0.0, min(1.0, 1.0 - ratio))

        def node_state(node: str) -> tuple[float, float]:
            return node_evolution.get(node, (0.0, 0.0))

        brake_values = [
            float(record.brake_temp_fl),
            float(record.brake_temp_fr),
            float(record.brake_temp_rl),
            float(record.brake_temp_rr),
        ]
        brake_finite = [
            value for value in brake_values if math.isfinite(value) and value > 0.0
        ]
        brake_peak = max(brake_finite) if brake_finite else 0.0
        brake_mean = mean(brake_finite) if brake_finite else 0.0

        return {
            "tyres": TyresNode(
                delta_nfr=node_deltas.get("tyres", 0.0),
                sense_index=node_si(node_deltas.get("tyres", 0.0)),
                nu_f=nu_f_by_node.get("tyres", 0.0),
                integrated_epi=node_state("tyres")[0],
                dEPI_dt=node_state("tyres")[1],
                load=record.vertical_load,
                slip_ratio=record.slip_ratio,
                mu_eff_front=record.mu_eff_front,
                mu_eff_rear=record.mu_eff_rear,
                mu_eff_front_lateral=record.mu_eff_front_lateral,
                mu_eff_front_longitudinal=record.mu_eff_front_longitudinal,
                mu_eff_rear_lateral=record.mu_eff_rear_lateral,
                mu_eff_rear_longitudinal=record.mu_eff_rear_longitudinal,
                tyre_temp_fl=record.tyre_temp_fl,
                tyre_temp_fr=record.tyre_temp_fr,
                tyre_temp_rl=record.tyre_temp_rl,
                tyre_temp_rr=record.tyre_temp_rr,
                tyre_temp_fl_inner=record.tyre_temp_fl_inner,
                tyre_temp_fr_inner=record.tyre_temp_fr_inner,
                tyre_temp_rl_inner=record.tyre_temp_rl_inner,
                tyre_temp_rr_inner=record.tyre_temp_rr_inner,
                tyre_temp_fl_middle=record.tyre_temp_fl_middle,
                tyre_temp_fr_middle=record.tyre_temp_fr_middle,
                tyre_temp_rl_middle=record.tyre_temp_rl_middle,
                tyre_temp_rr_middle=record.tyre_temp_rr_middle,
                tyre_temp_fl_outer=record.tyre_temp_fl_outer,
                tyre_temp_fr_outer=record.tyre_temp_fr_outer,
                tyre_temp_rl_outer=record.tyre_temp_rl_outer,
                tyre_temp_rr_outer=record.tyre_temp_rr_outer,
                tyre_pressure_fl=record.tyre_pressure_fl,
                tyre_pressure_fr=record.tyre_pressure_fr,
                tyre_pressure_rl=record.tyre_pressure_rl,
                tyre_pressure_rr=record.tyre_pressure_rr,
            ),
            "suspension": SuspensionNode(
                delta_nfr=node_deltas.get("suspension", 0.0),
                sense_index=node_si(node_deltas.get("suspension", 0.0)),
                nu_f=nu_f_by_node.get("suspension", 0.0),
                integrated_epi=node_state("suspension")[0],
                dEPI_dt=node_state("suspension")[1],
                travel_front=record.suspension_travel_front,
                travel_rear=record.suspension_travel_rear,
                velocity_front=record.suspension_velocity_front,
                velocity_rear=record.suspension_velocity_rear,
            ),
            "chassis": ChassisNode(
                delta_nfr=node_deltas.get("chassis", 0.0),
                sense_index=node_si(node_deltas.get("chassis", 0.0)),
                nu_f=nu_f_by_node.get("chassis", 0.0),
                integrated_epi=node_state("chassis")[0],
                dEPI_dt=node_state("chassis")[1],
                yaw=record.yaw,
                pitch=record.pitch,
                roll=record.roll,
                yaw_rate=record.yaw_rate,
                lateral_accel=record.lateral_accel,
                longitudinal_accel=record.longitudinal_accel,
            ),
            "brakes": BrakesNode(
                delta_nfr=node_deltas.get("brakes", 0.0),
                sense_index=node_si(node_deltas.get("brakes", 0.0)),
                nu_f=nu_f_by_node.get("brakes", 0.0),
                integrated_epi=node_state("brakes")[0],
                dEPI_dt=node_state("brakes")[1],
                brake_pressure=record.brake_pressure,
                locking=record.locking,
                brake_temp_fl=record.brake_temp_fl,
                brake_temp_fr=record.brake_temp_fr,
                brake_temp_rl=record.brake_temp_rl,
                brake_temp_rr=record.brake_temp_rr,
                brake_temp_peak=brake_peak,
                brake_temp_mean=brake_mean,
            ),
            "transmission": TransmissionNode(
                delta_nfr=node_deltas.get("transmission", 0.0),
                sense_index=node_si(node_deltas.get("transmission", 0.0)),
                nu_f=nu_f_by_node.get("transmission", 0.0),
                integrated_epi=node_state("transmission")[0],
                dEPI_dt=node_state("transmission")[1],
                throttle=record.throttle,
                gear=record.gear,
                speed=record.speed,
                longitudinal_accel=record.longitudinal_accel,
                rpm=record.rpm,
                line_deviation=record.line_deviation,
            ),
            "track": TrackNode(
                delta_nfr=node_deltas.get("track", 0.0),
                sense_index=node_si(node_deltas.get("track", 0.0)),
                nu_f=nu_f_by_node.get("track", 0.0),
                integrated_epi=node_state("track")[0],
                dEPI_dt=node_state("track")[1],
                axle_load_balance=record.vertical_load_front - record.vertical_load_rear,
                axle_velocity_balance=
                    record.suspension_velocity_front - record.suspension_velocity_rear,
                yaw=record.yaw,
                lateral_accel=record.lateral_accel,
                gradient=_resolve_track_gradient(record),
            ),
            "driver": DriverNode(
                delta_nfr=node_deltas.get("driver", 0.0),
                sense_index=node_si(node_deltas.get("driver", 0.0)),
                nu_f=nu_f_by_node.get("driver", 0.0),
                integrated_epi=node_state("driver")[0],
                dEPI_dt=node_state("driver")[1],
                steer=record.steer,
                throttle=record.throttle,
                style_index=record.si,
            ),
        }
