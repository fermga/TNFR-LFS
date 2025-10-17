"""EPI extraction and ΔNFR/ΔSi computations."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from importlib import import_module
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from tnfr_core.runtime.shared import (
    CacheOptions,
    SupportsTelemetrySample,
    cached_dynamic_multipliers,
    invalidate_dynamic_record,
    should_use_dynamic_cache,
)
from tnfr_core.equations.baseline import (
    BaselineResolver,
    DeltaCalculator,
    DEFAULT_PHASE_WEIGHTS,
    _ackermann_parallel_delta,
    _delta_nfr_by_node_uncached,
    compute_delta_nfr,
    delta_nfr_by_node,
)
from tnfr_core.equations.epi_models import (
    BrakesNode,
    ChassisNode,
    DriverNode,
    EPIBundle,
    SuspensionNode,
    TrackNode,
    TransmissionNode,
    TyresNode,
)
from tnfr_core.equations.phase_weights import _phase_weight
from tnfr_core.equations.telemetry import TelemetryRecord, _MISSING_FLOAT
from tnfr_core._canonical import CANONICAL_REQUESTED, import_tnfr
from tnfr_core.equations.epi_evolution import evolve_epi
__all__ = [
    "TelemetryRecord",
    "EPIExtractor",
    "NaturalFrequencyAnalyzer",
    "NaturalFrequencySnapshot",
    "NaturalFrequencySettings",
    "DEFAULT_PHASE_WEIGHTS",
    "DeltaCalculator",
    "compute_delta_nfr",
    "delta_nfr_by_node",
    "_ackermann_parallel_delta",
    "_delta_nfr_by_node_uncached",
]

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

_AVERAGE_NU_F = sum(NU_F_NODE_DEFAULTS.values()) / float(len(NU_F_NODE_DEFAULTS))


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
        self._history: Deque[SupportsTelemetrySample] = deque()
        self._smoothed: Dict[str, float] = {}
        self._last_car: str | None = None
        self._last_snapshot: NaturalFrequencySnapshot | None = None
        self._cache_options = cache_options

    def _dynamic_cache_active(self) -> bool:
        return should_use_dynamic_cache(self._cache_options)

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
        record: SupportsTelemetrySample,
        base_map: Mapping[str, float],
        *,
        car_model: str | None = None,
    ) -> NaturalFrequencySnapshot:
        self._append_record(record)
        mapping, dominant_frequency = self._resolve(base_map, car_model)
        return self._build_snapshot(mapping, dominant_frequency, car_model)

    def compute_from_history(
        self,
        history: Sequence[SupportsTelemetrySample],
        base_map: Mapping[str, float],
        *,
        record: SupportsTelemetrySample | None = None,
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

    def _append_record(self, record: SupportsTelemetrySample) -> None:
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
        if not isinstance(history, tuple):
            history = tuple(history)
        if len(history) < 2:
            return {}, 0.0

        duration = history[-1].timestamp - history[0].timestamp
        if duration < max(0.0, self.settings.min_window_seconds - 1e-6):
            return {}, 0.0

        from tnfr_core.signal.spectrum import (
            cross_spectrum,
            estimate_sample_rate,
            power_spectrum,
        )

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
        if len(self._history) < 2:
            return 0.0
        window = max(1, int(self.settings.structural_density_window))
        densities: Deque[float] = deque(maxlen=window)
        iterator = iter(self._history)
        previous = next(iterator, None)
        for current in iterator:
            if previous is None:
                previous = current
                continue
            chrono_dt = float(current.timestamp) - float(previous.timestamp)
            if chrono_dt <= 1e-9:
                previous = current
                continue
            structural_curr = getattr(current, "structural_timestamp", None)
            structural_prev = getattr(previous, "structural_timestamp", None)
            if structural_curr is None or structural_prev is None:
                previous = current
                continue
            structural_dt = float(structural_curr) - float(structural_prev)
            if structural_dt <= 0.0:
                previous = current
                continue
            ratio = structural_dt / chrono_dt
            densities.append(max(0.0, ratio - 1.0))
            previous = current
        if not densities:
            return 0.0
        # The bounded deque ensures we only retain the recent values needed for the
        # trailing average, matching the previous behaviour without materialising
        # the entire history on every coherence calculation.
        average_density = sum(densities) / len(densities)
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




def _base_nu_f_map(
    record: SupportsTelemetrySample,
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
    record: SupportsTelemetrySample,
    *,
    phase: str | None = None,
    phase_weights: Mapping[str, Mapping[str, float] | float] | None = None,
    history: Sequence[SupportsTelemetrySample] | None = None,
    car_model: str | None = None,
    analyzer: NaturalFrequencyAnalyzer | None = None,
    settings: NaturalFrequencySettings | None = None,
    cache_options: CacheOptions | None = None,
) -> NaturalFrequencySnapshot:
    """Return the natural frequency snapshot for a telemetry sample."""

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
        record: SupportsTelemetrySample,
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
            record_car = getattr(record, "car_model", None)
            record_track = getattr(record, "track_name", None)
            record_tyre = getattr(record, "tyre_compound", None)
            self._car_model = car_model or record_car
            self._track_name = track_name or record_track
            self._tyre_compound = tyre_compound or record_tyre
        else:
            record_car = getattr(record, "car_model", None)
            record_track = getattr(record, "track_name", None)
            record_tyre = getattr(record, "tyre_compound", None)
            if car_model:
                self._car_model = car_model
            elif record_car:
                self._car_model = record_car
            if track_name:
                self._track_name = track_name
            elif record_track:
                self._track_name = record_track
            if tyre_compound:
                self._tyre_compound = tyre_compound
            elif record_tyre:
                self._tyre_compound = record_tyre

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

    @property
    def baseline_record(self) -> TelemetryRecord | None:
        """Return the most recent baseline resolved during extraction."""

        return self._baseline_record

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
        record: SupportsTelemetrySample,
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
            car_model=car_model or getattr(record, "car_model", None),
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
        records: Sequence[SupportsTelemetrySample],
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
                    car_model=car_model or getattr(record, "car_model", None),
                    track_name=getattr(record, "track_name", None),
                    tyre_compound=getattr(record, "tyre_compound", None),
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

    def _compute_epi(self, record: SupportsTelemetrySample) -> float:
        # Normalise vertical load between 0 and 10 kN which is a typical
        # race car range.  Slip ratio is expected in -1..1.
        load_component = min(max(record.vertical_load / 10000.0, 0.0), 1.0)
        slip_component = min(max((record.slip_ratio + 1.0) / 2.0, 0.0), 1.0)
        return (load_component * self.load_weight) + (slip_component * self.slip_weight)

    def _resolve_baseline(
        self,
        *,
        calibration: BaselineResolver | None,
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


if CANONICAL_REQUESTED:  # pragma: no cover - depends on optional package
    tnfr = import_tnfr()
    canonical_epi = import_module(f"{tnfr.__name__}.equations.epi")

    canonical_exports = getattr(canonical_epi, "__all__", None)
    if canonical_exports is not None:
        __all__ = list(dict.fromkeys([*canonical_exports, *__all__]))

    for name in dir(canonical_epi):
        if name.startswith("__"):
            continue
        globals()[name] = getattr(canonical_epi, name)
