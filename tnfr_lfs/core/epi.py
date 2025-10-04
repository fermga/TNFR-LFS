"""EPI extraction and ΔNFR/ΔSi computations."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import mean
from collections.abc import Mapping as MappingABC
from typing import Deque, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .coherence_calibration import CoherenceCalibrationStore

from .coherence import compute_node_delta_nfr, sense_index
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
# "modelos de activación" tables in the manual.  They provide the base
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

    def resolve_vehicle_frequency(self, car_model: str | None) -> float:
        if car_model and car_model in self.vehicle_frequency:
            return max(1e-6, float(self.vehicle_frequency[car_model]))
        if "__default__" in self.vehicle_frequency:
            return max(1e-6, float(self.vehicle_frequency["__default__"]))
        return max(1e-6, _AVERAGE_NU_F)


class NaturalFrequencyAnalyzer:
    """Track dominant frequencies over a sliding telemetry window."""

    def __init__(self, settings: NaturalFrequencySettings | None = None) -> None:
        self.settings = settings or NaturalFrequencySettings()
        self._history: Deque[TelemetryRecord] = deque()
        self._smoothed: Dict[str, float] = {}
        self._last_car: str | None = None

    def reset(self) -> None:
        self._history.clear()
        self._smoothed.clear()
        self._last_car = None

    def update(
        self,
        record: TelemetryRecord,
        base_map: Mapping[str, float],
        *,
        car_model: str | None = None,
    ) -> Dict[str, float]:
        self._append_record(record)
        return self._resolve(base_map, car_model)

    def compute_from_history(
        self,
        history: Sequence[TelemetryRecord],
        base_map: Mapping[str, float],
        *,
        record: TelemetryRecord | None = None,
        car_model: str | None = None,
    ) -> Dict[str, float]:
        self._history.clear()
        for sample in history:
            self._append_record(sample)
        if record is not None and (not self._history or self._history[-1] is not record):
            self._append_record(record)
        return self._resolve(base_map, car_model)

    def _append_record(self, record: TelemetryRecord) -> None:
        self._history.append(record)
        max_window = max(self.settings.max_window_seconds, self.settings.min_window_seconds)
        while (
            len(self._history) >= 2
            and (self._history[-1].timestamp - self._history[0].timestamp) > max_window
        ):
            self._history.popleft()

    def _resolve(
        self, base_map: Mapping[str, float], car_model: str | None
    ) -> Dict[str, float]:
        if car_model != self._last_car:
            self._smoothed.clear()
            self._last_car = car_model

        dynamic = self._dynamic_multipliers(car_model)
        if not dynamic:
            # No dynamic adjustment available, fall back to smoothing defaults.
            return self._apply_smoothing(dict(base_map))

        adjusted = {}
        for node, value in base_map.items():
            multiplier = dynamic.get(node, 1.0)
            adjusted[node] = value * multiplier
        return self._apply_smoothing(adjusted)

    def _dynamic_multipliers(self, car_model: str | None) -> Dict[str, float]:
        history = list(self._history)
        if len(history) < 2:
            return {}

        duration = history[-1].timestamp - history[0].timestamp
        if duration < max(0.0, self.settings.min_window_seconds - 1e-6):
            return {}

        from .spectrum import cross_spectrum, power_spectrum, estimate_sample_rate

        sample_rate = estimate_sample_rate(history)
        if sample_rate <= 0.0:
            return {}

        min_samples = max(4, int(self.settings.min_window_seconds * sample_rate))
        if len(history) < min_samples:
            return {}

        steer_series = [float(record.steer) for record in history]
        throttle_series = [float(record.throttle) for record in history]
        brake_series = [float(record.brake_pressure) for record in history]
        suspension_front = [float(record.suspension_velocity_front) for record in history]
        suspension_rear = [float(record.suspension_velocity_rear) for record in history]
        suspension_combined = [
            (front + rear) * 0.5 for front, rear in zip(suspension_front, suspension_rear)
        ]

        low = max(0.0, self.settings.bandpass_low_hz)
        high = max(low, self.settings.bandpass_high_hz)

        def _dominant_frequency(series: Sequence[float]) -> float:
            spectrum = power_spectrum(series, sample_rate)
            band = [entry for entry in spectrum if low <= entry[0] <= high]
            if not band:
                return 0.0
            frequency, energy = max(band, key=lambda entry: entry[1])
            if energy <= 1e-9:
                return 0.0
            return frequency

        def _dominant_cross(x_series: Sequence[float], y_series: Sequence[float]) -> float:
            spectrum = cross_spectrum(x_series, y_series, sample_rate)
            band = [entry for entry in spectrum if low <= entry[0] <= high]
            if not band:
                return 0.0
            frequency, real, imag = max(
                band, key=lambda entry: math.hypot(entry[1], entry[2])
            )
            magnitude = math.hypot(real, imag)
            if magnitude <= 1e-9:
                return 0.0
            return frequency

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

        multipliers: Dict[str, float] = {}
        if steer_freq > 0.0:
            multipliers["driver"] = _normalise(steer_freq)
        if throttle_freq > 0.0:
            multipliers["transmission"] = _normalise(throttle_freq)
        if brake_freq > 0.0:
            multipliers["brakes"] = _normalise(brake_freq)
        if suspension_freq > 0.0:
            value = _normalise(suspension_freq)
            multipliers["suspension"] = value
            multipliers.setdefault("chassis", value)
        if tyre_freq > 0.0:
            value = _normalise(tyre_freq)
            multipliers["tyres"] = value
            multipliers["chassis"] = value
        elif suspension_freq > 0.0 and steer_freq > 0.0:
            blended = (multipliers["suspension"] + multipliers["driver"]) * 0.5
            multipliers["tyres"] = blended

        return multipliers

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
    structural_timestamp: float | None = None
    tyre_temp_fl: float = 0.0
    tyre_temp_fr: float = 0.0
    tyre_temp_rl: float = 0.0
    tyre_temp_rr: float = 0.0
    tyre_pressure_fl: float = 0.0
    tyre_pressure_fr: float = 0.0
    tyre_pressure_rl: float = 0.0
    tyre_pressure_rr: float = 0.0
    rpm: float = 0.0
    line_deviation: float = 0.0
    lap: int | str | None = None
    reference: Optional["TelemetryRecord"] = None


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
    total_signal = sum(node_signals.values())
    if total_signal <= 1e-9 or not math.isfinite(total_signal):
        node_count = len(node_signals)
        if node_count == 0:
            return {}
        uniform_share = delta_nfr / float(node_count)
        return {node: uniform_share for node in node_signals}

    magnitude = abs(delta_nfr)
    sign = 1.0 if delta_nfr >= 0.0 else -1.0
    return {
        node: sign * magnitude * (signal / total_signal)
        for node, signal in node_signals.items()
    }


def delta_nfr_by_node(record: TelemetryRecord) -> Mapping[str, float]:
    """Compute ΔNFR contributions for each subsystem.

    The function expects ``record`` to optionally provide a ``reference``
    sample, typically the baseline derived from the telemetry stint.  When a
    reference is available the signal strength for every subsystem is
    measured relative to it, otherwise the calculation degenerates into a
    uniform distribution.
    """

    baseline = record.reference or record
    delta_nfr = record.nfr - baseline.nfr
    feature_contributions = _node_feature_contributions(record, baseline)
    node_signals = {
        node: sum(values.values()) for node, values in feature_contributions.items()
    }
    return _distribute_node_delta(delta_nfr, node_signals)


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
) -> Dict[str, float]:
    """Return the natural frequency per node for a telemetry sample.

    When the optional ``analyzer`` or ``history`` arguments are provided the
    result incorporates sliding-window spectral analysis to align the
    instantaneous ν_f values with the dominant excitation frequencies observed
    in the steering, pedal and suspension signals.
    """

    base_map = _base_nu_f_map(record, phase=phase, phase_weights=phase_weights)

    if analyzer is None:
        if settings is not None:
            analyzer = NaturalFrequencyAnalyzer(settings)
        elif history is not None:
            analyzer = NaturalFrequencyAnalyzer()
        else:
            return dict(base_map)

    if history is not None:
        return analyzer.compute_from_history(
            history,
            base_map,
            record=record,
            car_model=car_model,
        )
    return analyzer.update(record, base_map, car_model=car_model)


class EPIExtractor:
    """Compute EPI bundles for a stream of telemetry records."""

    def __init__(
        self,
        load_weight: float = 0.6,
        slip_weight: float = 0.4,
        *,
        natural_frequency_settings: NaturalFrequencySettings | None = None,
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
        self._nu_f_analyzer = NaturalFrequencyAnalyzer(self.natural_frequency_settings)

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
        self._nu_f_analyzer.reset()
        baseline = DeltaCalculator.resolve_baseline(
            records,
            calibration=calibration,
            player_name=player_name,
            car_model=car_model,
        )
        results: List[EPIBundle] = []
        prev_integrated_epi: Optional[float] = None
        prev_timestamp = records[0].timestamp
        prev_structural = getattr(records[0], "structural_timestamp", None)
        for index, record in enumerate(records):
            epi_value = self._compute_epi(record)
            structural_ts = getattr(record, "structural_timestamp", None)
            if index == 0:
                dt = 0.0
            else:
                if (
                    structural_ts is not None
                    and prev_structural is not None
                    and math.isfinite(structural_ts)
                    and math.isfinite(prev_structural)
                ):
                    dt = max(0.0, structural_ts - prev_structural)
                else:
                    dt = max(0.0, record.timestamp - prev_timestamp)
            nu_f_map = resolve_nu_f_by_node(
                record,
                analyzer=self._nu_f_analyzer,
                car_model=car_model,
            )
            bundle = DeltaCalculator.compute_bundle(
                record,
                baseline,
                epi_value,
                prev_integrated_epi=prev_integrated_epi,
                dt=dt,
                nu_f_by_node=nu_f_map,
            )
            results.append(bundle)
            prev_integrated_epi = bundle.integrated_epi
            prev_timestamp = record.timestamp
            if structural_ts is not None and math.isfinite(structural_ts):
                prev_structural = structural_ts
        return results

    def _compute_epi(self, record: TelemetryRecord) -> float:
        # Normalise vertical load between 0 and 10 kN which is a typical
        # race car range.  Slip ratio is expected in -1..1.
        load_component = min(max(record.vertical_load / 10000.0, 0.0), 1.0)
        slip_component = min(max((record.slip_ratio + 1.0) / 2.0, 0.0), 1.0)
        return (load_component * self.load_weight) + (slip_component * self.slip_weight)


class DeltaCalculator:
    """Compute delta metrics relative to a baseline."""

    @staticmethod
    def derive_baseline(records: Sequence[TelemetryRecord]) -> TelemetryRecord:
        """Return a synthetic baseline record representing the average state."""

        return TelemetryRecord(
            timestamp=records[0].timestamp,
            structural_timestamp=(
                records[0].structural_timestamp
                if getattr(records[0], "structural_timestamp", None) is not None
                else records[0].timestamp
            ),
            vertical_load=mean(record.vertical_load for record in records),
            slip_ratio=mean(record.slip_ratio for record in records),
            lateral_accel=mean(record.lateral_accel for record in records),
            longitudinal_accel=mean(record.longitudinal_accel for record in records),
            yaw=mean(record.yaw for record in records),
            pitch=mean(record.pitch for record in records),
            roll=mean(record.roll for record in records),
            brake_pressure=mean(record.brake_pressure for record in records),
            locking=mean(record.locking for record in records),
            nfr=mean(record.nfr for record in records),
            si=mean(record.si for record in records),
            speed=mean(record.speed for record in records),
            yaw_rate=mean(record.yaw_rate for record in records),
            slip_angle=mean(record.slip_angle for record in records),
            steer=mean(record.steer for record in records),
            throttle=mean(record.throttle for record in records),
            gear=int(round(mean(record.gear for record in records))),
            vertical_load_front=mean(record.vertical_load_front for record in records),
            vertical_load_rear=mean(record.vertical_load_rear for record in records),
            mu_eff_front=mean(record.mu_eff_front for record in records),
            mu_eff_rear=mean(record.mu_eff_rear for record in records),
            mu_eff_front_lateral=mean(
                record.mu_eff_front_lateral for record in records
            ),
            mu_eff_front_longitudinal=mean(
                record.mu_eff_front_longitudinal for record in records
            ),
            mu_eff_rear_lateral=mean(record.mu_eff_rear_lateral for record in records),
            mu_eff_rear_longitudinal=mean(
                record.mu_eff_rear_longitudinal for record in records
            ),
            suspension_travel_front=mean(record.suspension_travel_front for record in records),
            suspension_travel_rear=mean(record.suspension_travel_rear for record in records),
            suspension_velocity_front=mean(record.suspension_velocity_front for record in records),
            suspension_velocity_rear=mean(record.suspension_velocity_rear for record in records),
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
        nu_f_map = dict(nu_f_by_node or resolve_nu_f_by_node(record))
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
        return EPIBundle(
            timestamp=record.timestamp,
            structural_timestamp=(
                float(structural_timestamp)
                if structural_timestamp is not None
                else float(record.timestamp)
            ),
            epi=epi_value,
            delta_nfr=delta_nfr,
            delta_nfr_longitudinal=longitudinal_delta,
            delta_nfr_lateral=lateral_delta,
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
