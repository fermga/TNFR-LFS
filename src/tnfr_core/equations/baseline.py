"""Shared baseline and ΔNFR helper utilities."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from statistics import mean
from typing import Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from tnfr_core.runtime.shared import (
    CacheOptions,
    SupportsTelemetrySample,
    cached_delta_nfr_map,
    should_use_delta_cache,
)
from tnfr_core.equations.coherence import compute_node_delta_nfr, sense_index
from tnfr_core.equations.delta_utils import distribute_weighted_delta
from tnfr_core.equations.epi_evolution import evolve_epi
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
from tnfr_core.equations.telemetry import TelemetryRecord

SampleFactory = Callable[..., SupportsTelemetrySample]

DEFAULT_PHASE_WEIGHTS: Mapping[str, float] = {"__default__": 1.0}

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


class BaselineResolver(Protocol):
    """Protocol describing baseline calibration stores."""

    def baseline_for(
        self,
        player_name: str,
        car_model: str,
        fallback: TelemetryRecord,
    ) -> TelemetryRecord:
        ...

    def observe_baseline(
        self,
        player_name: str,
        car_model: str,
        baseline: TelemetryRecord,
    ) -> None:
        ...


class SupportsNaturalFrequencySnapshot(Protocol):
    """Subset of the natural frequency snapshot API used by ΔNFR utilities."""

    by_node: Mapping[str, float]
    classification: str
    category: str
    dominant_frequency: float
    coherence_index: float

    @property
    def frequency_label(self) -> str:  # pragma: no cover - trivial property wrapper
        ...


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
    record: SupportsTelemetrySample, baseline: SupportsTelemetrySample
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
    record: SupportsTelemetrySample,
    baseline: SupportsTelemetrySample,
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


def _distribute_node_delta(delta_nfr: float, node_signals: Mapping[str, float]) -> Dict[str, float]:
    return distribute_weighted_delta(delta_nfr, node_signals)


def compute_delta_nfr(
    record: SupportsTelemetrySample,
    *,
    reference: SupportsTelemetrySample | None = None,
) -> tuple[float, Dict[str, float], Dict[str, Dict[str, float]]]:
    """Return the ΔNFR value and its distribution across telemetry nodes."""

    baseline = reference or getattr(record, "reference", None) or record
    feature_contributions = _node_feature_contributions(record, baseline)
    node_signals = {
        node: sum(values.values()) for node, values in feature_contributions.items()
    }
    delta_nfr = record.nfr - baseline.nfr
    node_deltas = _distribute_node_delta(delta_nfr, node_signals)
    return delta_nfr, node_deltas, feature_contributions


def _delta_nfr_by_node_uncached(record: SupportsTelemetrySample) -> Dict[str, float]:
    _, node_deltas, _ = compute_delta_nfr(record)
    return node_deltas


def _resolve_uncached_callable() -> Callable[[SupportsTelemetrySample], Dict[str, float]]:
    epi_module = sys.modules.get("tnfr_core.equations.epi")
    if epi_module is not None:
        override = getattr(epi_module, "_delta_nfr_by_node_uncached", None)
        if override is not None:
            return override
    return _delta_nfr_by_node_uncached


def delta_nfr_by_node(
    record: SupportsTelemetrySample,
    *,
    cache_options: CacheOptions | None = None,
) -> Mapping[str, float]:
    """Compute ΔNFR contributions for each subsystem."""

    uncached = _resolve_uncached_callable()
    if not should_use_delta_cache(cache_options):
        return uncached(record)
    return cached_delta_nfr_map(record, lambda: uncached(record))


class DeltaCalculator:
    """Compute delta metrics relative to a baseline."""

    @staticmethod
    def _resolve_sample_factory(
        prototype: SupportsTelemetrySample, sample_factory: SampleFactory | None
    ) -> SampleFactory:
        if sample_factory is not None:
            return sample_factory

        sample_type = type(prototype)

        def _default_factory(**payload: object) -> SupportsTelemetrySample:
            try:
                return sample_type(**payload)  # type: ignore[call-arg]
            except TypeError:
                return TelemetryRecord(**payload)

        return _default_factory

    @staticmethod
    def derive_baseline(
        records: Sequence[SupportsTelemetrySample],
        *,
        sample_factory: SampleFactory | None = None,
    ) -> SupportsTelemetrySample:
        """Return a synthetic baseline record representing the average state."""

        if not records:
            raise ValueError("Cannot derive baseline without telemetry samples")

        prototype = records[0]
        factory = DeltaCalculator._resolve_sample_factory(prototype, sample_factory)

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

        prototype = records[0]
        structural = getattr(prototype, "structural_timestamp", None)
        if structural is None or not math.isfinite(structural):
            structural = getattr(prototype, "timestamp", 0.0)

        payload: Dict[str, object] = {
            "timestamp": float(getattr(prototype, "timestamp", 0.0)),
            "structural_timestamp": float(structural),
            **baseline_kwargs,
        }

        for optional_field in ("car_model", "track_name", "tyre_compound"):
            if hasattr(prototype, optional_field):
                payload[optional_field] = getattr(prototype, optional_field)

        try:
            baseline = factory(**payload)
        except TypeError:
            baseline = TelemetryRecord(**payload)
        return baseline

    @staticmethod
    def resolve_baseline(
        records: Sequence[SupportsTelemetrySample],
        *,
        calibration: BaselineResolver | None = None,
        player_name: str | None = None,
        car_model: str | None = None,
        sample_factory: SampleFactory | None = None,
    ) -> SupportsTelemetrySample:
        baseline = DeltaCalculator.derive_baseline(
            records, sample_factory=sample_factory
        )
        if calibration is None or not player_name or not car_model:
            return baseline
        resolved = calibration.baseline_for(player_name, car_model, baseline)
        calibration.observe_baseline(player_name, car_model, baseline)
        return resolved

    @staticmethod
    def compute_bundle(
        record: SupportsTelemetrySample,
        baseline: SupportsTelemetrySample,
        epi_value: float,
        *,
        prev_integrated_epi: Optional[float] = None,
        dt: float = 0.0,
        nu_f_by_node: Optional[Mapping[str, float]] = None,
        nu_f_snapshot: SupportsNaturalFrequencySnapshot | None = None,
        phase: str = "entry",
        phase_weights: Optional[Mapping[str, Mapping[str, float] | float]] = None,
        phase_target_nu_f: Mapping[str, Mapping[str, float] | float]
        | Mapping[str, float]
        | float
        | None = None,
    ) -> EPIBundle:
        structural_timestamp = getattr(record, "structural_timestamp", None)
        delta_nfr, node_deltas, feature_contributions = compute_delta_nfr(
            record, reference=baseline
        )
        axis_signals = _axis_signal_components(record, baseline, feature_contributions)
        if nu_f_snapshot is None:
            from tnfr_core.equations.epi import resolve_nu_f_by_node  # local import to avoid cycle

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
        integrated_epi, derivative, nodal_evolution = evolve_epi(
            previous_state, node_deltas, dt, nu_f_map
        )
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
    def reproject_bundle_phase(
        bundle: EPIBundle,
        *,
        nu_f_snapshot: SupportsNaturalFrequencySnapshot,
        phase: str,
        phase_weights: Mapping[str, Mapping[str, float] | float] | Mapping[str, float] | float | None,
        baseline_nfr: float,
        dt: float,
        prev_integrated_epi: float | None = None,
        phase_target_nu_f: Mapping[str, Mapping[str, float] | float]
        | Mapping[str, float]
        | float
        | None = None,
    ) -> EPIBundle:
        """Reproject a cached bundle onto a new phase configuration."""

        phase_weight_map: Mapping[str, Mapping[str, float] | float] | Mapping[str, float] | float
        if phase_weights is None:
            phase_weight_map = DEFAULT_PHASE_WEIGHTS
        else:
            phase_weight_map = phase_weights

        nu_f_map = dict(nu_f_snapshot.by_node)
        node_deltas = {
            "tyres": bundle.tyres.delta_nfr,
            "suspension": bundle.suspension.delta_nfr,
            "chassis": bundle.chassis.delta_nfr,
            "brakes": bundle.brakes.delta_nfr,
            "transmission": bundle.transmission.delta_nfr,
            "track": bundle.track.delta_nfr,
            "driver": bundle.driver.delta_nfr,
        }

        global_si = sense_index(
            bundle.delta_nfr,
            node_deltas,
            baseline_nfr,
            nu_f_by_node=nu_f_map,
            active_phase=phase,
            w_phase=phase_weight_map,
            nu_f_targets=phase_target_nu_f,
        )

        previous_state = bundle.epi if prev_integrated_epi is None else prev_integrated_epi
        integrated_epi, derivative, nodal_evolution = evolve_epi(
            previous_state, node_deltas, dt, nu_f_map
        )

        def _update_node(node_name: str, node_model):
            integral, node_derivative = nodal_evolution.get(node_name, (0.0, 0.0))
            return replace(
                node_model,
                nu_f=nu_f_map.get(node_name, 0.0),
                integrated_epi=integral,
                dEPI_dt=node_derivative,
            )

        updated_bundle = replace(
            bundle,
            sense_index=global_si,
            dEPI_dt=derivative,
            integrated_epi=integrated_epi,
            node_evolution=dict(nodal_evolution),
            tyres=_update_node("tyres", bundle.tyres),
            suspension=_update_node("suspension", bundle.suspension),
            chassis=_update_node("chassis", bundle.chassis),
            brakes=_update_node("brakes", bundle.brakes),
            transmission=_update_node("transmission", bundle.transmission),
            track=_update_node("track", bundle.track),
            driver=_update_node("driver", bundle.driver),
            nu_f_classification=nu_f_snapshot.classification,
            nu_f_category=nu_f_snapshot.category,
            nu_f_label=nu_f_snapshot.frequency_label,
            nu_f_dominant=nu_f_snapshot.dominant_frequency,
            coherence_index=nu_f_snapshot.coherence_index,
        )

        return updated_bundle

    @staticmethod
    def _build_nodes(
        record: SupportsTelemetrySample,
        node_deltas: Mapping[str, float],
        delta_nfr: float,
        nu_f_by_node: Mapping[str, float],
        nodal_evolution: Mapping[str, Tuple[float, float]] | None,
    ) -> Dict[str, object]:
        node_evolution = nodal_evolution or {}

        def node_si(node_delta: float) -> float:
            if abs(delta_nfr) < 1e-9:
                return 1.0
            ratio = min(1.0, abs(node_delta) / (abs(delta_nfr) + 1e-9))
            return max(0.0, min(1.0, 1.0 - ratio))

        def node_state(name: str) -> Tuple[float, float]:
            return node_evolution.get(name, (0.0, 0.0))

        brake_values = [
            float(record.brake_temp_fl),
            float(record.brake_temp_fr),
            float(record.brake_temp_rl),
            float(record.brake_temp_rr),
        ]
        brake_finite = [value for value in brake_values if math.isfinite(value) and value > 0.0]
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


__all__ = [
    "AXIS_FEATURE_MAP",
    "BaselineResolver",
    "DEFAULT_PHASE_WEIGHTS",
    "DeltaCalculator",
    "_ackermann_parallel_delta",
    "_delta_nfr_by_node_uncached",
    "SupportsNaturalFrequencySnapshot",
    "compute_delta_nfr",
    "delta_nfr_by_node",
]
