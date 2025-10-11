"""Utilities for constructing steering-specific telemetry fixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from tnfr_lfs.core.epi import DeltaCalculator, TelemetryRecord, _ackermann_parallel_delta
from tnfr_lfs.core.epi_models import (
    BrakesNode,
    ChassisNode,
    DriverNode,
    EPIBundle,
    SuspensionNode,
    TrackNode,
    TransmissionNode,
    TyresNode,
)
from tnfr_lfs.core.metrics import WindowMetrics, compute_window_metrics


def build_steering_record(
    timestamp: float,
    *,
    yaw_rate: float,
    steer: float,
    slip_angle_fl: float,
    slip_angle_fr: float,
    nfr: float,
    si: float = 0.8,
    throttle: float = 0.4,
    speed: float = 45.0,
    gear: int = 3,
    **overrides: Any,
) -> TelemetryRecord:
    """Create a :class:`TelemetryRecord` tailored for steering tests.

    The helper mirrors the common defaults used across steering-related
    test modules while allowing callers to override individual telemetry
    attributes when they need to adjust the scenario.
    """

    base = TelemetryRecord(
        timestamp=timestamp,
        vertical_load=5000.0,
        slip_ratio=0.0,
        lateral_accel=0.0,
        longitudinal_accel=0.0,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        brake_pressure=0.0,
        locking=0.0,
        nfr=nfr,
        si=si,
        speed=speed,
        yaw_rate=yaw_rate,
        slip_angle=0.0,
        steer=steer,
        throttle=throttle,
        gear=gear,
        vertical_load_front=2500.0,
        vertical_load_rear=2500.0,
        mu_eff_front=1.0,
        mu_eff_rear=1.0,
        mu_eff_front_lateral=1.0,
        mu_eff_front_longitudinal=0.95,
        mu_eff_rear_lateral=1.0,
        mu_eff_rear_longitudinal=0.95,
        suspension_travel_front=0.0,
        suspension_travel_rear=0.0,
        suspension_velocity_front=0.0,
        suspension_velocity_rear=0.0,
        slip_angle_fl=slip_angle_fl,
        slip_angle_fr=slip_angle_fr,
    )

    if overrides:
        base = replace(base, **overrides)
    return base


def build_steering_bundle(
    record: TelemetryRecord,
    ackermann_delta: float,
    *,
    delta_nfr_proj_longitudinal: float = 0.0,
    delta_nfr_proj_lateral: float = 0.0,
    **overrides: Any,
) -> EPIBundle:
    """Build an :class:`EPIBundle` consistent with steering records."""

    share = record.nfr / 7.0
    bundle = EPIBundle(
        timestamp=record.timestamp,
        epi=0.0,
        delta_nfr=record.nfr,
        sense_index=record.si,
        tyres=TyresNode(delta_nfr=share, sense_index=record.si),
        suspension=SuspensionNode(delta_nfr=share, sense_index=record.si),
        chassis=ChassisNode(
            delta_nfr=share,
            sense_index=record.si,
            yaw=record.yaw,
            pitch=record.pitch,
            roll=record.roll,
            yaw_rate=record.yaw_rate,
            lateral_accel=record.lateral_accel,
            longitudinal_accel=record.longitudinal_accel,
        ),
        brakes=BrakesNode(delta_nfr=share, sense_index=record.si),
        transmission=TransmissionNode(
            delta_nfr=share,
            sense_index=record.si,
            throttle=record.throttle,
            gear=record.gear,
            speed=record.speed,
            longitudinal_accel=record.longitudinal_accel,
            rpm=record.rpm,
            line_deviation=record.line_deviation,
        ),
        track=TrackNode(
            delta_nfr=share,
            sense_index=record.si,
            axle_load_balance=0.0,
            axle_velocity_balance=0.0,
            yaw=record.yaw,
            lateral_accel=record.lateral_accel,
        ),
        driver=DriverNode(
            delta_nfr=share,
            sense_index=record.si,
            steer=record.steer,
            throttle=record.throttle,
            style_index=record.si,
        ),
        delta_nfr_proj_longitudinal=delta_nfr_proj_longitudinal,
        delta_nfr_proj_lateral=delta_nfr_proj_lateral,
        ackermann_parallel_index=ackermann_delta,
    )

    if overrides:
        bundle = replace(bundle, **overrides)
    return bundle


def _resolve_overrides(
    overrides: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    index: int,
) -> Mapping[str, Any]:
    if overrides is None:
        return {}
    if isinstance(overrides, Mapping):
        return overrides
    if index < len(overrides):
        return overrides[index]
    return {}


def build_parallel_window_metrics(
    slip_angles: Sequence[tuple[float, float]],
    *,
    yaw_sign: float = 1.0,
    yaw_rates: Sequence[float] | None = None,
    steer_series: Sequence[float] | None = None,
    nfr_series: Sequence[float] | None = None,
    timestamp_start: float = 0.0,
    timestamp_step: float = 0.4,
    record_overrides: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    bundle_overrides: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    return_components: bool = False,
) -> WindowMetrics | tuple[
    WindowMetrics,
    list[TelemetryRecord],
    list[EPIBundle],
    list[float],
    TelemetryRecord,
]:
    """Build :class:`WindowMetrics` for a parallel-turn slip-angle series."""

    records: list[TelemetryRecord] = []
    for index, (inner_slip, outer_slip) in enumerate(slip_angles):
        timestamp = timestamp_start + float(index) * timestamp_step
        yaw_rate = (
            float(yaw_rates[index])
            if yaw_rates is not None and index < len(yaw_rates)
            else yaw_sign * (0.5 + 0.05 * index)
        )
        steer = (
            float(steer_series[index])
            if steer_series is not None and index < len(steer_series)
            else yaw_sign * (0.2 + 0.04 * index)
        )
        if yaw_sign >= 0.0:
            slip_fl, slip_fr = inner_slip, outer_slip
        else:
            slip_fl, slip_fr = outer_slip, inner_slip
        nfr = (
            float(nfr_series[index])
            if nfr_series is not None and index < len(nfr_series)
            else 100.0 + index
        )
        overrides = _resolve_overrides(record_overrides, index)
        record = build_steering_record(
            timestamp,
            yaw_rate=yaw_rate,
            steer=steer,
            slip_angle_fl=slip_fl,
            slip_angle_fr=slip_fr,
            nfr=nfr,
            **overrides,
        )
        records.append(record)

    baseline = DeltaCalculator.derive_baseline(records)
    ackermann_values = [
        _ackermann_parallel_delta(record, baseline) for record in records
    ]
    bundles: list[EPIBundle] = []
    for index, (record, ackermann) in enumerate(zip(records, ackermann_values)):
        overrides = _resolve_overrides(bundle_overrides, index)
        bundles.append(build_steering_bundle(record, ackermann, **overrides))

    metrics = compute_window_metrics(records, bundles=bundles)

    if return_components:
        return metrics, records, bundles, ackermann_values, baseline
    return metrics
