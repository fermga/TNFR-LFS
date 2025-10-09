"""Utilities for constructing steering-specific telemetry fixtures."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from tnfr_lfs.core.epi import TelemetryRecord
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
