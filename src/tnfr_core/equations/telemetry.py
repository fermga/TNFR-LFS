"""Telemetry record definitions shared across EPI modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["_MISSING_FLOAT", "TelemetryRecord"]


_MISSING_FLOAT = float("nan")


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

