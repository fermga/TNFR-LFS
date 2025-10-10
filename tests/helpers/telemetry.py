"""Shared factory helpers for telemetry-centric tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from tnfr_lfs.core.epi import TelemetryRecord


_DEFAULT_TELEMETRY_RECORD = TelemetryRecord(
    timestamp=0.0,
    vertical_load=5000.0,
    slip_ratio=0.0,
    lateral_accel=0.0,
    longitudinal_accel=0.0,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    brake_pressure=0.0,
    locking=0.0,
    nfr=100.0,
    si=0.8,
    speed=0.0,
    yaw_rate=0.0,
    slip_angle=0.0,
    steer=0.0,
    throttle=0.0,
    gear=3,
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
)


def build_telemetry_record(*args: Any, **overrides: Any) -> TelemetryRecord:
    """Return a :class:`TelemetryRecord` with convenient defaults.

    The factory mirrors the historical helper signature used across the test
    suite, supporting positional arguments for ``timestamp`` and ``nfr`` while
    still allowing explicit keyword overrides for any dataclass field.
    """

    if args:
        timestamp = args[0]
        if "timestamp" in overrides and overrides["timestamp"] != timestamp:
            raise ValueError("timestamp provided both positionally and by keyword")
        overrides["timestamp"] = timestamp
        remaining = args[1:]
    else:
        try:
            timestamp = overrides["timestamp"]
        except KeyError as error:  # pragma: no cover - defensive guard
            raise TypeError("build_telemetry_record() missing required argument: 'timestamp'") from error
        remaining = ()

    if remaining:
        overrides.setdefault("nfr", remaining[0])
        if len(remaining) > 1:
            raise TypeError("build_telemetry_record() accepts at most two positional arguments")

    return replace(_DEFAULT_TELEMETRY_RECORD, **overrides)
