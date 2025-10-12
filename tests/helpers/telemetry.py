"""Shared factory helpers for telemetry-centric tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

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


class ProtocolTelemetrySample:
    """Concrete :class:`SupportsTelemetrySample` implementation for tests."""

    def __init__(self, source: TelemetryRecord) -> None:
        for field in TelemetryRecord.__dataclass_fields__:
            setattr(self, field, getattr(source, field))
        self.reference: "ProtocolTelemetrySample" | None = None


def clone_protocol_sample(
    record: TelemetryRecord,
    *,
    reference: "ProtocolTelemetrySample" | None = None,
) -> ProtocolTelemetrySample:
    """Clone ``record`` into a :class:`ProtocolTelemetrySample`."""

    sample = ProtocolTelemetrySample(record)
    sample.reference = reference
    return sample


def clone_protocol_series(
    records: Sequence[TelemetryRecord],
) -> list[ProtocolTelemetrySample]:
    """Return protocol-compatible clones for ``records`` preserving references."""

    mapping: dict[int, ProtocolTelemetrySample] = {}
    clones: list[ProtocolTelemetrySample] = []
    for record in records:
        sample = ProtocolTelemetrySample(record)
        clones.append(sample)
        mapping[id(record)] = sample

    for record, sample in zip(records, clones):
        reference = getattr(record, "reference", None)
        if reference is None:
            sample.reference = None
            continue
        cloned_reference = mapping.get(id(reference))
        sample.reference = cloned_reference

    return clones


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


def build_contextual_delta_record(**overrides: Any) -> TelemetryRecord:
    """Return a representative record for contextual delta tests."""

    payload = {
        "timestamp": 0.0,
        "vertical_load": 5200.0,
        "slip_ratio": 0.0,
        "lateral_accel": 1.2,
        "longitudinal_accel": 0.2,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "brake_pressure": 0.0,
        "locking": 0.0,
        "nfr": 500.0,
        "si": 0.8,
        "speed": 40.0,
        "yaw_rate": 0.02,
        "slip_angle": 0.01,
        "steer": 0.1,
        "throttle": 0.5,
        "gear": 3,
        "vertical_load_front": 2600.0,
        "vertical_load_rear": 2600.0,
        "mu_eff_front": 1.1,
        "mu_eff_rear": 1.1,
        "mu_eff_front_lateral": 1.1,
        "mu_eff_front_longitudinal": 1.1,
        "mu_eff_rear_lateral": 1.1,
        "mu_eff_rear_longitudinal": 1.1,
        "suspension_travel_front": 0.02,
        "suspension_travel_rear": 0.02,
        "suspension_velocity_front": 0.1,
        "suspension_velocity_rear": 0.1,
    }
    payload.update(overrides)
    return build_telemetry_record(**payload)


def build_resonance_record(
    timestamp: float,
    *,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    **overrides: Any,
) -> TelemetryRecord:
    """Return a simplified record focused on chassis resonance axes."""

    payload = {
        "timestamp": timestamp,
        "vertical_load": 0.0,
        "slip_ratio": 0.0,
        "lateral_accel": 0.0,
        "longitudinal_accel": 0.0,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "brake_pressure": 0.0,
        "locking": 0.0,
        "nfr": 0.0,
        "si": 0.0,
        "speed": 0.0,
        "yaw_rate": 0.0,
        "slip_angle": 0.0,
        "steer": 0.0,
        "throttle": 0.0,
        "gear": 0,
        "vertical_load_front": 0.0,
        "vertical_load_rear": 0.0,
        "mu_eff_front": 0.0,
        "mu_eff_rear": 0.0,
        "mu_eff_front_lateral": 0.0,
        "mu_eff_front_longitudinal": 0.0,
        "mu_eff_rear_lateral": 0.0,
        "mu_eff_rear_longitudinal": 0.0,
        "suspension_travel_front": 0.0,
        "suspension_travel_rear": 0.0,
        "suspension_velocity_front": 0.0,
        "suspension_velocity_rear": 0.0,
    }
    payload.update(overrides)
    return build_telemetry_record(**payload)


def build_calibration_record(
    vertical_load: float,
    *,
    nfr: float = 0.7,
    si: float = 0.8,
    **overrides: Any,
) -> TelemetryRecord:
    """Return a record tailored for coherence calibration tests."""

    payload = {
        "timestamp": 0.0,
        "vertical_load": vertical_load,
        "slip_ratio": 0.02,
        "lateral_accel": 0.0,
        "longitudinal_accel": 0.0,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "brake_pressure": 0.0,
        "locking": 0.0,
        "nfr": nfr,
        "si": si,
        "speed": 50.0,
        "yaw_rate": 0.0,
        "slip_angle": 0.0,
        "steer": 0.0,
        "throttle": 0.5,
        "gear": 3,
        "vertical_load_front": vertical_load * 0.6,
        "vertical_load_rear": vertical_load * 0.4,
        "mu_eff_front": 0.8,
        "mu_eff_rear": 0.8,
        "mu_eff_front_lateral": 0.8,
        "mu_eff_front_longitudinal": 0.8,
        "mu_eff_rear_lateral": 0.8,
        "mu_eff_rear_longitudinal": 0.8,
        "suspension_travel_front": 0.02,
        "suspension_travel_rear": 0.02,
        "suspension_velocity_front": 0.0,
        "suspension_velocity_rear": 0.0,
    }
    payload.update(overrides)
    return build_telemetry_record(**payload)


def build_dynamic_record(
    timestamp: float,
    vertical_load: float,
    slip_ratio: float,
    lateral_accel: float,
    longitudinal_accel: float,
    nfr: float,
    si: float,
    *,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    brake_pressure: float = 0.0,
    locking: float = 0.0,
    speed: float = 0.0,
    yaw_rate: float = 0.0,
    slip_angle: float = 0.0,
    steer: float = 0.0,
    throttle: float = 0.0,
    gear: int = 0,
    vertical_load_front: float = 0.0,
    vertical_load_rear: float = 0.0,
    mu_eff_front: float = 0.0,
    mu_eff_rear: float = 0.0,
    mu_eff_front_lateral: float = 0.0,
    mu_eff_front_longitudinal: float = 0.0,
    mu_eff_rear_lateral: float = 0.0,
    mu_eff_rear_longitudinal: float = 0.0,
    suspension_travel_front: float = 0.0,
    suspension_travel_rear: float = 0.0,
    suspension_velocity_front: float = 0.0,
    suspension_velocity_rear: float = 0.0,
    **overrides: Any,
) -> TelemetryRecord:
    """Return a record suited to dynamic/operator and segmentation tests."""

    payload = {
        "timestamp": timestamp,
        "vertical_load": vertical_load,
        "slip_ratio": slip_ratio,
        "lateral_accel": lateral_accel,
        "longitudinal_accel": longitudinal_accel,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "brake_pressure": brake_pressure,
        "locking": locking,
        "nfr": nfr,
        "si": si,
        "speed": speed,
        "yaw_rate": yaw_rate,
        "slip_angle": slip_angle,
        "steer": steer,
        "throttle": throttle,
        "gear": gear,
        "vertical_load_front": vertical_load_front,
        "vertical_load_rear": vertical_load_rear,
        "mu_eff_front": mu_eff_front,
        "mu_eff_rear": mu_eff_rear,
        "mu_eff_front_lateral": mu_eff_front_lateral,
        "mu_eff_front_longitudinal": mu_eff_front_longitudinal,
        "mu_eff_rear_lateral": mu_eff_rear_lateral,
        "mu_eff_rear_longitudinal": mu_eff_rear_longitudinal,
        "suspension_travel_front": suspension_travel_front,
        "suspension_travel_rear": suspension_travel_rear,
        "suspension_velocity_front": suspension_velocity_front,
        "suspension_velocity_rear": suspension_velocity_rear,
    }
    payload.update(overrides)
    return build_telemetry_record(**payload)


def build_frequency_record(
    timestamp: float,
    *,
    steer: float,
    throttle: float,
    brake: float,
    suspension: float,
    **overrides: Any,
) -> TelemetryRecord:
    """Return a record for frequency series tests with derived loads."""

    payload = {
        "timestamp": timestamp,
        "vertical_load": 5200.0 + suspension * 150.0,
        "slip_ratio": 0.02,
        "lateral_accel": 0.6 + steer * 0.1,
        "longitudinal_accel": 0.2 + (throttle - brake) * 0.05,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "brake_pressure": max(0.0, brake),
        "locking": 0.0,
        "nfr": 500.0,
        "si": 0.82,
        "speed": 45.0,
        "yaw_rate": steer * 0.15,
        "slip_angle": 0.01,
        "steer": steer,
        "throttle": max(0.0, min(1.0, throttle)),
        "gear": 3,
        "vertical_load_front": 2600.0 + suspension * 40.0,
        "vertical_load_rear": 2600.0 - suspension * 40.0,
        "mu_eff_front": 1.05,
        "mu_eff_rear": 1.05,
        "mu_eff_front_lateral": 1.05,
        "mu_eff_front_longitudinal": 1.0,
        "mu_eff_rear_lateral": 1.05,
        "mu_eff_rear_longitudinal": 1.0,
        "suspension_travel_front": 0.02 + suspension * 0.01,
        "suspension_travel_rear": 0.02 + suspension * 0.01,
        "suspension_velocity_front": suspension,
        "suspension_velocity_rear": suspension * 0.92,
    }
    payload.update(overrides)
    return build_telemetry_record(**payload)
