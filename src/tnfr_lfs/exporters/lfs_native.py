"""Future Live for Speed native exporter utilities."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Callable, Mapping

from tnfr_lfs.exporters.setup_plan import SetupPlan, serialise_setup_plan


FEATURE_FLAG_LFS_NATIVE_EXPORT = bool(os.getenv("TNFR_LFS_NATIVE_EXPORT"))


@dataclass(frozen=True)
class NativeSetupVector:
    """Lightweight container for serialised setup changes.

    This mirrors the information required to convert a :class:`SetupPlan`
    into the binary/text representation consumed directly by Live for Speed.
    """

    car_model: str
    session: str | None
    decision_vector: Mapping[str, float]


def build_native_vector(plan: SetupPlan) -> NativeSetupVector:
    """Extract the decision vector from a setup plan."""

    payload = serialise_setup_plan(plan)
    return NativeSetupVector(
        car_model=str(payload.get("car_model", "")),
        session=payload.get("session"),
        decision_vector={
            change["parameter"]: float(change.get("delta", 0.0))
            for change in payload.get("changes", [])
            if change.get("parameter")
        },
    )


def encode_native_setup(plan: SetupPlan) -> bytes:
    """Encode a setup plan into the native LFS payload.

    The implementation is guarded behind ``TNFR_LFS_NATIVE_EXPORT`` to avoid
    surprising side effects while the mapping logic is still a work in
    progress.
    """

    if not FEATURE_FLAG_LFS_NATIVE_EXPORT:  # pragma: no cover - feature gated
        raise RuntimeError(
            "Native LFS exporter is disabled. Set TNFR_LFS_NATIVE_EXPORT=1 to enable."
        )

    vector = build_native_vector(plan)

    buffer = bytearray(132)
    _write_bytes(buffer, 0, b"SRSETT")
    _write_byte(buffer, 6, 0)
    _write_byte(buffer, 7, 252)
    _write_byte(buffer, 8, 2)
    _write_bytes(buffer, 9, b"\x00\x00\x00")
    # Bit 7 marks modern setups; keep the rest cleared by default unless
    # explicitly overridden by the decision vector.
    flags = 0x80
    if _as_bool(vector.decision_vector.get("abs_enabled")):
        flags |= 0x04
    if _as_bool(vector.decision_vector.get("traction_control")):
        flags |= 0x02
    if _as_bool(vector.decision_vector.get("asymmetric_setup")):
        flags |= 0x01
    _write_byte(buffer, 12, flags)

    encoders: Mapping[str, Callable[[bytearray, float], None]] = {
        "front_camber_deg": lambda buf, val: _encode_camber(buf, (120, 121), val),
        "rear_camber_deg": lambda buf, val: _encode_camber(buf, (80, 81), val),
        "front_toe_deg": lambda buf, val: _encode_toe(buf, 116, val),
        "rear_toe_deg": lambda buf, val: _encode_toe(buf, 76, val),
        "caster_deg": lambda buf, val: _encode_caster(buf, 117, val),
        "parallel_steer": lambda buf, val: _encode_parallel_steer(buf, 118, val),
        "steering_lock_deg": lambda buf, val: _encode_steering_lock(buf, 122, val),
        "front_ride_height": lambda buf, val: _write_float(buf, 92, val),
        "rear_ride_height": lambda buf, val: _write_float(buf, 52, val),
        "front_spring_stiffness": lambda buf, val: _write_float(buf, 96, val),
        "rear_spring_stiffness": lambda buf, val: _write_float(buf, 56, val),
        "front_rebound_clicks": lambda buf, val: _write_float(buf, 104, val),
        "rear_rebound_clicks": lambda buf, val: _write_float(buf, 64, val),
        "front_compression_clicks": lambda buf, val: _write_float(buf, 100, val),
        "rear_compression_clicks": lambda buf, val: _write_float(buf, 60, val),
        "front_arb_steps": lambda buf, val: _write_float(buf, 108, val),
        "rear_arb_steps": lambda buf, val: _write_float(buf, 68, val),
        "front_tyre_pressure": lambda buf, val: _encode_pressure(buf, 128, val),
        "rear_tyre_pressure": lambda buf, val: _encode_pressure(buf, 88, val),
        "brake_max_per_wheel": lambda buf, val: _encode_brake_force(buf, 16, val),
        "brake_bias_pct": lambda buf, val: _encode_brake_bias(buf, 26, val),
        "diff_power_lock": lambda buf, val: _encode_percentage(buf, 86, val),
        "diff_coast_lock": lambda buf, val: _encode_percentage(buf, 87, val),
        "diff_preload_nm": lambda buf, val: _encode_preload(buf, 83, val),
        "rear_wing_angle": lambda buf, val: _encode_angle(buf, 20, val),
        "front_wing_angle": lambda buf, val: _encode_angle(buf, 21, val),
        "final_drive_ratio": lambda buf, val: _write_float(buf, 28, val),
        "gear_1_ratio": lambda buf, val: _write_float(buf, 32, val),
        "gear_2_ratio": lambda buf, val: _write_float(buf, 36, val),
        "gear_3_ratio": lambda buf, val: _write_float(buf, 40, val),
        "gear_4_ratio": lambda buf, val: _write_float(buf, 44, val),
        "gear_5_ratio": lambda buf, val: _write_float(buf, 48, val),
        "gear_6_ratio": lambda buf, val: _write_float(buf, 72, val),
        "gear_7_ratio": lambda buf, val: _write_float(buf, 112, val),
    }

    for parameter, encoder in encoders.items():
        if parameter not in vector.decision_vector:
            continue
        encoder(buffer, float(vector.decision_vector[parameter]))

    return bytes(buffer)


def _as_bool(value: float | int | bool | None) -> bool:
    return bool(value)


def _write_bytes(buffer: bytearray, offset: int, value: bytes) -> None:
    buffer[offset : offset + len(value)] = value


def _write_byte(buffer: bytearray, offset: int, value: int) -> None:
    buffer[offset] = max(0, min(255, int(value)))


def _write_word(buffer: bytearray, offset: int, value: int) -> None:
    buffer[offset : offset + 2] = struct.pack("<H", max(0, min(0xFFFF, int(value))))


def _write_float(buffer: bytearray, offset: int, value: float) -> None:
    buffer[offset : offset + 4] = struct.pack("<f", float(value))


def _encode_camber(buffer: bytearray, offsets: tuple[int, int], value: float) -> None:
    clamped = max(-4.5, min(4.5, value))
    encoded = int(round(clamped * 10.0 + 45.0))
    for offset in offsets:
        _write_byte(buffer, offset, encoded)


def _encode_toe(buffer: bytearray, offset: int, value: float) -> None:
    clamped = max(-0.9, min(0.9, value))
    encoded = int(round((clamped + 0.9) / 0.1))
    _write_byte(buffer, offset, encoded)


def _encode_caster(buffer: bytearray, offset: int, value: float) -> None:
    encoded = int(round(max(0.0, min(25.5, value)) * 10.0))
    _write_byte(buffer, offset, encoded)


def _encode_pressure(buffer: bytearray, offset: int, value: float) -> None:
    _write_word(buffer, offset, round(max(0.0, min(6553.5, value)) * 1.0))


def _encode_parallel_steer(buffer: bytearray, offset: int, value: float) -> None:
    clamped = max(-1.0, min(1.0, float(value)))
    encoded = int(round((clamped + 1.0) * 100.0))
    _write_byte(buffer, offset, encoded)


def _encode_steering_lock(buffer: bytearray, offset: int, value: float) -> None:
    clamped = max(5.0, min(65.0, float(value)))
    encoded = int(round(clamped * 10.0))
    _write_word(buffer, offset, encoded)


def _encode_brake_bias(buffer: bytearray, offset: int, value: float) -> None:
    encoded = int(round(max(0.0, min(100.0, value)) * 2.0))
    _write_byte(buffer, offset, encoded)


def _encode_brake_force(buffer: bytearray, offset: int, value: float) -> None:
    clamped = max(-1.0, min(1.0, float(value)))
    _write_float(buffer, offset, clamped)


def _encode_percentage(buffer: bytearray, offset: int, value: float) -> None:
    encoded = int(round(max(0.0, min(100.0, value))))
    _write_byte(buffer, offset, encoded)


def _encode_preload(buffer: bytearray, offset: int, value: float) -> None:
    encoded = int(round(max(0.0, min(2550.0, value)) / 10.0))
    _write_byte(buffer, offset, encoded)


def _encode_angle(buffer: bytearray, offset: int, value: float) -> None:
    encoded = int(round(max(0.0, min(255.0, value))))
    _write_byte(buffer, offset, encoded)


__all__ = [
    "FEATURE_FLAG_LFS_NATIVE_EXPORT",
    "NativeSetupVector",
    "build_native_vector",
    "encode_native_setup",
]
