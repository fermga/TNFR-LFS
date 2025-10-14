"""Utilities for reading Live for Speed RAF telemetry captures.

The RAF (Replay Analyser Format) files produced by Live for Speed contain a
compact binary snapshot of a car's static configuration followed by a stream of
per-frame telemetry samples.  The helpers in this module expose the RAF
structure through small ``dataclass`` based containers and provide conversions
to :class:`~tnfr_core.equations.epi.TelemetryRecord` instances so that callers can use
RAF recordings with the rest of the telemetry tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import struct
from typing import BinaryIO, Mapping, Sequence

from tnfr_core.equations.epi import TelemetryRecord
from .._tyre_compound import normalise_compound_label, resolve_compound_metadata

_MAGIC = b"LFSRAF"
_U16 = struct.Struct("<H")
_I32 = struct.Struct("<i")
_F32 = struct.Struct("<f")


@dataclass(frozen=True)
class RafHeader:
    """Top level metadata extracted from the RAF file header."""

    magic: str
    version: int
    raw_version: int
    interval_ms: float
    header_size: int
    frame_size: int
    car_static_words: int
    wheel_static_words: int
    frame_count: int
    track_code: str
    player_name: str
    car_model: str
    track_name: str
    track_layout: str
    session_name: str

    @property
    def interval_seconds(self) -> float:
        """Sampling interval for the telemetry stream in seconds."""

        return self.interval_ms / 1000.0


@dataclass(frozen=True)
class RafCarStatic:
    """Static car information stored in the RAF header."""

    idle_rpm: float
    mass_kg: float
    peak_torque_rpm: float
    rev_limiter_rpm: float
    final_drive_ratio: float
    gear_ratios: tuple[float, ...]
    front_track_width: float
    wheelbase: float


@dataclass(frozen=True)
class RafWheelStatic:
    """Static data for a single wheel."""

    index: int
    x: float
    y: float
    rim_radius: float
    tyre_radius: float
    tyre_width: float
    tyre_height: float
    spring_rate: float
    bump_damping: float
    rebound_damping: float
    suspension_travel: float


@dataclass(frozen=True)
class RafWheelFrame:
    """Per-frame telemetry captured for a single wheel."""

    slip_ratio: float
    slip_angle: float
    lateral_force: float
    longitudinal_force: float
    vertical_load: float
    angular_speed: float
    suspension_deflection: float


@dataclass(frozen=True)
class RafFrame:
    """Single telemetry sample extracted from the RAF stream."""

    index: int
    position: tuple[float, float, float]
    speed: float
    distance: float
    engine_rpm: float
    gear: int | None
    pitch: float
    wheels: tuple[RafWheelFrame, ...]


@dataclass(frozen=True)
class RafFile:
    """Fully parsed RAF file."""

    header: RafHeader
    car: RafCarStatic
    wheels: tuple[RafWheelStatic, ...]
    frames: tuple[RafFrame, ...]


def read_raf(path: Path | str) -> RafFile:
    """Parse ``path`` into a :class:`RafFile` instance.

    The loader validates the RAF magic, the reported block sizes and the overall
    file size before decoding the static configuration and all telemetry frames.
    """

    source = Path(path)
    with source.open("rb") as handle:
        prefix = handle.read(24)
        if len(prefix) < 24:
            raise ValueError("RAF file is truncated")

        magic = prefix[:6]
        if magic != _MAGIC:
            raise ValueError("Invalid RAF magic header")

        raw_version = _U16.unpack_from(prefix, 6)[0]
        interval_raw = _U16.unpack_from(prefix, 8)[0]
        header_size = _U16.unpack_from(prefix, 12)[0]
        frame_size = _U16.unpack_from(prefix, 14)[0]
        car_static_words = _U16.unpack_from(prefix, 16)[0]
        wheel_static_words = _U16.unpack_from(prefix, 18)[0]
        frame_count = struct.unpack_from("<I", prefix, 20)[0]

        header_rest = handle.read(header_size - len(prefix))
        if len(header_rest) + len(prefix) != header_size:
            raise ValueError("RAF header is truncated")
        header_bytes = prefix + header_rest

        expected_size = header_size + frame_size * frame_count
        actual_size = source.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                "RAF size mismatch: expected %d bytes, found %d"
                % (expected_size, actual_size)
            )

        header = _parse_header(
            header_bytes,
            raw_version=raw_version,
            interval_raw=interval_raw,
            header_size=header_size,
            frame_size=frame_size,
            car_static_words=car_static_words,
            wheel_static_words=wheel_static_words,
            frame_count=frame_count,
        )

        wheel_statics = _parse_wheel_statics(header_bytes, header)
        car_static = _parse_car_static(header_bytes, header, wheel_statics)

        frames = _parse_frames(handle, header, wheel_statics, car_static)

    return RafFile(header=header, car=car_static, wheels=wheel_statics, frames=frames)


def raf_to_telemetry_records(
    raf: RafFile, *, metadata: Mapping[str, object] | None = None
) -> list[TelemetryRecord]:
    """Convert a parsed RAF file to :class:`TelemetryRecord` samples."""

    wheel_order = _resolve_wheel_order(raf.wheels)
    interval = raf.header.interval_seconds
    front_track_width = raf.car.front_track_width if raf.car.front_track_width else math.nan
    wheelbase = raf.car.wheelbase if raf.car.wheelbase else math.nan
    tyre_compound = normalise_compound_label(resolve_compound_metadata(metadata))

    records: list[TelemetryRecord] = []
    for frame in raf.frames:
        timestamp = frame.index * interval

        wheel_data: dict[str, RafWheelFrame] = {
            slot: frame.wheels[idx] for slot, idx in wheel_order.items()
        }

        total_vertical_load = sum(w.vertical_load for w in frame.wheels)
        front_vertical_load = sum(
            wheel_data[key].vertical_load for key in ("fl", "fr") if key in wheel_data
        )
        rear_vertical_load = sum(
            wheel_data[key].vertical_load for key in ("rl", "rr") if key in wheel_data
        )

        slip_ratio = sum(w.slip_ratio for w in frame.wheels) / max(len(frame.wheels), 1)
        slip_angle = sum(w.slip_angle for w in frame.wheels) / max(len(frame.wheels), 1)

        front_deflections = [wheel_data[key].suspension_deflection for key in ("fl", "fr")]
        rear_deflections = [wheel_data[key].suspension_deflection for key in ("rl", "rr")]

        avg_front_deflection = (
            sum(front_deflections) / len(front_deflections) if front_deflections else math.nan
        )
        avg_rear_deflection = (
            sum(rear_deflections) / len(rear_deflections) if rear_deflections else math.nan
        )

        rpm = frame.engine_rpm if not math.isnan(frame.engine_rpm) else math.nan

        car_model = raf.header.car_model or None
        track_name = raf.header.track_name or None

        record = TelemetryRecord(
            timestamp=timestamp,
            vertical_load=total_vertical_load,
            slip_ratio=slip_ratio,
            lateral_accel=math.nan,
            longitudinal_accel=math.nan,
            yaw=math.nan,
            pitch=frame.pitch,
            roll=math.nan,
            brake_pressure=math.nan,
            locking=math.nan,
            nfr=math.nan,
            si=math.nan,
            speed=frame.speed,
            yaw_rate=math.nan,
            slip_angle=slip_angle,
            steer=math.nan,
            throttle=math.nan,
            gear=frame.gear or 0,
            vertical_load_front=front_vertical_load,
            vertical_load_rear=rear_vertical_load,
            mu_eff_front=math.nan,
            mu_eff_rear=math.nan,
            mu_eff_front_lateral=math.nan,
            mu_eff_front_longitudinal=math.nan,
            mu_eff_rear_lateral=math.nan,
            mu_eff_rear_longitudinal=math.nan,
            suspension_travel_front=avg_front_deflection,
            suspension_travel_rear=avg_rear_deflection,
            suspension_velocity_front=math.nan,
            suspension_velocity_rear=math.nan,
            slip_ratio_fl=wheel_data["fl"].slip_ratio,
            slip_ratio_fr=wheel_data["fr"].slip_ratio,
            slip_ratio_rl=wheel_data["rl"].slip_ratio,
            slip_ratio_rr=wheel_data["rr"].slip_ratio,
            slip_angle_fl=wheel_data["fl"].slip_angle,
            slip_angle_fr=wheel_data["fr"].slip_angle,
            slip_angle_rl=wheel_data["rl"].slip_angle,
            slip_angle_rr=wheel_data["rr"].slip_angle,
            brake_input=math.nan,
            clutch_input=math.nan,
            handbrake_input=math.nan,
            steer_input=math.nan,
            wheel_load_fl=wheel_data["fl"].vertical_load,
            wheel_load_fr=wheel_data["fr"].vertical_load,
            wheel_load_rl=wheel_data["rl"].vertical_load,
            wheel_load_rr=wheel_data["rr"].vertical_load,
            wheel_lateral_force_fl=wheel_data["fl"].lateral_force,
            wheel_lateral_force_fr=wheel_data["fr"].lateral_force,
            wheel_lateral_force_rl=wheel_data["rl"].lateral_force,
            wheel_lateral_force_rr=wheel_data["rr"].lateral_force,
            wheel_longitudinal_force_fl=wheel_data["fl"].longitudinal_force,
            wheel_longitudinal_force_fr=wheel_data["fr"].longitudinal_force,
            wheel_longitudinal_force_rl=wheel_data["rl"].longitudinal_force,
            wheel_longitudinal_force_rr=wheel_data["rr"].longitudinal_force,
            suspension_deflection_fl=wheel_data["fl"].suspension_deflection,
            suspension_deflection_fr=wheel_data["fr"].suspension_deflection,
            suspension_deflection_rl=wheel_data["rl"].suspension_deflection,
            suspension_deflection_rr=wheel_data["rr"].suspension_deflection,
            rpm=rpm,
            line_deviation=math.nan,
            instantaneous_radius=math.nan,
            front_track_width=front_track_width,
            wheelbase=wheelbase,
            lap=None,
            structural_timestamp=timestamp,
            car_model=car_model,
            track_name=track_name,
            tyre_compound=tyre_compound,
        )

        records.append(record)

    return records


def _parse_header(
    header: bytes,
    *,
    raw_version: int,
    interval_raw: int,
    header_size: int,
    frame_size: int,
    car_static_words: int,
    wheel_static_words: int,
    frame_count: int,
) -> RafHeader:
    """Decode the header metadata block."""

    version = raw_version >> 8
    interval_ms = interval_raw / 256.0

    track_code = header[24:28].split(b"\0", 1)[0].decode("latin-1")
    player_name = _read_fixed_string(header, 30, 32)
    car_model = _read_fixed_string(header, 64, 32)
    track_name = _read_fixed_string(header, 96, 32)
    layout_name = _read_fixed_string(header, 128, 16)
    session_name = _read_fixed_string(header, 144, 16)

    return RafHeader(
        magic=_MAGIC.decode("ascii"),
        version=version,
        raw_version=raw_version,
        interval_ms=interval_ms,
        header_size=header_size,
        frame_size=frame_size,
        car_static_words=car_static_words,
        wheel_static_words=wheel_static_words,
        frame_count=frame_count,
        track_code=track_code,
        player_name=player_name,
        car_model=car_model,
        track_name=track_name,
        track_layout=layout_name,
        session_name=session_name,
    )


def _parse_car_static(
    header: bytes,
    metadata: RafHeader,
    wheels: Sequence[RafWheelStatic],
) -> RafCarStatic:
    """Extract static car data from the header."""

    idle_rpm = _F32.unpack_from(header, 188)[0]
    mass_kg = _F32.unpack_from(header, 192)[0]
    peak_torque_rpm = _F32.unpack_from(header, 196)[0] / 10.0
    rev_limiter_rpm = _F32.unpack_from(header, 200)[0] / 10.0
    final_drive = _F32.unpack_from(header, 204)[0]

    ratios: list[float] = []
    ratios_offset = 212
    ratios_end = min(ratios_offset + metadata.car_static_words * 2, len(header))
    for offset in range(ratios_offset, ratios_end, 4):
        value = _F32.unpack_from(header, offset)[0]
        if value == 0.0:
            break
        ratios.append(value)

    order = _resolve_wheel_order(wheels)
    front_left = wheels[order["fl"]]
    front_right = wheels[order["fr"]]
    rear_left = wheels[order["rl"]]
    rear_right = wheels[order["rr"]]

    front_track_width = abs(front_right.x - front_left.x)

    front_axle_y = (front_left.y + front_right.y) / 2.0
    rear_axle_y = (rear_left.y + rear_right.y) / 2.0
    wheelbase = abs(rear_axle_y - front_axle_y)

    return RafCarStatic(
        idle_rpm=idle_rpm,
        mass_kg=mass_kg,
        peak_torque_rpm=peak_torque_rpm,
        rev_limiter_rpm=rev_limiter_rpm,
        final_drive_ratio=final_drive,
        gear_ratios=tuple(ratios),
        front_track_width=front_track_width,
        wheelbase=wheelbase,
    )


def _parse_wheel_statics(header: bytes, metadata: RafHeader) -> tuple[RafWheelStatic, ...]:
    """Parse the static data for each wheel."""

    block_size = metadata.wheel_static_words * 2
    wheels: list[RafWheelStatic] = []

    offset = metadata.header_size
    while offset >= block_size:
        offset -= block_size
        block = header[offset : offset + block_size]
        if not any(block):
            if wheels:
                break
            continue
        floats = struct.unpack("<32f", block)
        wheels.append(
            RafWheelStatic(
                index=len(wheels),
                x=floats[0],
                y=floats[1],
                rim_radius=floats[2],
                tyre_radius=floats[3],
                tyre_width=floats[4],
                tyre_height=floats[5],
                spring_rate=floats[8],
                bump_damping=floats[9],
                rebound_damping=floats[10],
                suspension_travel=floats[11],
            )
        )

    wheels.reverse()
    return tuple(wheels)


def _parse_frames(
    handle: BinaryIO,
    metadata: RafHeader,
    wheels: Sequence[RafWheelStatic],
    car: RafCarStatic,
) -> tuple[RafFrame, ...]:
    """Parse all telemetry frames."""

    frames: list[RafFrame] = []
    wheel_count = len(wheels)
    wheel_block_size = 8 * 4

    for index in range(metadata.frame_count):
        data = handle.read(metadata.frame_size)
        if len(data) != metadata.frame_size:
            raise ValueError("RAF frame %d truncated" % index)

        pitch = _F32.unpack_from(data, 8)[0]
        speed = _F32.unpack_from(data, 24)[0]
        pos_x = _I32.unpack_from(data, 32)[0] / 65536.0
        pos_y = _I32.unpack_from(data, 36)[0] / 65536.0
        pos_z = _I32.unpack_from(data, 40)[0] / 65536.0
        engine_rad = _F32.unpack_from(data, 44)[0]
        distance = _F32.unpack_from(data, 48)[0]

        engine_rpm = engine_rad * 60.0 / math.tau if engine_rad else 0.0

        wheel_frames: list[RafWheelFrame] = []
        start = 64
        for wheel_index in range(wheel_count):
            block_offset = start + wheel_index * wheel_block_size
            (
                slip_ratio,
                slip_angle,
                lateral_force,
                longitudinal_force,
                vertical_load,
                angular_speed,
                suspension_deflection,
                _,
            ) = struct.unpack_from("<8f", data, block_offset)
            wheel_frames.append(
                RafWheelFrame(
                    slip_ratio=slip_ratio,
                    slip_angle=slip_angle,
                    lateral_force=lateral_force,
                    longitudinal_force=longitudinal_force,
                    vertical_load=vertical_load,
                    angular_speed=angular_speed,
                    suspension_deflection=suspension_deflection,
                )
            )

        gear = _estimate_gear(engine_rpm, wheel_frames, car)

        frames.append(
            RafFrame(
                index=index,
                position=(pos_x, pos_y, pos_z),
                speed=speed,
                distance=distance,
                engine_rpm=engine_rpm,
                gear=gear,
                pitch=pitch,
                wheels=tuple(wheel_frames),
            )
        )

    return tuple(frames)


def _estimate_gear(
    engine_rpm: float, wheels: Sequence[RafWheelFrame], car: RafCarStatic
) -> int | None:
    if engine_rpm <= 0:
        return None

    driven_speed = sum(w.angular_speed for w in wheels) / max(len(wheels), 1)
    if driven_speed <= 0:
        return None

    wheel_rpm = driven_speed * 60.0 / math.tau

    if wheel_rpm <= 0:
        return None

    if not car.gear_ratios or car.final_drive_ratio <= 0:
        return None

    ratios = [ratio * car.final_drive_ratio for ratio in car.gear_ratios]
    ratio = engine_rpm / wheel_rpm
    best: tuple[float, int] | None = None
    for idx, expected in enumerate(ratios, start=1):
        diff = abs(ratio - expected)
        if best is None or diff < best[0]:
            best = (diff, idx)

    if best and ratios[best[1] - 1] and best[0] / ratios[best[1] - 1] < 0.25:
        return best[1]
    return None


def _resolve_wheel_order(wheels: Sequence[RafWheelStatic]) -> Mapping[str, int]:
    if len(wheels) != 4:
        raise ValueError("Only four-wheel RAF files are currently supported")

    sorted_indices = sorted(range(len(wheels)), key=lambda idx: wheels[idx].y)
    front_indices = sorted_indices[:2]
    rear_indices = sorted_indices[2:]

    front_left = min(front_indices, key=lambda idx: wheels[idx].x)
    front_right = max(front_indices, key=lambda idx: wheels[idx].x)
    rear_left = min(rear_indices, key=lambda idx: wheels[idx].x)
    rear_right = max(rear_indices, key=lambda idx: wheels[idx].x)

    return {"fl": front_left, "fr": front_right, "rl": rear_left, "rr": rear_right}


def _read_fixed_string(buffer: bytes, offset: int, length: int) -> str:
    raw = buffer[offset : offset + length]
    return raw.split(b"\0", 1)[0].decode("latin-1").strip()

