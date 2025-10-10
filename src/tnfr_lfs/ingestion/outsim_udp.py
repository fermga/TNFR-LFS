"""OutSim UDP client implementation.

The Live for Speed simulator broadcasts physics-oriented telemetry using
the OutSim protocol.  Packets are transmitted as a binary structure over
UDP and contain orientation, acceleration and position data.  This
module provides a small client capable of decoding the official packet
layout while operating in non-blocking mode so that it can be safely
polled from high-frequency loops.

``OutSimUDPClient`` keeps a short reordering buffer keyed by the
``OutSimPacket.time`` field, automatically deduplicating, re-sequencing
and tracking suspected gaps.  Applications may call
:meth:`OutSimUDPClient.drain_ready` to extract the current buffer without
blocking and inspect :attr:`OutSimUDPClient.statistics` to monitor
packet loss or recovery counters emitted while the client operates.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import logging
from types import TracebackType
import socket
import struct
import time
from typing import Optional, Tuple

from tnfr_lfs.ingestion._socket_poll import wait_for_read_ready
from tnfr_lfs.ingestion._reorder_buffer import (
    CircularReorderBuffer,
    DEFAULT_REORDER_BUFFER_SIZE,
)
from tnfr_lfs.ingestion.pools import PacketPool, PoolItem

__all__ = [
    "OUTSIM_MAX_PACKET_SIZE",
    "AsyncOutSimUDPClient",
    "OutSimDriverInputs",
    "FrozenOutSimDriverInputs",
    "OutSimPacket",
    "FrozenOutSimPacket",
    "OutSimUDPClient",
    "OutSimWheelState",
    "FrozenOutSimWheelState",
]


logger = logging.getLogger(__name__)


_BASE_STRUCT = struct.Struct("<I15f")
_ID_STRUCT = struct.Struct("<I15fI")
_INPUT_STRUCT = struct.Struct("<5f")
_WHEEL_STRUCT = struct.Struct("<6f")

OUTSIM_MAX_PACKET_SIZE = (
    _BASE_STRUCT.size
    + struct.calcsize("<I")
    + _INPUT_STRUCT.size
    + 4 * _WHEEL_STRUCT.size
)


@dataclass(frozen=True)
class FrozenOutSimDriverInputs:
    throttle: float = 0.0
    brake: float = 0.0
    clutch: float = 0.0
    handbrake: float = 0.0
    steer: float = 0.0


@dataclass(frozen=True)
class FrozenOutSimWheelState:
    slip_ratio: float = 0.0
    slip_angle: float = 0.0
    longitudinal_force: float = 0.0
    lateral_force: float = 0.0
    load: float = 0.0
    suspension_deflection: float = 0.0
    decoded: bool = False


@dataclass(frozen=True)
class FrozenOutSimPacket:
    time: int
    ang_vel_x: float
    ang_vel_y: float
    ang_vel_z: float
    heading: float
    pitch: float
    roll: float
    accel_x: float
    accel_y: float
    accel_z: float
    vel_x: float
    vel_y: float
    vel_z: float
    pos_x: float
    pos_y: float
    pos_z: float
    player_id: Optional[int] = None
    inputs: Optional[FrozenOutSimDriverInputs] = None
    wheels: Tuple[
        FrozenOutSimWheelState,
        FrozenOutSimWheelState,
        FrozenOutSimWheelState,
        FrozenOutSimWheelState,
    ] = (
        FrozenOutSimWheelState(),
        FrozenOutSimWheelState(),
        FrozenOutSimWheelState(),
        FrozenOutSimWheelState(),
    )


class OutSimDriverInputs:
    __slots__ = ("throttle", "brake", "clutch", "handbrake", "steer", "_present")

    def __init__(
        self,
        *,
        throttle: float = 0.0,
        brake: float = 0.0,
        clutch: float = 0.0,
        handbrake: float = 0.0,
        steer: float = 0.0,
    ) -> None:
        self.throttle = 0.0
        self.brake = 0.0
        self.clutch = 0.0
        self.handbrake = 0.0
        self.steer = 0.0
        self._present = False
        if any(value != 0.0 for value in (throttle, brake, clutch, handbrake, steer)):
            self.set_values(throttle, brake, clutch, handbrake, steer)

    def clear(self) -> None:
        self.throttle = 0.0
        self.brake = 0.0
        self.clutch = 0.0
        self.handbrake = 0.0
        self.steer = 0.0
        self._present = False

    def set_values(
        self,
        throttle: float,
        brake: float,
        clutch: float,
        handbrake: float,
        steer: float,
    ) -> None:
        self.throttle = throttle
        self.brake = brake
        self.clutch = clutch
        self.handbrake = handbrake
        self.steer = steer
        self._present = True

    def freeze(self) -> FrozenOutSimDriverInputs:
        return FrozenOutSimDriverInputs(
            throttle=self.throttle,
            brake=self.brake,
            clutch=self.clutch,
            handbrake=self.handbrake,
            steer=self.steer,
        )


class OutSimWheelState:
    __slots__ = (
        "slip_ratio",
        "slip_angle",
        "longitudinal_force",
        "lateral_force",
        "load",
        "suspension_deflection",
        "decoded",
    )

    def __init__(
        self,
        *,
        slip_ratio: float = 0.0,
        slip_angle: float = 0.0,
        longitudinal_force: float = 0.0,
        lateral_force: float = 0.0,
        load: float = 0.0,
        suspension_deflection: float = 0.0,
        decoded: bool = False,
    ) -> None:
        self.slip_ratio = slip_ratio
        self.slip_angle = slip_angle
        self.longitudinal_force = longitudinal_force
        self.lateral_force = lateral_force
        self.load = load
        self.suspension_deflection = suspension_deflection
        self.decoded = decoded

    def clear(self) -> None:
        self.slip_ratio = 0.0
        self.slip_angle = 0.0
        self.longitudinal_force = 0.0
        self.lateral_force = 0.0
        self.load = 0.0
        self.suspension_deflection = 0.0
        self.decoded = False

    def set_values(
        self,
        slip_ratio: float,
        slip_angle: float,
        long_force: float,
        lat_force: float,
        load: float,
        suspension_deflection: float,
    ) -> None:
        self.slip_ratio = slip_ratio
        self.slip_angle = slip_angle
        self.longitudinal_force = long_force
        self.lateral_force = lat_force
        self.load = load
        self.suspension_deflection = suspension_deflection
        self.decoded = True

    def freeze(self) -> FrozenOutSimWheelState:
        return FrozenOutSimWheelState(
            slip_ratio=self.slip_ratio,
            slip_angle=self.slip_angle,
            longitudinal_force=self.longitudinal_force,
            lateral_force=self.lateral_force,
            load=self.load,
            suspension_deflection=self.suspension_deflection,
            decoded=self.decoded,
        )


class OutSimPacket(PoolItem):
    """Representation of a decoded OutSim datagram."""

    __slots__ = (
        "time",
        "ang_vel_x",
        "ang_vel_y",
        "ang_vel_z",
        "heading",
        "pitch",
        "roll",
        "accel_x",
        "accel_y",
        "accel_z",
        "vel_x",
        "vel_y",
        "vel_z",
        "pos_x",
        "pos_y",
        "pos_z",
        "player_id",
        "_inputs",
        "_has_inputs",
        "_wheels",
    )

    def __init__(
        self,
        *,
        time: int = 0,
        ang_vel_x: float = 0.0,
        ang_vel_y: float = 0.0,
        ang_vel_z: float = 0.0,
        heading: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        accel_x: float = 0.0,
        accel_y: float = 0.0,
        accel_z: float = 0.0,
        vel_x: float = 0.0,
        vel_y: float = 0.0,
        vel_z: float = 0.0,
        pos_x: float = 0.0,
        pos_y: float = 0.0,
        pos_z: float = 0.0,
        player_id: Optional[int] = None,
        inputs: Optional[object] = None,
        wheels: Tuple[object, object, object, object] | tuple[object, ...] = (),
    ) -> None:
        super().__init__()
        self._inputs = OutSimDriverInputs()
        self._wheels = (
            OutSimWheelState(),
            OutSimWheelState(),
            OutSimWheelState(),
            OutSimWheelState(),
        )
        self._reset_values()
        self.time = int(time)
        self.ang_vel_x = float(ang_vel_x)
        self.ang_vel_y = float(ang_vel_y)
        self.ang_vel_z = float(ang_vel_z)
        self.heading = float(heading)
        self.pitch = float(pitch)
        self.roll = float(roll)
        self.accel_x = float(accel_x)
        self.accel_y = float(accel_y)
        self.accel_z = float(accel_z)
        self.vel_x = float(vel_x)
        self.vel_y = float(vel_y)
        self.vel_z = float(vel_z)
        self.pos_x = float(pos_x)
        self.pos_y = float(pos_y)
        self.pos_z = float(pos_z)
        self.player_id = player_id
        if inputs is not None:
            self.inputs = inputs  # type: ignore[assignment]
        if wheels:
            self._assign_wheels(wheels)

    @property
    def inputs(self) -> Optional[OutSimDriverInputs]:
        return self._inputs if self._has_inputs else None

    @inputs.setter
    def inputs(self, value: Optional[OutSimDriverInputs]) -> None:
        if value is None:
            self._inputs.clear()
            self._has_inputs = False
        else:
            self._inputs.set_values(
                value.throttle,
                value.brake,
                value.clutch,
                value.handbrake,
                value.steer,
            )
            self._has_inputs = True

    @property
    def wheels(
        self,
    ) -> Tuple[OutSimWheelState, OutSimWheelState, OutSimWheelState, OutSimWheelState]:
        return self._wheels

    def _assign_wheels(self, wheels: tuple[object, ...] | Tuple[object, object, object, object]) -> None:
        for index in range(4):
            target = self._wheels[index]
            if index < len(wheels):
                source = wheels[index]
                target.slip_ratio = float(getattr(source, "slip_ratio", 0.0))
                target.slip_angle = float(getattr(source, "slip_angle", 0.0))
                target.longitudinal_force = float(getattr(source, "longitudinal_force", 0.0))
                target.lateral_force = float(getattr(source, "lateral_force", 0.0))
                target.load = float(getattr(source, "load", 0.0))
                target.suspension_deflection = float(getattr(source, "suspension_deflection", 0.0))
                target.decoded = bool(getattr(source, "decoded", False))
            else:
                target.clear()

    def _reset_values(self) -> None:
        self.time = 0
        self.ang_vel_x = 0.0
        self.ang_vel_y = 0.0
        self.ang_vel_z = 0.0
        self.heading = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0
        self.player_id = None
        self._inputs.clear()
        self._has_inputs = False
        for wheel in self._wheels:
            wheel.clear()

    def _reset(self) -> None:
        self._reset_values()

    def _populate(
        self,
        base_values: tuple[object, ...],
        player_id: Optional[int],
        inputs_values: Optional[tuple[float, float, float, float, float]],
        wheel_values: list[tuple[float, float, float, float, float, float, bool]],
    ) -> None:
        (
            self.time,
            self.ang_vel_x,
            self.ang_vel_y,
            self.ang_vel_z,
            self.heading,
            self.pitch,
            self.roll,
            self.accel_x,
            self.accel_y,
            self.accel_z,
            self.vel_x,
            self.vel_y,
            self.vel_z,
            self.pos_x,
            self.pos_y,
            self.pos_z,
        ) = base_values  # type: ignore[assignment]
        self.player_id = player_id
        if inputs_values is None:
            self._inputs.clear()
            self._has_inputs = False
        else:
            self._inputs.set_values(*inputs_values)
            self._has_inputs = True
        for wheel, values in zip(self._wheels, wheel_values):
            if not values[-1]:
                wheel.clear()
                continue
            slip_ratio, slip_angle, long_force, lat_force, load, deflection, _ = values
            wheel.set_values(slip_ratio, slip_angle, long_force, lat_force, load, deflection)
        for remaining in range(len(wheel_values), 4):
            self._wheels[remaining].clear()

    def freeze(self) -> FrozenOutSimPacket:
        return FrozenOutSimPacket(
            time=self.time,
            ang_vel_x=self.ang_vel_x,
            ang_vel_y=self.ang_vel_y,
            ang_vel_z=self.ang_vel_z,
            heading=self.heading,
            pitch=self.pitch,
            roll=self.roll,
            accel_x=self.accel_x,
            accel_y=self.accel_y,
            accel_z=self.accel_z,
            vel_x=self.vel_x,
            vel_y=self.vel_y,
            vel_z=self.vel_z,
            pos_x=self.pos_x,
            pos_y=self.pos_y,
            pos_z=self.pos_z,
            player_id=self.player_id,
            inputs=self._inputs.freeze() if self._has_inputs else None,
            wheels=tuple(wheel.freeze() for wheel in self._wheels),
        )

    @classmethod
    def from_bytes(
        cls, payload: bytes, *, freeze: bool = False
    ) -> "OutSimPacket | FrozenOutSimPacket":
        """Deserialize a byte payload following the official layout."""

        if len(payload) < _BASE_STRUCT.size:
            raise ValueError(
                f"OutSim payload too small: {len(payload)} bytes (expected {_BASE_STRUCT.size})"
            )

        base_values = _BASE_STRUCT.unpack_from(payload)
        offset = _BASE_STRUCT.size

        player_id: Optional[int] = None
        if len(payload) >= offset + struct.calcsize("<I"):
            candidate_id = struct.unpack_from("<I", payload, offset)[0]
            if candidate_id <= 1_000_000:
                player_id = candidate_id
                offset += struct.calcsize("<I")

        inputs_values: Optional[tuple[float, float, float, float, float]] = None
        if len(payload) >= offset + _INPUT_STRUCT.size:
            inputs_values = _INPUT_STRUCT.unpack_from(payload, offset)
            offset += _INPUT_STRUCT.size

        wheel_values: list[tuple[float, float, float, float, float, float, bool]] = []
        for _ in range(4):
            if len(payload) < offset + _WHEEL_STRUCT.size:
                break
            slip_ratio, slip_angle, long_force, lat_force, load, deflection = (
                _WHEEL_STRUCT.unpack_from(payload, offset)
            )
            wheel_values.append(
                (
                    slip_ratio,
                    slip_angle,
                    long_force,
                    lat_force,
                    load,
                    deflection,
                    True,
                )
            )
            offset += _WHEEL_STRUCT.size

        if freeze:
            wheels = [
                FrozenOutSimWheelState(
                    slip_ratio=values[0],
                    slip_angle=values[1],
                    longitudinal_force=values[2],
                    lateral_force=values[3],
                    load=values[4],
                    suspension_deflection=values[5],
                    decoded=values[6],
                )
                for values in wheel_values
            ]
            while len(wheels) < 4:
                wheels.append(FrozenOutSimWheelState())
            return FrozenOutSimPacket(
                *base_values,
                player_id=player_id,
                inputs=(
                    FrozenOutSimDriverInputs(*inputs_values)
                    if inputs_values is not None
                    else None
                ),
                wheels=tuple(wheels[:4]),
            )

        packet = _OUTSIM_POOL.acquire()
        packet._populate(base_values, player_id, inputs_values, wheel_values)
        return packet


_OUTSIM_POOL: PacketPool[OutSimPacket] = PacketPool(OutSimPacket)


class _OutSimPacketProcessor:
    """Shared buffering and accounting logic for OutSim UDP clients."""

    def __init__(
        self,
        *,
        remote_host: str,
        port: int,
        buffer_capacity: int,
        buffer_grace: float,
        jump_tolerance: int,
    ) -> None:
        self._remote_host = remote_host
        self._port = port
        self._buffer = CircularReorderBuffer[OutSimPacket](buffer_capacity)
        self._buffer_grace = buffer_grace
        self._jump_tolerance = jump_tolerance
        self._received_packets = 0
        self._delivered_packets = 0
        self._duplicate_packets = 0
        self._out_of_order_packets = 0
        self._late_reordered_packets = 0
        self._loss_events = 0
        self._last_emitted_time: Optional[int] = None
        self._last_seen_time: Optional[int] = None
        self._reordered_times: set[int] = set()
        self._pending_buffered = 0
        self._pending_deadline: float | None = None

    @property
    def statistics(self) -> dict[str, int]:
        return {
            "received": self._received_packets,
            "delivered": self._delivered_packets,
            "duplicates": self._duplicate_packets,
            "reordered": self._out_of_order_packets,
            "late_recovered": self._late_reordered_packets,
            "loss_events": self._loss_events,
        }

    def record_packet(self, packet: OutSimPacket, arrival: float) -> bool:
        self._received_packets += 1

        if self._last_seen_time is not None and packet.time < self._last_seen_time:
            self._out_of_order_packets += 1
            self._reordered_times.add(packet.time)
            logger.warning(
                "OutSim out-of-order packet received.",
                extra={
                    "event": "outsim.out_of_order",
                    "last_time": self._last_seen_time,
                    "current_time": packet.time,
                    "port": self._port,
                    "remote_host": self._remote_host,
                },
            )
        self._last_seen_time = (
            packet.time
            if self._last_seen_time is None
            else max(self._last_seen_time, packet.time)
        )

        if self._is_duplicate(packet.time):
            self._duplicate_packets += 1
            logger.warning(
                "OutSim duplicate packet dropped.",
                extra={
                    "event": "outsim.duplicate",
                    "time": packet.time,
                    "port": self._port,
                    "remote_host": self._remote_host,
                },
            )
            packet.release()
            return False

        index, evicted = self._buffer.insert(arrival, packet, packet.time)
        if evicted is not None and hasattr(evicted.packet, "release"):
            evicted.packet.release()
        current_length = len(self._buffer)
        if (
            current_length > 1
            and index < (current_length - 1)
            and packet.time not in self._reordered_times
        ):
            self._reordered_times.add(packet.time)
        if self._pending_buffered:
            self._pending_buffered = min(self._pending_buffered, current_length)
            if self._pending_buffered == 0:
                self._pending_deadline = None
        return True

    def pop_ready_packet(self, now: float, *, allow_grace: bool) -> Optional[OutSimPacket]:
        while self._buffer:
            peeked = self._buffer.peek_oldest()
            if peeked is None:
                break
            arrival, packet = peeked
            if (
                not allow_grace
                and len(self._buffer) == 1
                and self._last_emitted_time is not None
                and (now - arrival) < self._buffer_grace
            ):
                self._pending_buffered = len(self._buffer)
                self._pending_deadline = arrival + self._buffer_grace
                break
            popped = self._buffer.pop_oldest()
            if popped is None:
                break
            _, packet = popped
            if self._pending_buffered:
                self._pending_buffered = max(0, self._pending_buffered - 1)
                if self._pending_buffered == 0:
                    self._pending_deadline = None
            if (
                self._last_emitted_time is not None
                and packet.time <= self._last_emitted_time
            ):
                packet.release()
                continue
            if self._jump_tolerance and self._last_emitted_time is not None:
                delta = packet.time - self._last_emitted_time
                if delta > self._jump_tolerance:
                    self._loss_events += 1
                    logger.warning(
                        "OutSim time jump detected (possible packet loss).",
                        extra={
                            "event": "outsim.packet_gap",
                            "delta": delta,
                            "last_time": self._last_emitted_time,
                            "current_time": packet.time,
                            "port": self._port,
                            "remote_host": self._remote_host,
                        },
                    )
            if packet.time in self._reordered_times:
                self._reordered_times.discard(packet.time)
                self._late_reordered_packets += 1
            self._delivered_packets += 1
            self._last_emitted_time = packet.time
            return packet
        if not self._buffer:
            self._pending_buffered = 0
            self._pending_deadline = None
        return None

    def drain_ready(self, now: float) -> list[OutSimPacket]:
        ready: list[OutSimPacket] = []
        while True:
            packet = self.pop_ready_packet(now, allow_grace=False)
            if packet is None:
                break
            ready.append(packet)
            now = time.monotonic()
        return ready

    @property
    def pending_buffered(self) -> int:
        return self._pending_buffered

    def pending_deadline(self) -> float | None:
        if self._pending_buffered <= 0:
            return None
        return self._pending_deadline

    def _is_duplicate(self, timestamp: int) -> bool:
        if self._last_emitted_time is not None and timestamp == self._last_emitted_time:
            return True
        return self._buffer.contains_key(timestamp)

    def flush(self) -> None:
        while True:
            popped = self._buffer.pop_oldest()
            if popped is None:
                break
            _, packet = popped
            if hasattr(packet, "release"):
                packet.release()
        self._reordered_times.clear()
        self._pending_buffered = 0
        self._pending_deadline = None


class OutSimUDPClient:
    """Non-blocking UDP client that yields :class:`OutSimPacket` objects."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4123,
        *,
        timeout: float = 0.05,
        retries: int = 5,
        reorder_grace: float | None = None,
        jump_tolerance: int = 200,
        buffer_size: int | None = None,
        max_batch: int | None = None,
    ) -> None:
        """Create a UDP client bound to the local host.

        Parameters
        ----------
        host:
            Remote host expected to provide the OutSim datagrams.  The client
            still binds locally, accepting packets from all interfaces unless a
            remote host is provided, in which case only datagrams originating
            from that host are processed.
        port:
            UDP port for both binding locally and filtering remote packets.
        timeout:
            Sleep interval between retry attempts when no packet is available.
        retries:
            Maximum number of non-blocking reads issued per :meth:`recv`
            invocation before considering the call a timeout.
        reorder_grace:
            Optional number of seconds to retain a lone packet inside the
            internal buffer so late datagrams with smaller timestamps can be
            slotted ahead of it.  ``None`` defaults to ``timeout``.
        jump_tolerance:
            Maximum tolerated gap (in milliseconds) between successive packet
            timestamps before the client flags a suspected loss event.
        buffer_size:
            Optional maximum number of packets kept in the internal reordering
            buffer. ``None`` leaves the buffer unbounded.
        max_batch:
            Optional maximum number of datagrams drained from the socket after
            a readiness notification.  ``None`` keeps draining until the socket
            raises :class:`BlockingIOError`.
        """

        buffer_capacity = DEFAULT_REORDER_BUFFER_SIZE
        if buffer_size is not None:
            try:
                numeric = int(buffer_size)
            except (TypeError, ValueError):
                numeric = 0
            if numeric > 0:
                buffer_capacity = numeric
        batch_limit = None
        if max_batch is not None:
            try:
                numeric = int(max_batch)
            except (TypeError, ValueError):
                numeric = 0
            if numeric > 0:
                batch_limit = numeric
        self._remote_host = host
        self._remote_addresses = self._resolve_remote_addresses(host)
        self._timeout = timeout
        self._retries = retries
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._socket.bind(("", port))
        local_host, local_port = self._socket.getsockname()
        self._address: Tuple[str, int] = (local_host, local_port)
        self._max_batch = batch_limit
        buffer_grace = max(float(reorder_grace) if reorder_grace is not None else timeout, 0.0)
        self._processor = _OutSimPacketProcessor(
            remote_host=self._remote_host,
            port=self._address[1],
            buffer_capacity=buffer_capacity,
            buffer_grace=buffer_grace,
            jump_tolerance=max(int(jump_tolerance), 0),
        )
        self._timeouts = 0
        self._ignored_hosts = 0

    @property
    def address(self) -> Tuple[str, int]:
        """Return the bound socket address."""

        return self._address

    @property
    def timeouts(self) -> int:
        """Number of calls that exhausted the retry budget."""

        return self._timeouts

    @property
    def ignored_hosts(self) -> int:
        """Number of datagrams dropped due to unexpected hosts."""

        return self._ignored_hosts

    @property
    def statistics(self) -> dict[str, int]:
        """Return packet accounting metrics collected by the client."""

        return self._processor.statistics

    def recv(self) -> Optional[OutSimPacket]:
        """Attempt to receive a packet, returning ``None`` on timeout."""

        now = time.monotonic()
        ready = self._processor.pop_ready_packet(now, allow_grace=False)
        if ready is not None:
            return ready

        deadline = None
        if self._timeout > 0.0:
            deadline = now + self._timeout

        for _ in range(self._retries):
            if self._drain_datagrams():
                ready = self._processor.pop_ready_packet(time.monotonic(), allow_grace=False)
                if ready is not None:
                    return ready
                if self._processor.pending_buffered:
                    break
                continue
            if self._processor.pending_buffered:
                break
            if not wait_for_read_ready(
                self._socket,
                timeout=self._timeout,
                deadline=deadline,
            ):
                break
        if self._processor.pending_buffered:
            deadline = self._processor.pending_deadline()
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(remaining, self._timeout if self._timeout > 0 else remaining))
        ready = self._processor.pop_ready_packet(time.monotonic(), allow_grace=True)
        if ready is not None:
            return ready
        self._timeouts += 1
        logger.warning(
            "OutSim recv retries exhausted without receiving a packet.",
            extra={
                "event": "outsim.recv_timeout",
                "retries": self._retries,
                "timeout": self._timeout,
                "remote_host": self._remote_host,
                "port": self._address[1],
            },
        )
        return None

    def drain_ready(self) -> list[OutSimPacket]:
        """Return all packets ready for immediate consumption.

        The method never blocks; it simply flushes the buffered packets whose
        arrival timestamps have aged beyond ``reorder_grace`` or which are
        already followed by newer datagrams.
        """

        return self._processor.drain_ready(time.monotonic())

    def _record_packet(self, packet: OutSimPacket) -> None:
        self._processor.record_packet(packet, time.monotonic())

    def _pop_ready_packet(self, *, now: float, allow_grace: bool) -> Optional[OutSimPacket]:
        return self._processor.pop_ready_packet(now, allow_grace=allow_grace)

    def _drain_datagrams(self) -> bool:
        """Drain ready datagrams into the reorder buffer."""

        processed = False
        limit = self._max_batch
        drained = 0
        while limit is None or drained < limit:
            try:
                payload, source = self._socket.recvfrom(OUTSIM_MAX_PACKET_SIZE)
            except BlockingIOError:
                break
            drained += 1
            processed = True
            if not payload:
                continue
            if self._remote_addresses and source[0] not in self._remote_addresses:
                self._ignored_hosts += 1
                logger.warning(
                    "Ignoring OutSim datagram from unexpected host.",
                    extra={
                        "event": "outsim.ignored_host",
                        "expected_hosts": sorted(self._remote_addresses),
                        "source_host": source[0],
                        "port": self._address[1],
                    },
                )
                continue
            packet = OutSimPacket.from_bytes(payload)
            self._processor.record_packet(packet, time.monotonic())
        return processed

    @staticmethod
    def _resolve_remote_addresses(host: str) -> set[str]:
        if not host or host in {"", "0.0.0.0"}:
            return set()
        try:
            info = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_DGRAM)
        except socket.gaierror:
            return {host}
        addresses = {record[4][0] for record in info if record[0] == socket.AF_INET}
        return addresses or {host}

    def close(self) -> None:
        """Close the underlying socket."""

        if self._processor is not None:
            self._processor.flush()
        self._socket.close()

    def __enter__(self) -> "OutSimUDPClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class _AsyncOutSimProtocol(asyncio.DatagramProtocol):
    def __init__(self, client: "AsyncOutSimUDPClient") -> None:
        self._client = client

    def connection_made(self, transport: asyncio.BaseTransport) -> None:  # pragma: no cover - exercised indirectly
        self._client._connection_made(transport)

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self._client._on_datagram(data, addr)

    def error_received(self, exc: Exception) -> None:  # pragma: no cover - defensive
        self._client._on_error(exc)

    def connection_lost(self, exc: Exception | None) -> None:
        self._client._connection_lost(exc)


class AsyncOutSimUDPClient:
    """Asynchronous UDP client delivering :class:`OutSimPacket` objects."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4123,
        *,
        timeout: float = 0.05,
        reorder_grace: float | None = None,
        jump_tolerance: int = 200,
        buffer_size: int | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        buffer_capacity = DEFAULT_REORDER_BUFFER_SIZE
        if buffer_size is not None:
            try:
                numeric = int(buffer_size)
            except (TypeError, ValueError):
                numeric = 0
            if numeric > 0:
                buffer_capacity = numeric
        self._remote_host = host
        self._remote_addresses = OutSimUDPClient._resolve_remote_addresses(host)
        self._timeout = timeout
        self._requested_port = port
        self._buffer_grace = max(float(reorder_grace) if reorder_grace is not None else timeout, 0.0)
        self._jump_tolerance = max(int(jump_tolerance), 0)
        self._buffer_capacity = buffer_capacity
        self._loop = loop
        self._transport: asyncio.DatagramTransport | None = None
        self._address: Tuple[str, int] = ("", 0)
        self._processor: _OutSimPacketProcessor | None = None
        self._ready_packets: deque[OutSimPacket] = deque()
        self._condition: asyncio.Condition | None = None
        self._notify_scheduled = False
        self._pending_error: BaseException | None = None
        self._pending_timer: asyncio.TimerHandle | None = None
        self._timeouts = 0
        self._ignored_hosts = 0
        self._closed_event = asyncio.Event()
        self._closed_event.set()
        self._closing = False

    @classmethod
    async def create(
        cls,
        host: str = "127.0.0.1",
        port: int = 4123,
        *,
        timeout: float = 0.05,
        reorder_grace: float | None = None,
        jump_tolerance: int = 200,
        buffer_size: int | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> "AsyncOutSimUDPClient":
        self = cls(
            host=host,
            port=port,
            timeout=timeout,
            reorder_grace=reorder_grace,
            jump_tolerance=jump_tolerance,
            buffer_size=buffer_size,
            loop=loop,
        )
        await self.start()
        return self

    async def start(self) -> None:
        if self._transport is not None:
            return
        loop = self._loop or asyncio.get_running_loop()
        self._loop = loop
        transport, _ = await loop.create_datagram_endpoint(
            lambda: _AsyncOutSimProtocol(self),
            local_addr=(None, self._requested_port),
            family=socket.AF_INET,
        )
        self._transport = transport  # connection_made will configure the rest

    async def __aenter__(self) -> "AsyncOutSimUDPClient":
        if self._transport is None:
            await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    @property
    def address(self) -> Tuple[str, int]:
        return self._address

    @property
    def timeouts(self) -> int:
        return self._timeouts

    @property
    def ignored_hosts(self) -> int:
        return self._ignored_hosts

    @property
    def statistics(self) -> dict[str, int]:
        if self._processor is None:
            return {
                "received": 0,
                "delivered": 0,
                "duplicates": 0,
                "reordered": 0,
                "late_recovered": 0,
                "loss_events": 0,
            }
        return self._processor.statistics

    async def recv(self, timeout: float | None = None) -> Optional[OutSimPacket]:
        if self._transport is None or self._processor is None or self._condition is None:
            raise RuntimeError("AsyncOutSimUDPClient is not started")
        loop = self._loop
        assert loop is not None
        if timeout is None:
            timeout = self._timeout
        if timeout is not None and timeout < 0:
            timeout = 0.0
        start = loop.time()
        while True:
            if self._pending_error is not None:
                error = self._pending_error
                self._pending_error = None
                raise error
            if self._ready_packets:
                return self._ready_packets.popleft()
            if self._closing:
                raise RuntimeError("AsyncOutSimUDPClient is closed")
            if timeout == 0.0:
                break
            remaining: float | None
            if timeout is None:
                remaining = None
            else:
                elapsed = loop.time() - start
                remaining = timeout - elapsed
                if remaining <= 0:
                    break
            condition = self._condition
            async with condition:
                waiter = condition.wait()
                try:
                    if remaining is None:
                        await waiter
                    else:
                        await asyncio.wait_for(waiter, remaining)
                except asyncio.TimeoutError:
                    break
                continue
        packet = self._processor.pop_ready_packet(time.monotonic(), allow_grace=True)
        if packet is not None:
            return packet
        self._timeouts += 1
        logger.warning(
            "OutSim recv retries exhausted without receiving a packet.",
            extra={
                "event": "outsim.recv_timeout",
                "retries": 0,
                "timeout": timeout,
                "remote_host": self._remote_host,
                "port": self._address[1],
            },
        )
        return None

    def drain_ready(self) -> list[OutSimPacket]:
        packets: list[OutSimPacket] = []
        while self._ready_packets:
            packets.append(self._ready_packets.popleft())
        if self._processor is not None:
            packets.extend(self._processor.drain_ready(time.monotonic()))
        return packets

    async def close(self) -> None:
        if self._closing:
            await self._closed_event.wait()
            return
        self._closing = True
        if self._pending_timer is not None:
            self._pending_timer.cancel()
            self._pending_timer = None
        transport = self._transport
        if transport is not None:
            transport.close()
        else:
            self._closed_event.set()
        await self._closed_event.wait()
        self._wake_waiters()

    def _connection_made(self, transport: asyncio.BaseTransport) -> None:
        datagram = transport  # type: ignore[assignment]
        assert isinstance(datagram, asyncio.DatagramTransport)
        self._transport = datagram
        sockname = datagram.get_extra_info("sockname")
        if isinstance(sockname, tuple) and len(sockname) >= 2:
            host = sockname[0] if isinstance(sockname[0], str) else ""
            port = int(sockname[1]) if isinstance(sockname[1], int) else 0
            self._address = (host, port)
        else:  # pragma: no cover - defensive
            self._address = ("", self._requested_port)
        self._processor = _OutSimPacketProcessor(
            remote_host=self._remote_host,
            port=self._address[1],
            buffer_capacity=self._buffer_capacity,
            buffer_grace=self._buffer_grace,
            jump_tolerance=self._jump_tolerance,
        )
        self._condition = asyncio.Condition()
        self._closed_event = asyncio.Event()

    def _on_datagram(self, payload: bytes, source: tuple[str, int]) -> None:
        if not payload:
            return
        if self._remote_addresses and source[0] not in self._remote_addresses:
            self._ignored_hosts += 1
            logger.warning(
                "Ignoring OutSim datagram from unexpected host.",
                extra={
                    "event": "outsim.ignored_host",
                    "expected_hosts": sorted(self._remote_addresses),
                    "source_host": source[0],
                    "port": self._address[1],
                },
            )
            return
        processor = self._processor
        if processor is None:
            return
        try:
            packet = OutSimPacket.from_bytes(payload)
        except Exception as exc:  # pragma: no cover - propagate decoding issues
            self._pending_error = exc
            self._wake_waiters()
            return
        processor.record_packet(packet, time.monotonic())
        made_ready = False
        while True:
            ready = processor.pop_ready_packet(time.monotonic(), allow_grace=False)
            if ready is None:
                break
            self._ready_packets.append(ready)
            made_ready = True
        if made_ready:
            self._wake_waiters()
        if processor.pending_buffered:
            self._schedule_pending_release()

    def _on_error(self, exc: Exception) -> None:
        self._pending_error = exc
        self._wake_waiters()

    def _connection_lost(self, _exc: Exception | None) -> None:
        condition = self._condition
        processor = self._processor
        self._transport = None
        self._processor = None
        self._condition = None
        if self._pending_timer is not None:
            self._pending_timer.cancel()
            self._pending_timer = None
        while self._ready_packets:
            packet = self._ready_packets.popleft()
            packet.release()
        if processor is not None:
            processor.flush()
        self._closed_event.set()
        self._wake_waiters(condition)

    def _wake_waiters(self, condition: asyncio.Condition | None = None) -> None:
        if self._notify_scheduled:
            return
        loop = self._loop
        if loop is None:
            return
        if condition is None:
            condition = self._condition
        if condition is None:
            return
        self._notify_scheduled = True
        loop.call_soon(self._schedule_notification, condition)

    def _schedule_notification(self, condition: asyncio.Condition) -> None:
        self._notify_scheduled = False
        loop = self._loop
        if loop is None:
            return
        loop.create_task(self._notify_condition(condition))

    async def _notify_condition(self, condition: asyncio.Condition) -> None:
        async with condition:
            condition.notify_all()

    def _schedule_pending_release(self) -> None:
        processor = self._processor
        loop = self._loop
        if processor is None or loop is None:
            return
        deadline = processor.pending_deadline()
        if deadline is None:
            if self._pending_timer is not None:
                self._pending_timer.cancel()
                self._pending_timer = None
            return
        delay = max(0.0, deadline - time.monotonic())
        if self._pending_timer is not None:
            self._pending_timer.cancel()
        self._pending_timer = loop.call_later(delay, self._release_pending_packets)

    def _release_pending_packets(self) -> None:
        self._pending_timer = None
        processor = self._processor
        if processor is None:
            return
        made_ready = False
        while True:
            packet = processor.pop_ready_packet(time.monotonic(), allow_grace=True)
            if packet is None:
                break
            self._ready_packets.append(packet)
            made_ready = True
        if made_ready:
            self._wake_waiters()
