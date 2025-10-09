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

from collections import deque
from dataclasses import dataclass
import logging
from types import TracebackType
import socket
import struct
import time
from typing import Deque, Optional, Tuple

__all__ = [
    "OUTSIM_MAX_PACKET_SIZE",
    "OutSimDriverInputs",
    "OutSimPacket",
    "OutSimUDPClient",
    "OutSimWheelState",
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
class OutSimDriverInputs:
    """Driver control inputs contained in extended OutSim packets."""

    throttle: float = 0.0
    brake: float = 0.0
    clutch: float = 0.0
    handbrake: float = 0.0
    steer: float = 0.0


@dataclass(frozen=True)
class OutSimWheelState:
    """Per-wheel telemetry sampled from the OutSim stream."""

    slip_ratio: float = 0.0
    slip_angle: float = 0.0
    longitudinal_force: float = 0.0
    lateral_force: float = 0.0
    load: float = 0.0
    suspension_deflection: float = 0.0
    decoded: bool = False


@dataclass(frozen=True)
class OutSimPacket:
    """Representation of a decoded OutSim datagram."""

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
    inputs: Optional[OutSimDriverInputs] = None
    wheels: Tuple[OutSimWheelState, OutSimWheelState, OutSimWheelState, OutSimWheelState] = (
        OutSimWheelState(),
        OutSimWheelState(),
        OutSimWheelState(),
        OutSimWheelState(),
    )

    @classmethod
    def from_bytes(cls, payload: bytes) -> "OutSimPacket":
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

        inputs: Optional[OutSimDriverInputs] = None
        if len(payload) >= offset + _INPUT_STRUCT.size:
            throttle, brake, clutch, handbrake, steer = _INPUT_STRUCT.unpack_from(
                payload, offset
            )
            inputs = OutSimDriverInputs(
                throttle=throttle,
                brake=brake,
                clutch=clutch,
                handbrake=handbrake,
                steer=steer,
            )
            offset += _INPUT_STRUCT.size

        wheels: list[OutSimWheelState] = []
        for _ in range(4):
            if len(payload) < offset + _WHEEL_STRUCT.size:
                break
            slip_ratio, slip_angle, long_force, lat_force, load, deflection = (
                _WHEEL_STRUCT.unpack_from(payload, offset)
            )
            wheels.append(
                OutSimWheelState(
                    slip_ratio=slip_ratio,
                    slip_angle=slip_angle,
                    longitudinal_force=long_force,
                    lateral_force=lat_force,
                    load=load,
                    suspension_deflection=deflection,
                    decoded=True,
                )
            )
            offset += _WHEEL_STRUCT.size

        if len(wheels) > 4:
            wheels = wheels[:4]
        if len(wheels) < 4:
            wheels.extend([OutSimWheelState()] * (4 - len(wheels)))
        # Ensure ``wheels`` always contains exactly four elements for the tuple below.
        w_fl, w_fr, w_rl, w_rr = wheels

        return cls(
            *base_values,
            player_id=player_id,
            inputs=inputs,
            wheels=(w_fl, w_fr, w_rl, w_rr),
        )


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
        """

        self._remote_host = host
        self._remote_addresses = self._resolve_remote_addresses(host)
        self._timeout = timeout
        self._retries = retries
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._socket.bind(("", port))
        local_host, local_port = self._socket.getsockname()
        self._address: Tuple[str, int] = (local_host, local_port)
        self._timeouts = 0
        self._ignored_hosts = 0
        self._buffer: Deque[tuple[float, OutSimPacket]] = deque()
        self._buffer_grace = max(float(reorder_grace) if reorder_grace is not None else timeout, 0.0)
        self._received_packets = 0
        self._delivered_packets = 0
        self._duplicate_packets = 0
        self._out_of_order_packets = 0
        self._late_reordered_packets = 0
        self._loss_events = 0
        self._last_emitted_time: Optional[int] = None
        self._last_seen_time: Optional[int] = None
        self._jump_tolerance = max(int(jump_tolerance), 0)
        self._reordered_times: set[int] = set()

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

        return {
            "received": self._received_packets,
            "delivered": self._delivered_packets,
            "duplicates": self._duplicate_packets,
            "reordered": self._out_of_order_packets,
            "late_recovered": self._late_reordered_packets,
            "loss_events": self._loss_events,
        }

    def recv(self) -> Optional[OutSimPacket]:
        """Attempt to receive a packet, returning ``None`` on timeout."""

        ready = self._pop_ready_packet(time.monotonic(), allow_grace=False)
        if ready is not None:
            return ready

        for _ in range(self._retries):
            try:
                payload, source = self._socket.recvfrom(OUTSIM_MAX_PACKET_SIZE)
            except BlockingIOError:
                if self._timeout > 0.0:
                    time.sleep(self._timeout)
                ready = self._pop_ready_packet(time.monotonic(), allow_grace=False)
                if ready is not None:
                    return ready
                continue
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
            self._record_packet(packet)
            ready = self._pop_ready_packet(time.monotonic(), allow_grace=False)
            if ready is not None:
                return ready
        ready = self._pop_ready_packet(time.monotonic(), allow_grace=True)
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

        ready: list[OutSimPacket] = []
        while True:
            packet = self._pop_ready_packet(time.monotonic(), allow_grace=False)
            if packet is None:
                break
            ready.append(packet)
        return ready

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

    def _pop_ready_packet(
        self, now: float, *, allow_grace: bool
    ) -> Optional[OutSimPacket]:
        while self._buffer:
            arrival, packet = self._buffer[0]
            if not allow_grace and len(self._buffer) == 1 and (now - arrival) < self._buffer_grace:
                break
            self._buffer.popleft()
            if (
                self._last_emitted_time is not None
                and packet.time <= self._last_emitted_time
            ):
                # Drop stale packets that somehow slipped through.
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
                            "port": self._address[1],
                            "remote_host": self._remote_host,
                        },
                    )
            if packet.time in self._reordered_times:
                self._reordered_times.discard(packet.time)
                self._late_reordered_packets += 1
            self._delivered_packets += 1
            self._last_emitted_time = packet.time
            return packet
        return None

    def _record_packet(self, packet: OutSimPacket) -> None:
        self._received_packets += 1
        arrival = time.monotonic()

        if self._last_seen_time is not None and packet.time < self._last_seen_time:
            self._out_of_order_packets += 1
            self._reordered_times.add(packet.time)
            logger.warning(
                "OutSim out-of-order packet received.",
                extra={
                    "event": "outsim.out_of_order",
                    "last_time": self._last_seen_time,
                    "current_time": packet.time,
                    "port": self._address[1],
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
                    "port": self._address[1],
                    "remote_host": self._remote_host,
                },
            )
            return

        inserted = False
        for index, (_, existing) in enumerate(self._buffer):
            if packet.time < existing.time:
                self._buffer.insert(index, (arrival, packet))
                inserted = True
                if packet.time not in self._reordered_times:
                    self._reordered_times.add(packet.time)
                break
        if not inserted:
            self._buffer.append((arrival, packet))

    def _is_duplicate(self, timestamp: int) -> bool:
        if self._last_emitted_time is not None and timestamp == self._last_emitted_time:
            return True
        return any(existing.time == timestamp for _, existing in self._buffer)
