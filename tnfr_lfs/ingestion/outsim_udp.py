"""OutSim UDP client implementation.

The Live for Speed simulator broadcasts physics-oriented telemetry using
the OutSim protocol.  Packets are transmitted as a binary structure over
UDP and contain orientation, acceleration and position data.  This
module provides a small client capable of decoding the official packet
layout while operating in non-blocking mode so that it can be safely
polled from high-frequency loops.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from types import TracebackType
import socket
import struct
import time
from typing import Optional, Tuple

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

    def recv(self) -> Optional[OutSimPacket]:
        """Attempt to receive a packet, returning ``None`` on timeout."""

        for _ in range(self._retries):
            try:
                payload, source = self._socket.recvfrom(OUTSIM_MAX_PACKET_SIZE)
            except BlockingIOError:
                time.sleep(self._timeout)
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
            return OutSimPacket.from_bytes(payload)
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
