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
import socket
import struct
import time
from typing import Optional, Tuple

__all__ = ["OutSimPacket", "OutSimUDPClient"]


_BASE_STRUCT = struct.Struct("<I15f")
_ID_STRUCT = struct.Struct("<I15fI")


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

    @classmethod
    def from_bytes(cls, payload: bytes) -> "OutSimPacket":
        """Deserialize a byte payload following the official layout."""

        if len(payload) < _BASE_STRUCT.size:
            raise ValueError(
                f"OutSim payload too small: {len(payload)} bytes (expected {_BASE_STRUCT.size})"
            )

        if len(payload) >= _ID_STRUCT.size:
            values = _ID_STRUCT.unpack_from(payload)
            *base_values, player_id = values
        else:
            base_values = _BASE_STRUCT.unpack_from(payload)
            player_id = None

        return cls(*base_values, player_id=player_id)


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
        self._address: Tuple[str, int] = (host, port)
        self._timeout = timeout
        self._retries = retries
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._socket.bind(self._address)
        # When port 0 is used, store the real port assigned by the OS.
        self._address = (host, self._socket.getsockname()[1])

    @property
    def address(self) -> Tuple[str, int]:
        """Return the bound socket address."""

        return self._address

    def recv(self) -> Optional[OutSimPacket]:
        """Attempt to receive a packet, returning ``None`` on timeout."""

        for _ in range(self._retries):
            try:
                payload, _ = self._socket.recvfrom(_ID_STRUCT.size)
            except BlockingIOError:
                time.sleep(self._timeout)
                continue
            if not payload:
                continue
            return OutSimPacket.from_bytes(payload)
        return None

    def close(self) -> None:
        """Close the underlying socket."""

        self._socket.close()

    def __enter__(self) -> "OutSimUDPClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
