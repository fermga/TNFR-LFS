"""OutGauge UDP client implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import socket
import struct
import time
from typing import Optional, Tuple

__all__ = ["OutGaugePacket", "OutGaugeUDPClient"]


_PACK_STRUCT = struct.Struct("<I4s16s8s6s6sHBBfffffffIIfff16s16sI")
_FLOAT_STRUCT = struct.Struct("<f")


def _decode_string(value: bytes) -> str:
    return value.split(b"\x00", 1)[0].decode("latin-1")


@dataclass(frozen=True)
class OutGaugePacket:
    """Representation of a decoded OutGauge datagram."""

    time: int
    car: str
    player_name: str
    plate: str
    track: str
    layout: str
    flags: int
    gear: int
    plid: int
    speed: float
    rpm: float
    turbo: float
    eng_temp: float
    fuel: float
    oil_pressure: float
    oil_temp: float
    dash_lights: int
    show_lights: int
    throttle: float
    brake: float
    clutch: float
    display1: str
    display2: str
    packet_id: int
    tyre_temps: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    tyre_pressures: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    tyre_temps_inner: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    tyre_temps_middle: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    tyre_temps_outer: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "OutGaugePacket":
        if len(payload) < _PACK_STRUCT.size:
            raise ValueError(
                f"OutGauge payload too small: {len(payload)} bytes (expected {_PACK_STRUCT.size})"
            )
        unpacked = _PACK_STRUCT.unpack_from(payload)
        (
            time_value,
            car,
            player_name,
            plate,
            track,
            layout,
            flags,
            gear,
            plid,
            speed,
            rpm,
            turbo,
            eng_temp,
            fuel,
            oil_pressure,
            oil_temp,
            dash_lights,
            show_lights,
            throttle,
            brake,
            clutch,
            display1,
            display2,
            packet_id,
        ) = unpacked

        extra_inner = (0.0, 0.0, 0.0, 0.0)
        extra_middle = (0.0, 0.0, 0.0, 0.0)
        extra_outer = (0.0, 0.0, 0.0, 0.0)
        extra_offset = _PACK_STRUCT.size
        if len(payload) > extra_offset:
            remainder = memoryview(payload)[extra_offset:]
            float_count = len(remainder) // _FLOAT_STRUCT.size
            if float_count > 0:
                try:
                    floats = struct.unpack(
                        "<" + "f" * float_count, remainder[: float_count * _FLOAT_STRUCT.size]
                    )
                except struct.error:
                    floats = ()
                if floats:
                    inner = list(extra_inner)
                    middle = list(extra_middle)
                    outer = list(extra_outer)
                    for index in range(4):
                        try:
                            value = float(floats[index])
                            if math.isfinite(value):
                                inner[index] = value
                        except (IndexError, TypeError, ValueError):
                            break
                    for index in range(4):
                        try:
                            value = float(floats[4 + index])
                            if math.isfinite(value):
                                middle[index] = value
                        except (IndexError, TypeError, ValueError):
                            break
                    for index in range(4):
                        try:
                            value = float(floats[8 + index])
                            if math.isfinite(value):
                                outer[index] = value
                        except (IndexError, TypeError, ValueError):
                            break
                    extra_inner = tuple(inner)  # type: ignore[assignment]
                    extra_middle = tuple(middle)  # type: ignore[assignment]
                    extra_outer = tuple(outer)  # type: ignore[assignment]

        def _average_layers(index: int) -> float:
            values = [extra_inner[index], extra_middle[index], extra_outer[index]]
            finite = [value for value in values if math.isfinite(value) and value > 0.0]
            if not finite:
                return 0.0
            return float(sum(finite) / len(finite))

        averaged = (
            _average_layers(0),
            _average_layers(1),
            _average_layers(2),
            _average_layers(3),
        )
        return cls(
            time=time_value,
            car=_decode_string(car),
            player_name=_decode_string(player_name),
            plate=_decode_string(plate),
            track=_decode_string(track),
            layout=_decode_string(layout),
            flags=flags,
            gear=gear,
            plid=plid,
            speed=speed,
            rpm=rpm,
            turbo=turbo,
            eng_temp=eng_temp,
            fuel=fuel,
            oil_pressure=oil_pressure,
            oil_temp=oil_temp,
            dash_lights=dash_lights,
            show_lights=show_lights,
            throttle=throttle,
            brake=brake,
            clutch=clutch,
            display1=_decode_string(display1),
            display2=_decode_string(display2),
            packet_id=packet_id,
            tyre_temps=averaged,
            tyre_pressures=(0.0, 0.0, 0.0, 0.0),
            tyre_temps_inner=extra_inner,
            tyre_temps_middle=extra_middle,
            tyre_temps_outer=extra_outer,
        )


class OutGaugeUDPClient:
    """Non-blocking UDP client for OutGauge telemetry."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 3000,
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
        self._address = (host, self._socket.getsockname()[1])

    @property
    def address(self) -> Tuple[str, int]:
        return self._address

    def recv(self) -> Optional[OutGaugePacket]:
        for _ in range(self._retries):
            try:
                payload, _ = self._socket.recvfrom(_PACK_STRUCT.size)
            except BlockingIOError:
                time.sleep(self._timeout)
                continue
            if not payload:
                continue
            return OutGaugePacket.from_bytes(payload)
        return None

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "OutGaugeUDPClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
