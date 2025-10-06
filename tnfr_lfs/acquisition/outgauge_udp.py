"""OutGauge UDP client implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import socket
import struct
import time
from typing import Optional, Tuple, cast

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
    brake_temps: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

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

        extra_inner: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        extra_middle: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        extra_outer: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        extra_pressures: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        extra_brakes: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
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
                    values = list(floats)
                    # OutGauge appends the extended tyre payload as five
                    # consecutive blocks of four floats (inner, middle, outer,
                    # pressure and brake temperatures) when the corresponding
                    # OG_EXT_* flags are enabled.

                    def _extract_block(offset: int) -> tuple[float, float, float, float]:
                        block = [0.0, 0.0, 0.0, 0.0]
                        for index in range(4):
                            position = offset + index
                            if position >= len(values):
                                break
                            try:
                                numeric = float(values[position])
                            except (TypeError, ValueError):
                                continue
                            if not math.isfinite(numeric) or numeric <= 0.0:
                                continue
                            block[index] = numeric
                        block_tuple = tuple(block)
                        if len(block_tuple) < 4:
                            block_tuple = block_tuple + (0.0,) * (4 - len(block_tuple))
                        return cast(tuple[float, float, float, float], block_tuple)

                    extra_inner = _extract_block(0)
                    extra_middle = _extract_block(4)
                    extra_outer = _extract_block(8)
                    extra_pressures = _extract_block(12)
                    extra_brakes = _extract_block(16)

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
            tyre_pressures=extra_pressures,
            tyre_temps_inner=extra_inner,
            tyre_temps_middle=extra_middle,
            tyre_temps_outer=extra_outer,
            brake_temps=extra_brakes,
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
