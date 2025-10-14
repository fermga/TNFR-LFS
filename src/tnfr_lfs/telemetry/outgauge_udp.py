"""OutGauge UDP client implementation.

The client understands Live for Speed's OutGauge datagrams and preserves a
small buffer keyed by :attr:`OutGaugePacket.packet_id` so that late
packets can be reinserted in-order.  Duplicate datagrams are discarded and
suspected gaps are tracked via :attr:`OutGaugeUDPClient.statistics`,
allowing monitoring code to differentiate between packet loss and
successful recoveries.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import logging
from types import TracebackType
import math
import socket
import struct
import time
from typing import Optional, Tuple, cast

from ._socket_poll import wait_for_read_ready
from ._reorder_buffer import CircularReorderBuffer, DEFAULT_REORDER_BUFFER_SIZE
from .pools import PacketPool, PoolItem
from ._tyre_compound import extract_compound_hint, normalise_compound_label

__all__ = [
    "AsyncOutGaugeUDPClient",
    "OutGaugePacket",
    "OutGaugeUDPClient",
    "FrozenOutGaugePacket",
]


logger = logging.getLogger(__name__)


_SINGLE_PACKET_RELEASE_LIMIT = 0.01


def _resolve_buffer_grace(timeout: float, reorder_grace: float | None) -> float:
    """Return the grace period capped to ensure single packets emit quickly."""

    if reorder_grace is None:
        candidate = float(timeout)
    else:
        candidate = float(reorder_grace)
    if not math.isfinite(candidate):
        return _SINGLE_PACKET_RELEASE_LIMIT
    if candidate <= 0.0:
        return 0.0
    return min(candidate, _SINGLE_PACKET_RELEASE_LIMIT)


_PACK_STRUCT = struct.Struct("<I4s16s8s6s6sHBBfffffffIIfff16s16sI")
_FLOAT_STRUCT = struct.Struct("<f")
_EXTENDED_BLOCKS = 5
_EXTENDED_BLOCK_WIDTH = 4
_MAX_DATAGRAM_SIZE = _PACK_STRUCT.size + _EXTENDED_BLOCKS * _EXTENDED_BLOCK_WIDTH * _FLOAT_STRUCT.size


def _normalise_identifier(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = "".join(ch for ch in value.lower() if ch.isalnum())
    return cleaned or None


def _decode_string(value: bytes) -> str:
    return value.split(b"\x00", 1)[0].decode("latin-1")


@dataclass(frozen=True)
class FrozenOutGaugePacket:
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
    tyre_compound: str | None = None


class OutGaugePacket(PoolItem):
    """Representation of a decoded OutGauge datagram."""

    __slots__ = (
        "time",
        "car",
        "player_name",
        "plate",
        "track",
        "layout",
        "flags",
        "gear",
        "plid",
        "speed",
        "rpm",
        "turbo",
        "eng_temp",
        "fuel",
        "oil_pressure",
        "oil_temp",
        "dash_lights",
        "show_lights",
        "throttle",
        "brake",
        "clutch",
        "display1",
        "display2",
        "packet_id",
        "tyre_temps",
        "tyre_pressures",
        "tyre_temps_inner",
        "tyre_temps_middle",
        "tyre_temps_outer",
        "brake_temps",
        "tyre_compound",
    )

    def __init__(
        self,
        *,
        time: int = 0,
        car: str = "",
        player_name: str = "",
        plate: str = "",
        track: str = "",
        layout: str = "",
        flags: int = 0,
        gear: int = 0,
        plid: int = 0,
        speed: float = 0.0,
        rpm: float = 0.0,
        turbo: float = 0.0,
        eng_temp: float = 0.0,
        fuel: float = 0.0,
        oil_pressure: float = 0.0,
        oil_temp: float = 0.0,
        dash_lights: int = 0,
        show_lights: int = 0,
        throttle: float = 0.0,
        brake: float = 0.0,
        clutch: float = 0.0,
        display1: str = "",
        display2: str = "",
        packet_id: int = 0,
        tyre_temps: tuple[float, float, float, float] | None = None,
        tyre_pressures: tuple[float, float, float, float] | None = None,
        tyre_temps_inner: tuple[float, float, float, float] | None = None,
        tyre_temps_middle: tuple[float, float, float, float] | None = None,
        tyre_temps_outer: tuple[float, float, float, float] | None = None,
        brake_temps: tuple[float, float, float, float] | None = None,
        tyre_compound: str | None = None,
    ) -> None:
        super().__init__()
        self._reset_values()
        self.time = int(time)
        self.car = car
        self.player_name = player_name
        self.plate = plate
        self.track = track
        self.layout = layout
        self.flags = int(flags)
        self.gear = int(gear)
        self.plid = int(plid)
        self.speed = float(speed)
        self.rpm = float(rpm)
        self.turbo = float(turbo)
        self.eng_temp = float(eng_temp)
        self.fuel = float(fuel)
        self.oil_pressure = float(oil_pressure)
        self.oil_temp = float(oil_temp)
        self.dash_lights = int(dash_lights)
        self.show_lights = int(show_lights)
        self.throttle = float(throttle)
        self.brake = float(brake)
        self.clutch = float(clutch)
        self.display1 = display1
        self.display2 = display2
        self.packet_id = int(packet_id)
        if tyre_temps is not None:
            self.tyre_temps = tuple(tyre_temps)
        if tyre_pressures is not None:
            self.tyre_pressures = tuple(tyre_pressures)
        if tyre_temps_inner is not None:
            self.tyre_temps_inner = tuple(tyre_temps_inner)
        if tyre_temps_middle is not None:
            self.tyre_temps_middle = tuple(tyre_temps_middle)
        if tyre_temps_outer is not None:
            self.tyre_temps_outer = tuple(tyre_temps_outer)
        if brake_temps is not None:
            self.brake_temps = tuple(brake_temps)
        self.tyre_compound = normalise_compound_label(tyre_compound)

    def _reset_values(self) -> None:
        self.time = 0
        self.car = ""
        self.player_name = ""
        self.plate = ""
        self.track = ""
        self.layout = ""
        self.flags = 0
        self.gear = 0
        self.plid = 0
        self.speed = 0.0
        self.rpm = 0.0
        self.turbo = 0.0
        self.eng_temp = 0.0
        self.fuel = 0.0
        self.oil_pressure = 0.0
        self.oil_temp = 0.0
        self.dash_lights = 0
        self.show_lights = 0
        self.throttle = 0.0
        self.brake = 0.0
        self.clutch = 0.0
        self.display1 = ""
        self.display2 = ""
        self.packet_id = 0
        self.tyre_temps = (0.0, 0.0, 0.0, 0.0)
        self.tyre_pressures = (0.0, 0.0, 0.0, 0.0)
        self.tyre_temps_inner = (0.0, 0.0, 0.0, 0.0)
        self.tyre_temps_middle = (0.0, 0.0, 0.0, 0.0)
        self.tyre_temps_outer = (0.0, 0.0, 0.0, 0.0)
        self.brake_temps = (0.0, 0.0, 0.0, 0.0)
        self.tyre_compound = None

    def _reset(self) -> None:
        self._reset_values()

    def _populate(
        self,
        values: tuple,
        *,
        extra_inner: tuple[float, float, float, float],
        extra_middle: tuple[float, float, float, float],
        extra_outer: tuple[float, float, float, float],
        extra_pressures: tuple[float, float, float, float],
        extra_brakes: tuple[float, float, float, float],
        averaged: tuple[float, float, float, float],
    ) -> None:
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
        ) = values

        self.time = int(time_value)
        self.car = _decode_string(car)
        self.player_name = _decode_string(player_name)
        self.plate = _decode_string(plate)
        self.track = _decode_string(track)
        self.layout = _decode_string(layout)
        self.flags = int(flags)
        self.gear = int(gear)
        self.plid = int(plid)
        self.speed = float(speed)
        self.rpm = float(rpm)
        self.turbo = float(turbo)
        self.eng_temp = float(eng_temp)
        self.fuel = float(fuel)
        self.oil_pressure = float(oil_pressure)
        self.oil_temp = float(oil_temp)
        self.dash_lights = int(dash_lights)
        self.show_lights = int(show_lights)
        self.throttle = float(throttle)
        self.brake = float(brake)
        self.clutch = float(clutch)
        self.display1 = _decode_string(display1)
        self.display2 = _decode_string(display2)
        self.packet_id = int(packet_id)
        self.tyre_temps = averaged
        self.tyre_pressures = extra_pressures
        self.tyre_temps_inner = extra_inner
        self.tyre_temps_middle = extra_middle
        self.tyre_temps_outer = extra_outer
        self.brake_temps = extra_brakes
        compound_hint = extract_compound_hint(self.display1, self.display2)
        if compound_hint is not None:
            self.tyre_compound = compound_hint

    def freeze(self) -> FrozenOutGaugePacket:
        return FrozenOutGaugePacket(
            time=self.time,
            car=self.car,
            player_name=self.player_name,
            plate=self.plate,
            track=self.track,
            layout=self.layout,
            flags=self.flags,
            gear=self.gear,
            plid=self.plid,
            speed=self.speed,
            rpm=self.rpm,
            turbo=self.turbo,
            eng_temp=self.eng_temp,
            fuel=self.fuel,
            oil_pressure=self.oil_pressure,
            oil_temp=self.oil_temp,
            dash_lights=self.dash_lights,
            show_lights=self.show_lights,
            throttle=self.throttle,
            brake=self.brake,
            clutch=self.clutch,
            display1=self.display1,
            display2=self.display2,
            packet_id=self.packet_id,
            tyre_temps=self.tyre_temps,
            tyre_pressures=self.tyre_pressures,
            tyre_temps_inner=self.tyre_temps_inner,
            tyre_temps_middle=self.tyre_temps_middle,
            tyre_temps_outer=self.tyre_temps_outer,
            brake_temps=self.brake_temps,
            tyre_compound=self.tyre_compound,
        )

    @classmethod
    def from_bytes(
        cls, payload: bytes, *, freeze: bool = False
    ) -> "OutGaugePacket | FrozenOutGaugePacket":
        if len(payload) < _PACK_STRUCT.size:
            raise ValueError(
                f"OutGauge payload too small: {len(payload)} bytes (expected {_PACK_STRUCT.size})"
            )
        values = _PACK_STRUCT.unpack_from(payload)

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
                    values_list = list(floats)

                    def _extract_block(offset: int) -> tuple[float, float, float, float]:
                        block = [0.0, 0.0, 0.0, 0.0]
                        for index in range(4):
                            position = offset + index
                            if position >= len(values_list):
                                break
                            try:
                                numeric = float(values_list[position])
                            except (TypeError, ValueError):
                                continue
                            if not math.isfinite(numeric) or numeric <= 0.0:
                                continue
                            block[index] = numeric
                        return cast(
                            tuple[float, float, float, float],
                            tuple(block),
                        )

                    extra_inner = _extract_block(0)
                    extra_middle = _extract_block(4)
                    extra_outer = _extract_block(8)
                    extra_pressures = _extract_block(12)
                    extra_brakes = _extract_block(16)

        def _average_layers(index: int) -> float:
            values_list = [extra_inner[index], extra_middle[index], extra_outer[index]]
            finite = [value for value in values_list if math.isfinite(value) and value > 0.0]
            if not finite:
                return 0.0
            return float(sum(finite) / len(finite))

        averaged = (
            _average_layers(0),
            _average_layers(1),
            _average_layers(2),
            _average_layers(3),
        )

        display1 = _decode_string(values[21])
        display2 = _decode_string(values[22])
        compound_hint = extract_compound_hint(display1, display2)

        if freeze:
            return FrozenOutGaugePacket(
                time=int(values[0]),
                car=_decode_string(values[1]),
                player_name=_decode_string(values[2]),
                plate=_decode_string(values[3]),
                track=_decode_string(values[4]),
                layout=_decode_string(values[5]),
                flags=int(values[6]),
                gear=int(values[7]),
                plid=int(values[8]),
                speed=float(values[9]),
                rpm=float(values[10]),
                turbo=float(values[11]),
                eng_temp=float(values[12]),
                fuel=float(values[13]),
                oil_pressure=float(values[14]),
                oil_temp=float(values[15]),
                dash_lights=int(values[16]),
                show_lights=int(values[17]),
                throttle=float(values[18]),
                brake=float(values[19]),
                clutch=float(values[20]),
                display1=display1,
                display2=display2,
                packet_id=int(values[23]),
                tyre_temps=averaged,
                tyre_pressures=extra_pressures,
                tyre_temps_inner=extra_inner,
                tyre_temps_middle=extra_middle,
                tyre_temps_outer=extra_outer,
                brake_temps=extra_brakes,
                tyre_compound=compound_hint,
            )

        packet = _OUTGAUGE_POOL.acquire()
        packet._populate(
            values,
            extra_inner=extra_inner,
            extra_middle=extra_middle,
            extra_outer=extra_outer,
            extra_pressures=extra_pressures,
            extra_brakes=extra_brakes,
            averaged=averaged,
        )
        return packet


_OUTGAUGE_POOL: PacketPool[OutGaugePacket] = PacketPool(OutGaugePacket)


class _OutGaugePacketProcessor:
    """Shared buffering and accounting logic for OutGauge UDP clients."""

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
        self._buffer = CircularReorderBuffer[OutGaugePacket](buffer_capacity)
        self._buffer_grace = buffer_grace
        self._jump_tolerance = jump_tolerance
        self._received_packets = 0
        self._delivered_packets = 0
        self._duplicate_packets = 0
        self._out_of_order_packets = 0
        self._loss_events = 0
        self._recovered_packets = 0
        self._last_seen_id: Optional[int] = None
        self._last_emitted_id: Optional[int] = None
        self._last_emitted_time: Optional[int] = None
        self._missing_ids: set[int] = set()
        self._pending_buffered = 0
        self._pending_deadline: float | None = None

    @property
    def statistics(self) -> dict[str, int]:
        return {
            "received": self._received_packets,
            "delivered": self._delivered_packets,
            "duplicates": self._duplicate_packets,
            "reordered": self._out_of_order_packets,
            "loss_events": self._loss_events,
            "recovered": self._recovered_packets,
        }

    def record_packet(self, packet: OutGaugePacket, arrival: float) -> bool:
        self._received_packets += 1

        if self._last_seen_id is not None:
            delta = packet.packet_id - self._last_seen_id
            if delta > 1:
                missing = list(range(self._last_seen_id + 1, packet.packet_id))
                if missing:
                    self._loss_events += len(missing)
                    self._missing_ids.update(missing)
                    logger.warning(
                        "OutGauge packet gap detected (possible loss).",
                        extra={
                            "event": "outgauge.packet_gap",
                            "missing": missing,
                            "last_packet_id": self._last_seen_id,
                            "current_packet_id": packet.packet_id,
                            "port": self._port,
                            "remote_host": self._remote_host,
                        },
                    )
            elif delta < 0:
                self._out_of_order_packets += 1
                logger.warning(
                    "OutGauge out-of-order packet received.",
                    extra={
                        "event": "outgauge.out_of_order",
                        "last_packet_id": self._last_seen_id,
                        "current_packet_id": packet.packet_id,
                        "port": self._port,
                        "remote_host": self._remote_host,
                    },
                )
        self._last_seen_id = (
            packet.packet_id
            if self._last_seen_id is None
            else max(self._last_seen_id, packet.packet_id)
        )

        if self._is_duplicate(packet.packet_id):
            self._duplicate_packets += 1
            logger.warning(
                "OutGauge duplicate packet dropped.",
                extra={
                    "event": "outgauge.duplicate",
                    "packet_id": packet.packet_id,
                    "port": self._port,
                    "remote_host": self._remote_host,
                },
            )
            packet.release()
            return False

        _, evicted = self._buffer.insert(arrival, packet, packet.packet_id)
        if evicted is not None:
            self._loss_events += 1
            logger.warning(
                "OutGauge reorder buffer overflow; evicting oldest packet.",
                extra={
                    "event": "outgauge.buffer_overflow",
                    "evicted_packet_id": evicted.key,
                    "evicted_arrival": evicted.arrival,
                    "port": self._port,
                    "remote_host": self._remote_host,
                },
            )
            if hasattr(evicted.packet, "release"):
                evicted.packet.release()
        if self._pending_buffered:
            self._pending_buffered = min(self._pending_buffered, len(self._buffer))
            if self._pending_buffered == 0:
                self._pending_deadline = None
        return True

    def pop_ready_packet(self, now: float, *, allow_grace: bool) -> Optional[OutGaugePacket]:
        while self._buffer:
            peeked = self._buffer.peek_oldest()
            if peeked is None:
                break
            arrival, packet = peeked
            if (
                not allow_grace
                and len(self._buffer) == 1
                and self._last_emitted_id is not None
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
                self._last_emitted_id is not None
                and packet.packet_id <= self._last_emitted_id
            ):
                packet.release()
                continue
            if self._jump_tolerance and self._last_emitted_time is not None:
                delta = packet.time - self._last_emitted_time
                if delta > self._jump_tolerance:
                    self._loss_events += 1
                    logger.warning(
                        "OutGauge time jump detected (possible packet loss).",
                        extra={
                            "event": "outgauge.time_jump",
                            "delta": delta,
                            "last_time": self._last_emitted_time,
                            "current_time": packet.time,
                            "port": self._port,
                            "remote_host": self._remote_host,
                        },
                    )
            if packet.packet_id in self._missing_ids:
                self._missing_ids.discard(packet.packet_id)
                self._recovered_packets += 1
                logger.info(
                    "OutGauge packet recovered after gap.",
                    extra={
                        "event": "outgauge.recovered",
                        "packet_id": packet.packet_id,
                        "port": self._port,
                        "remote_host": self._remote_host,
                    },
                )
            self._delivered_packets += 1
            self._last_emitted_id = packet.packet_id
            self._last_emitted_time = packet.time
            return packet
        if not self._buffer:
            self._pending_buffered = 0
            self._pending_deadline = None
        return None

    def drain_ready(self, now: float) -> list[OutGaugePacket]:
        ready: list[OutGaugePacket] = []
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

    def _is_duplicate(self, packet_id: int) -> bool:
        if self._last_emitted_id is not None and packet_id == self._last_emitted_id:
            return True
        return self._buffer.contains_key(packet_id)

    def flush(self) -> None:
        while True:
            popped = self._buffer.pop_oldest()
            if popped is None:
                break
            _, packet = popped
            if hasattr(packet, "release"):
                packet.release()
        self._missing_ids.clear()
        self._pending_buffered = 0
        self._pending_deadline = None


class OutGaugeUDPClient:
    """Non-blocking UDP client for OutGauge telemetry."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 3000,
        *,
        timeout: float = 0.05,
        retries: int = 5,
        reorder_grace: float | None = None,
        jump_tolerance: int = 200,
        buffer_size: int | None = None,
        max_batch: int | None = None,
    ) -> None:
        """Create a UDP client bound locally while expecting a remote host.

        Parameters
        ----------
        host:
            Remote host expected to provide the OutGauge datagrams.  The client
            still binds locally while filtering unexpected sources when
            ``host`` is provided.
        port:
            UDP port for both the local binding and remote filtering.
        timeout:
            Sleep interval between retry attempts when no packet is available.
        retries:
            Maximum number of non-blocking reads issued per :meth:`recv`
            invocation before considering the call a timeout.
        reorder_grace:
            Optional number of seconds to retain a lone packet inside the
            reordering buffer to allow late datagrams to be inserted ahead of
            it.  ``None`` defaults to ``min(timeout, 0.01)`` so single packets
            never wait more than 10 ms before emission.
        jump_tolerance:
            Maximum tolerated gap (in milliseconds) between successive packet
            timestamps before the client flags a suspected loss event.
        buffer_size:
            Optional maximum number of packets retained in the internal
            reordering buffer. ``None`` leaves the buffer unbounded.
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
        buffer_grace = _resolve_buffer_grace(timeout, reorder_grace)
        self._processor = _OutGaugePacketProcessor(
            remote_host=self._remote_host,
            port=self._address[1],
            buffer_capacity=buffer_capacity,
            buffer_grace=buffer_grace,
            jump_tolerance=max(int(jump_tolerance), 0),
        )
        self._timeouts = 0
        self._ignored_hosts = 0
        self._tyre_compounds: dict[str, str] = {}

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
        """Return packet accounting metrics collected by the client."""

        return self._processor.statistics

    def update_tyre_compound(self, car: str, compound: str | None) -> None:
        """Register ``compound`` for ``car`` so packets inherit the metadata."""

        key = _normalise_identifier(car)
        if key is None:
            return
        label = normalise_compound_label(compound)
        if label is None:
            self._tyre_compounds.pop(key, None)
        else:
            self._tyre_compounds[key] = label

    def _apply_tyre_compound(self, packet: OutGaugePacket) -> OutGaugePacket:
        if packet.tyre_compound:
            return packet
        key = _normalise_identifier(packet.car)
        if key is None:
            return packet
        compound = self._tyre_compounds.get(key)
        if compound is not None:
            packet.tyre_compound = compound
        return packet

    def recv(self) -> Optional[OutGaugePacket]:
        now = time.monotonic()
        ready = self._processor.pop_ready_packet(now, allow_grace=False)
        if ready is not None:
            return self._apply_tyre_compound(ready)

        deadline = None
        if self._timeout > 0.0:
            deadline = now + self._timeout

        for _ in range(self._retries):
            if self._drain_datagrams():
                ready = self._processor.pop_ready_packet(time.monotonic(), allow_grace=False)
                if ready is not None:
                    return self._apply_tyre_compound(ready)
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
            while deadline is not None:
                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    break
                wait_slice = remaining if self._timeout <= 0 else min(self._timeout, remaining)
                if wait_slice <= 0:
                    break
                if not wait_for_read_ready(
                    self._socket,
                    timeout=wait_slice,
                    deadline=deadline,
                ):
                    break
                if self._drain_datagrams():
                    ready = self._processor.pop_ready_packet(
                        time.monotonic(), allow_grace=False
                    )
                    if ready is not None:
                        return self._apply_tyre_compound(ready)
                    if not self._processor.pending_buffered:
                        break
                deadline = self._processor.pending_deadline()
        ready = self._processor.pop_ready_packet(time.monotonic(), allow_grace=True)
        if ready is not None:
            return self._apply_tyre_compound(ready)
        self._timeouts += 1
        logger.warning(
            "OutGauge recv retries exhausted without receiving a packet.",
            extra={
                "event": "outgauge.recv_timeout",
                "retries": self._retries,
                "timeout": self._timeout,
                "remote_host": self._remote_host,
                "port": self._address[1],
            },
        )
        return None

    def close(self) -> None:
        if self._processor is not None:
            self._processor.flush()
        self._socket.close()

    def __enter__(self) -> "OutGaugeUDPClient":
        return self

    def _drain_datagrams(self) -> bool:
        processed = False
        limit = self._max_batch
        drained = 0
        while limit is None or drained < limit:
            try:
                payload, source = self._socket.recvfrom(_MAX_DATAGRAM_SIZE)
            except BlockingIOError:
                break
            drained += 1
            processed = True
            if not payload:
                continue
            if self._remote_addresses and source[0] not in self._remote_addresses:
                self._ignored_hosts += 1
                logger.warning(
                    "Ignoring OutGauge datagram from unexpected host.",
                    extra={
                        "event": "outgauge.ignored_host",
                        "expected_hosts": sorted(self._remote_addresses),
                        "source_host": source[0],
                        "port": self._address[1],
                    },
                )
                continue
            packet = OutGaugePacket.from_bytes(payload)
            self._processor.record_packet(packet, time.monotonic())
        return processed

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @staticmethod
    def _resolve_remote_addresses(host: str) -> set[str]:
        if not host or host in {"", "0.0.0.0"}:
            return set()
        try:
            info = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_DGRAM)
        except socket.gaierror:
            logger.warning(
                "Failed to resolve OutGauge remote host; disabling host filtering.",
                extra={
                    "event": "outgauge.resolve_failed",
                    "remote_host": host,
                },
            )
            return set()
        addresses = {record[4][0] for record in info if record[0] == socket.AF_INET}
        if not addresses:
            logger.warning(
                "OutGauge remote host resolved without IPv4 addresses; disabling host filtering.",
                extra={
                    "event": "outgauge.resolve_failed",
                    "remote_host": host,
                },
            )
            return set()
        return addresses

    def drain_ready(self) -> list[OutGaugePacket]:
        packets = self._processor.drain_ready(time.monotonic())
        return [self._apply_tyre_compound(packet) for packet in packets]

    def _record_packet(self, packet: OutGaugePacket) -> None:
        self._processor.record_packet(packet, time.monotonic())

    def _pop_ready_packet(self, *, now: float, allow_grace: bool) -> Optional[OutGaugePacket]:
        return self._processor.pop_ready_packet(now, allow_grace=allow_grace)


class _AsyncOutGaugeProtocol(asyncio.DatagramProtocol):
    def __init__(self, client: "AsyncOutGaugeUDPClient") -> None:
        self._client = client

    def connection_made(self, transport: asyncio.BaseTransport) -> None:  # pragma: no cover - exercised indirectly
        self._client._connection_made(transport)

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self._client._on_datagram(data, addr)

    def error_received(self, exc: Exception) -> None:  # pragma: no cover - defensive
        self._client._on_error(exc)

    def connection_lost(self, exc: Exception | None) -> None:
        self._client._connection_lost(exc)


class AsyncOutGaugeUDPClient:
    """Asynchronous UDP client delivering :class:`OutGaugePacket` objects."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 3000,
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
        self._remote_addresses = OutGaugeUDPClient._resolve_remote_addresses(host)
        self._timeout = timeout
        self._requested_port = port
        self._buffer_grace = _resolve_buffer_grace(timeout, reorder_grace)
        self._jump_tolerance = max(int(jump_tolerance), 0)
        self._buffer_capacity = buffer_capacity
        self._loop = loop
        self._transport: asyncio.DatagramTransport | None = None
        self._address: Tuple[str, int] = ("", 0)
        self._processor: _OutGaugePacketProcessor | None = None
        self._ready_packets: deque[OutGaugePacket] = deque()
        self._condition: asyncio.Condition | None = None
        self._notify_scheduled = False
        self._pending_error: BaseException | None = None
        self._pending_timer: asyncio.TimerHandle | None = None
        self._timeouts = 0
        self._ignored_hosts = 0
        self._closed_event = asyncio.Event()
        self._closed_event.set()
        self._closing = False
        self._tyre_compounds: dict[str, str] = {}

    @classmethod
    async def create(
        cls,
        host: str = "127.0.0.1",
        port: int = 3000,
        *,
        timeout: float = 0.05,
        reorder_grace: float | None = None,
        jump_tolerance: int = 200,
        buffer_size: int | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> "AsyncOutGaugeUDPClient":
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
            lambda: _AsyncOutGaugeProtocol(self),
            local_addr=(None, self._requested_port),
            family=socket.AF_INET,
        )
        self._transport = transport

    async def __aenter__(self) -> "AsyncOutGaugeUDPClient":
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
                "loss_events": 0,
                "recovered": 0,
        }
        return self._processor.statistics

    def update_tyre_compound(self, car: str, compound: str | None) -> None:
        """Register ``compound`` for ``car`` packets handled by the client."""

        key = _normalise_identifier(car)
        if key is None:
            return
        label = normalise_compound_label(compound)
        if label is None:
            self._tyre_compounds.pop(key, None)
        else:
            self._tyre_compounds[key] = label

    def _apply_tyre_compound(self, packet: OutGaugePacket) -> OutGaugePacket:
        if packet.tyre_compound:
            return packet
        key = _normalise_identifier(packet.car)
        if key is None:
            return packet
        compound = self._tyre_compounds.get(key)
        if compound is not None:
            packet.tyre_compound = compound
        return packet

    async def recv(self, timeout: float | None = None) -> Optional[OutGaugePacket]:
        if self._transport is None or self._processor is None or self._condition is None:
            raise RuntimeError("AsyncOutGaugeUDPClient is not started")
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
                packet = self._ready_packets.popleft()
                return self._apply_tyre_compound(packet)
            if self._closing:
                raise RuntimeError("AsyncOutGaugeUDPClient is closed")
            if timeout == 0.0:
                break
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
            return self._apply_tyre_compound(packet)
        self._timeouts += 1
        logger.warning(
            "OutGauge recv retries exhausted without receiving a packet.",
            extra={
                "event": "outgauge.recv_timeout",
                "retries": 0,
                "timeout": timeout,
                "remote_host": self._remote_host,
                "port": self._address[1],
            },
        )
        return None

    def drain_ready(self) -> list[OutGaugePacket]:
        packets: list[OutGaugePacket] = []
        while self._ready_packets:
            packets.append(self._apply_tyre_compound(self._ready_packets.popleft()))
        if self._processor is not None:
            packets.extend(
                self._apply_tyre_compound(packet)
                for packet in self._processor.drain_ready(time.monotonic())
            )
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
        self._processor = _OutGaugePacketProcessor(
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
                "Ignoring OutGauge datagram from unexpected host.",
                extra={
                    "event": "outgauge.ignored_host",
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
            packet = OutGaugePacket.from_bytes(payload)
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
            self._ready_packets.append(self._apply_tyre_compound(ready))
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
            self._ready_packets.append(self._apply_tyre_compound(packet))
            made_ready = True
        if made_ready:
            self._wake_waiters()
