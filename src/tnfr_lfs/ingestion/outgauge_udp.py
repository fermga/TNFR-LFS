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

from tnfr_lfs.ingestion._socket_poll import wait_for_read_ready
from tnfr_lfs.ingestion._reorder_buffer import (
    CircularReorderBuffer,
    DEFAULT_REORDER_BUFFER_SIZE,
)

__all__ = ["AsyncOutGaugeUDPClient", "OutGaugePacket", "OutGaugeUDPClient"]


logger = logging.getLogger(__name__)


_PACK_STRUCT = struct.Struct("<I4s16s8s6s6sHBBfffffffIIfff16s16sI")
_FLOAT_STRUCT = struct.Struct("<f")
_EXTENDED_BLOCKS = 5
_EXTENDED_BLOCK_WIDTH = 4
_MAX_DATAGRAM_SIZE = _PACK_STRUCT.size + _EXTENDED_BLOCKS * _EXTENDED_BLOCK_WIDTH * _FLOAT_STRUCT.size


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

    def record_packet(self, packet: OutGaugePacket, arrival: float) -> None:
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
            return

        self._buffer.insert(arrival, packet, packet.packet_id)

    def pop_ready_packet(self, now: float, *, allow_grace: bool) -> Optional[OutGaugePacket]:
        while self._buffer:
            peeked = self._buffer.peek_oldest()
            if peeked is None:
                break
            arrival, packet = peeked
            if not allow_grace and len(self._buffer) == 1 and (now - arrival) < self._buffer_grace:
                break
            popped = self._buffer.pop_oldest()
            if popped is None:
                break
            _, packet = popped
            if (
                self._last_emitted_id is not None
                and packet.packet_id <= self._last_emitted_id
            ):
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

    def _is_duplicate(self, packet_id: int) -> bool:
        if self._last_emitted_id is not None and packet_id == self._last_emitted_id:
            return True
        return self._buffer.contains_key(packet_id)


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
            it.  ``None`` defaults to ``timeout``.
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
        buffer_grace = max(float(reorder_grace) if reorder_grace is not None else timeout, 0.0)
        self._processor = _OutGaugePacketProcessor(
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

    def recv(self) -> Optional[OutGaugePacket]:
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
                continue
            if not wait_for_read_ready(
                self._socket,
                timeout=self._timeout,
                deadline=deadline,
            ):
                break
        ready = self._processor.pop_ready_packet(time.monotonic(), allow_grace=True)
        if ready is not None:
            return ready
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
            return {host}
        addresses = {record[4][0] for record in info if record[0] == socket.AF_INET}
        return addresses or {host}

    def drain_ready(self) -> list[OutGaugePacket]:
        return self._processor.drain_ready(time.monotonic())


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
        self._buffer_grace = max(float(reorder_grace) if reorder_grace is not None else timeout, 0.0)
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
        self._timeouts = 0
        self._ignored_hosts = 0
        self._closed_event = asyncio.Event()
        self._closed_event.set()
        self._closing = False

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
                return self._ready_packets.popleft()
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
            return packet
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
            packets.append(self._ready_packets.popleft())
        if self._processor is not None:
            packets.extend(self._processor.drain_ready(time.monotonic()))
        return packets

    async def close(self) -> None:
        if self._closing:
            await self._closed_event.wait()
            return
        self._closing = True
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
            self._ready_packets.append(ready)
            made_ready = True
        if made_ready:
            self._wake_waiters()

    def _on_error(self, exc: Exception) -> None:
        self._pending_error = exc
        self._wake_waiters()

    def _connection_lost(self, _exc: Exception | None) -> None:
        condition = self._condition
        self._transport = None
        self._processor = None
        self._condition = None
        self._ready_packets.clear()
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
