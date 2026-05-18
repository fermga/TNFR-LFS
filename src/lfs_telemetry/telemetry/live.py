"""Live UDP/TCP ingestion for OutSim, OutGauge and (optional) InSim.

:class:`LiveTelemetry` runs two non-blocking UDP sockets (OutSim + OutGauge)
and, optionally, a TCP InSim client. Packets are decoded, fused by LFS
time_ms, and yielded through an asyncio queue for downstream nodal analysis.

The OutSim listener auto-detects three wire formats based on packet size:

* legacy basic OutSim (64 / 68 B with ID) → :class:`OutSimPacket`,
* extended ``OutSimPack2`` (size driven by ``OutSim Opts`` cfg flags) →
  :class:`OutSimPack2` (gives real per-wheel forces, slip and steer torque).

When extended packets arrive we also derive a basic OutSim snapshot from
them so the downstream pipeline keeps working unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .protocol.insim import InSimClient, RaceContext
from .protocol.packets import (
    OSO_ALL,
    OUTGAUGE_SIZE,
    OUTGAUGE_SIZE_WITH_ID,
    OUTSIM_SIZE,
    OUTSIM_SIZE_WITH_ID,
    OutGaugePacket,
    OutSimPack2,
    OutSimPacket,
    outsim2_size,
)

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class TelemetrySample:
    """Fused snapshot of physics + dashboard at a given LFS time.

    ``outsim2`` is set when the OutSim sender uses extended ``OutSim Opts``
    (recommended ``1ff`` for full 280-byte packets). When present it carries
    real per-wheel telemetry (vertical load, slip ratio, slip angle,
    susp deflect, steer torque, lap distance) which the observables layer
    will prefer over quasi-static estimates.

    ``race_context`` is a *live reference* to the InSim client's
    :class:`RaceContext`; it mutates as new InSim packets arrive. Snapshot
    fields (track, weather, view player, last lap, etc.) at consumption
    time if you need a stable copy.
    """

    time_ms: int
    outsim: OutSimPacket | None = None
    outgauge: OutGaugePacket | None = None
    outsim2: OutSimPack2 | None = None
    race_context: RaceContext | None = None

    @property
    def is_complete(self) -> bool:
        return self.outsim is not None and self.outgauge is not None


@dataclass(slots=True)
class _PendingByTime:
    """Tiny join buffer that pairs OutSim/OutGauge packets by LFS time_ms."""

    window_ms: int = 50
    pending: dict[int, TelemetrySample] = field(default_factory=dict)

    def add_outsim(self, pkt: OutSimPacket) -> TelemetrySample | None:
        sample = self.pending.setdefault(
            pkt.time_ms, TelemetrySample(pkt.time_ms)
        )
        sample.outsim = pkt
        return self._maybe_release(sample)

    def add_outsim2(
        self, pkt: OutSimPack2, basic: OutSimPacket
    ) -> TelemetrySample | None:
        time_ms = basic.time_ms
        sample = self.pending.setdefault(time_ms, TelemetrySample(time_ms))
        sample.outsim = basic
        sample.outsim2 = pkt
        return self._maybe_release(sample)

    def add_outgauge(self, pkt: OutGaugePacket) -> TelemetrySample | None:
        sample = self.pending.setdefault(
            pkt.time_ms, TelemetrySample(pkt.time_ms)
        )
        sample.outgauge = pkt
        return self._maybe_release(sample)

    def _maybe_release(
        self, sample: TelemetrySample
    ) -> TelemetrySample | None:
        if sample.is_complete:
            self.pending.pop(sample.time_ms, None)
            self._evict_old(sample.time_ms)
            return sample
        return None

    def _evict_old(self, now_ms: int) -> None:
        cutoff = now_ms - self.window_ms
        stale = [t for t in self.pending if t < cutoff]
        for t in stale:
            self.pending.pop(t, None)


class LiveTelemetry:
    """Async producer of :class:`TelemetrySample` values from live LFS UDP."""

    def __init__(
        self,
        outsim_port: int = 30000,
        outgauge_port: int = 30001,
        bind_host: str = "0.0.0.0",
        queue_size: int = 1024,
        join_window_ms: int = 50,
        *,
        outsim_opts: int = OSO_ALL,
        insim_host: str | None = None,
        insim_port: int = 29999,
        insim_admin_password: str = "",
        insim_request_mci: bool = False,
        insim_request_nlp: bool = False,
        insim_mci_interval_ms: int = 100,
        insim_nlp_interval_ms: int = 100,
        insim_connect_retry_interval_s: float = 0.0,
    ) -> None:
        self.outsim_port = outsim_port
        self.outgauge_port = outgauge_port
        self.bind_host = bind_host
        self.outsim_opts = int(outsim_opts)
        self.outsim2_size = outsim2_size(self.outsim_opts)
        self.insim_host = insim_host
        self.insim_port = insim_port
        self.insim_admin_password = insim_admin_password
        self.insim_request_mci = insim_request_mci
        self.insim_request_nlp = insim_request_nlp
        self.insim_mci_interval_ms = insim_mci_interval_ms
        self.insim_nlp_interval_ms = insim_nlp_interval_ms
        self.insim_connect_retry_interval_s = float(
            insim_connect_retry_interval_s
        )
        self._queue: asyncio.Queue[TelemetrySample] = asyncio.Queue(queue_size)
        self._buffer = _PendingByTime(window_ms=join_window_ms)
        self._tasks: list[asyncio.Task] = []
        self._sockets: list[socket.socket] = []
        self._insim: InSimClient | None = None

    @property
    def race_context(self) -> RaceContext | None:
        return self._insim.context if self._insim is not None else None

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._sockets = [
            self._open_udp(self.outsim_port),
            self._open_udp(self.outgauge_port),
        ]
        self._tasks = [
            loop.create_task(self._read_outsim(self._sockets[0])),
            loop.create_task(self._read_outgauge(self._sockets[1])),
        ]
        if self.insim_host is not None:
            self._insim = InSimClient(
                host=self.insim_host,
                port=self.insim_port,
                admin_password=self.insim_admin_password,
                request_hlv=True,
                request_mci=self.insim_request_mci,
                request_nlp=self.insim_request_nlp,
                mci_interval_ms=self.insim_mci_interval_ms,
                nlp_interval_ms=self.insim_nlp_interval_ms,
                connect_retry_interval_s=self.insim_connect_retry_interval_s,
            )
            await self._insim.start()

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                # Log unexpected errors during shutdown but keep cleaning up
                # the remaining tasks / sockets.
                _LOG.exception("unexpected error while cancelling task")
        for sock in self._sockets:
            sock.close()
        self._tasks.clear()
        self._sockets.clear()
        if self._insim is not None:
            await self._insim.stop()
            self._insim = None

    async def __aenter__(self) -> LiveTelemetry:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    # -- consumption --------------------------------------------------------

    async def samples(self) -> AsyncIterator[TelemetrySample]:
        """Yield fused samples as they become available."""
        while True:
            yield await self._queue.get()

    # -- internals ----------------------------------------------------------

    def _open_udp(self, port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.bind_host, port))
        sock.setblocking(False)
        return sock

    async def _read_outsim(self, sock: socket.socket) -> None:
        loop = asyncio.get_running_loop()
        basic_sizes = (OUTSIM_SIZE, OUTSIM_SIZE_WITH_ID)
        ext_size = self.outsim2_size
        while True:
            data = await loop.sock_recv(sock, 4096)
            sample: TelemetrySample | None = None
            if len(data) in basic_sizes:
                try:
                    pkt = OutSimPacket.parse(data)
                except ValueError:
                    continue
                sample = self._buffer.add_outsim(pkt)
            elif len(data) == ext_size:
                try:
                    pkt2 = OutSimPack2.parse(data, self.outsim_opts)
                except ValueError:
                    continue
                basic = _outsim2_to_basic(pkt2)
                if basic is None:
                    continue
                sample = self._buffer.add_outsim2(pkt2, basic)
            else:
                continue
            if sample is not None:
                if self._insim is not None:
                    sample.race_context = self._insim.context
                await self._queue.put(sample)

    async def _read_outgauge(self, sock: socket.socket) -> None:
        loop = asyncio.get_running_loop()
        while True:
            data = await loop.sock_recv(sock, 4096)
            if len(data) not in (OUTGAUGE_SIZE, OUTGAUGE_SIZE_WITH_ID):
                continue
            try:
                pkt = OutGaugePacket.parse(data)
            except ValueError:
                continue
            sample = self._buffer.add_outgauge(pkt)
            if sample is not None:
                if self._insim is not None:
                    sample.race_context = self._insim.context
                await self._queue.put(sample)


def _outsim2_to_basic(pkt2: OutSimPack2) -> OutSimPacket | None:
    """Project an :class:`OutSimPack2` into the legacy :class:`OutSimPacket`.

    Returns ``None`` if the extended packet does not include the OSO_TIME or
    OSO_MAIN sections (those carry the basic-OutSim payload).
    """
    if pkt2.time_ms is None or pkt2.ang_vel is None:
        return None
    return OutSimPacket(
        time_ms=pkt2.time_ms,
        ang_vel=pkt2.ang_vel,
        heading=pkt2.heading or 0.0,
        pitch=pkt2.pitch or 0.0,
        roll=pkt2.roll or 0.0,
        accel=pkt2.accel or (0.0, 0.0, 0.0),
        vel=pkt2.vel or (0.0, 0.0, 0.0),
        pos=pkt2.pos or (0.0, 0.0, 0.0),
        packet_id=pkt2.packet_id,
    )


__all__ = ["TelemetrySample", "LiveTelemetry"]
