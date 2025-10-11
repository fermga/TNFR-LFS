"""Reusable helpers for UDP-focused tests."""

from __future__ import annotations

import asyncio
import socket
import time
from collections import deque
from typing import Any, Awaitable, Callable, Deque, Iterable, List, Sequence

import pytest
from _pytest.monkeypatch import MonkeyPatch

from tnfr_lfs.ingestion import outgauge_udp as outgauge_module
from tnfr_lfs.ingestion import outsim_udp as outsim_module

UDPPayload = bytes | tuple[bytes, tuple[str, int]]


def raise_gaierror(*_args: object, **_kwargs: object) -> list[object]:
    """Raise :class:`socket.gaierror` for monkeypatched ``getaddrinfo`` calls."""

    import socket

    raise socket.gaierror()


class QueueUDPSocket:
    """A minimal UDP socket stub backed by a queue of payloads."""

    def __init__(
        self,
        payloads: Iterable[UDPPayload] | None = None,
        *,
        address: tuple[str, int] = ("127.0.0.1", 0),
        queue: Deque[UDPPayload] | None = None,
    ) -> None:
        self.queue: Deque[UDPPayload] = queue if queue is not None else deque()
        self._default_address = (address[0] or "127.0.0.1", address[1])
        if payloads is not None:
            self.extend(payloads)

    def bind(self, address: tuple[str, int]) -> None:  # pragma: no cover - compatibility shim
        self._default_address = (address[0] or "127.0.0.1", address[1])

    def getsockname(self) -> tuple[str, int]:
        return self._default_address

    def setblocking(self, _flag: bool) -> None:  # pragma: no cover - compatibility shim
        return

    def append(self, payload: UDPPayload) -> None:
        self.queue.append(payload)

    def extend(self, payloads: Iterable[UDPPayload]) -> None:
        self.queue.extend(payloads)

    def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
        if not self.queue:
            raise BlockingIOError()
        item = self.queue.popleft()
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], tuple):
            payload, address = item
        else:
            payload, address = item, self._default_address
        return payload, address

    def close(self) -> None:
        self.queue.clear()


SelectHook = Callable[[list[object], list[object], list[object], float | None], tuple[list[object], list[object], list[object]] | None]


def make_select_stub(
    *,
    hook: SelectHook | None = None,
) -> tuple[
    Callable[[list[object], list[object], list[object], float | None], tuple[list[object], list[object], list[object]]],
    List[float | None],
]:
    """Return a ``select.select`` replacement that records timeouts."""

    timeouts: List[float | None] = []

    def fake_select(
        read: list[object],
        write: list[object],
        err: list[object],
        timeout: float | None = None,
    ) -> tuple[list[object], list[object], list[object]]:
        timeouts.append(timeout)
        if hook is not None:
            result = hook(read, write, err, timeout)
            if result is not None:
                return result
        return ([], [], [])

    return fake_select, timeouts


WaitHook = Callable[[object, float, float | None], bool | None]


def make_wait_stub(
    *,
    hook: WaitHook | None = None,
    return_value: bool = True,
) -> tuple[Callable[..., bool], List[float]]:
    """Return a ``wait_for_read_ready`` replacement that records timeouts."""

    timeouts: List[float] = []

    def fake_wait(sock: object, *, timeout: float, deadline: float | None) -> bool:
        timeouts.append(timeout)
        if hook is not None:
            result = hook(sock, timeout, deadline)
            if result is not None:
                return result
        return return_value

    return fake_wait, timeouts


def _resolve_payload_spec(
    payload_factory: Callable[..., UDPPayload],
    spec: Any,
) -> UDPPayload:
    """Build a UDP payload from ``spec`` using ``payload_factory``.

    ``spec`` may be a mapping of keyword arguments, a sequence of positional
    arguments, or a single positional argument for the factory.
    """

    if isinstance(spec, dict):
        return payload_factory(**spec)
    if isinstance(spec, Sequence) and not isinstance(spec, (bytes, bytearray, str)):
        return payload_factory(*spec)
    return payload_factory(spec)


def build_payload_batch(
    payload_factory: Callable[..., UDPPayload],
    specs: Iterable[Any],
) -> list[UDPPayload]:
    """Return a list of payloads constructed from ``specs``."""

    return [_resolve_payload_spec(payload_factory, spec) for spec in specs]


def enqueue_payloads(
    queue: Deque[UDPPayload],
    payload_factory: Callable[..., UDPPayload],
    specs: Iterable[Any],
) -> None:
    """Append a series of payloads constructed from ``specs`` to ``queue``."""

    queue.extend(build_payload_batch(payload_factory, specs))


def patch_udp_client_socket(
    monkeypatch: MonkeyPatch,
    client: Any,
    *,
    address: tuple[str, int] = ("127.0.0.1", 0),
) -> QueueUDPSocket:
    """Replace ``client``'s socket with :class:`QueueUDPSocket`."""

    fake_socket = QueueUDPSocket(address=address)
    original_socket = client._socket
    monkeypatch.setattr(client, "_socket", fake_socket)
    original_socket.close()
    return fake_socket


def patch_wait_for_read_ready(
    monkeypatch: MonkeyPatch,
    module: Any,
    *,
    hook: WaitHook | None = None,
    return_value: bool = True,
) -> list[float]:
    """Replace ``module.wait_for_read_ready`` with a controllable stub."""

    fake_wait, timeouts = make_wait_stub(hook=hook, return_value=return_value)
    monkeypatch.setattr(module, "wait_for_read_ready", fake_wait)
    return timeouts


def extend_queue_on_wait(
    queue: Deque[UDPPayload],
    iterable_factory: Callable[[], Iterable[UDPPayload]],
) -> WaitHook:
    """Return a wait hook that extends ``queue`` when it is empty."""

    def on_wait(_sock: object, _timeout: float, _deadline: float | None) -> None:
        if not queue:
            queue.extend(iterable_factory())
        return None

    return on_wait


def append_once_on_wait(
    queue: Deque[UDPPayload],
    payload_factory: Callable[[], UDPPayload],
) -> WaitHook:
    """Return a wait hook that appends a payload to ``queue`` at most once."""

    appended = False

    def on_wait(_sock: object, _timeout: float, _deadline: float | None) -> None:
        nonlocal appended
        if not appended:
            queue.append(payload_factory())
            appended = True
        return None

    return on_wait


def assert_udp_batch_drained(
    monkeypatch: MonkeyPatch,
    *,
    client_factory: Callable[[], Any],
    module: Any,
    payload_factory: Callable[..., UDPPayload],
    batch_specs: Iterable[Any],
    expected_values: list[Any],
    value_extractor: Callable[[Any], Any],
    address: tuple[str, int] = ("127.0.0.1", 0),
    release_packets: bool = False,
) -> None:
    """Assert that a UDP client drains a queued batch after a wait call."""

    client = client_factory()
    timeout = client._timeout
    fake_socket = patch_udp_client_socket(monkeypatch, client, address=address)
    hook = extend_queue_on_wait(
        fake_socket.queue,
        lambda: build_payload_batch(payload_factory, batch_specs),
    )
    wait_calls = patch_wait_for_read_ready(monkeypatch, module, hook=hook)

    try:
        packets = [client.recv() for _ in range(len(expected_values))]
    finally:
        client.close()

    extracted = [value_extractor(packet) for packet in packets if packet is not None]
    assert extracted == expected_values
    if release_packets:
        for packet in packets:
            if packet is not None:
                packet.release()
    assert wait_calls, "wait_for_read_ready should record at least one call"
    assert wait_calls[0] == pytest.approx(timeout, rel=0.1)


def assert_udp_pending_flush(
    monkeypatch: MonkeyPatch,
    *,
    client_factory: Callable[[], Any],
    module: Any,
    payload_factory: Callable[..., UDPPayload],
    first_spec: Any,
    successor_spec: Any,
    appended_spec: Any,
    expected_first: Any,
    expected_second: Any,
    value_extractor: Callable[[Any], Any],
    max_elapsed: float,
    address: tuple[str, int] = ("127.0.0.1", 0),
    release_packets: bool = False,
) -> None:
    """Assert that a pending packet flushes quickly when a successor arrives."""

    client = client_factory()
    fake_socket = patch_udp_client_socket(monkeypatch, client, address=address)
    enqueue_payloads(fake_socket.queue, payload_factory, [first_spec])

    try:
        first_packet = client.recv()
        assert first_packet is not None
        assert value_extractor(first_packet) == expected_first
        if release_packets:
            first_packet.release()

        enqueue_payloads(fake_socket.queue, payload_factory, [successor_spec])
        hook = append_once_on_wait(
            fake_socket.queue,
            lambda: _resolve_payload_spec(payload_factory, appended_spec),
        )
        wait_calls = patch_wait_for_read_ready(monkeypatch, module, hook=hook)

        start = time.perf_counter()
        next_packet = client.recv()
        elapsed = time.perf_counter() - start
        assert next_packet is not None
        assert value_extractor(next_packet) == expected_second
        assert elapsed < max_elapsed
        if release_packets:
            next_packet.release()
        assert wait_calls, "wait_for_read_ready should be invoked during flush"
    finally:
        client.close()


def assert_udp_isolated_flush(
    monkeypatch: MonkeyPatch,
    *,
    client_factory: Callable[[], Any],
    module: Any,
    payload_factory: Callable[..., UDPPayload],
    first_spec: Any,
    second_spec: Any,
    expected_first: Any,
    expected_second: Any,
    value_extractor: Callable[[Any], Any],
    address: tuple[str, int] = ("127.0.0.1", 0),
    release_packets: bool = False,
    max_elapsed_ms: float = 10.0,
    max_wait_timeout: float = 0.012,
) -> None:
    """Assert that an isolated pending packet flushes quickly by default."""

    client = client_factory()
    fake_socket = patch_udp_client_socket(monkeypatch, client, address=address)
    wait_calls = patch_wait_for_read_ready(
        monkeypatch,
        module,
        return_value=False,
    )

    try:
        enqueue_payloads(fake_socket.queue, payload_factory, [first_spec])
        first_packet = client.recv()
        assert first_packet is not None
        assert value_extractor(first_packet) == expected_first
        if release_packets:
            first_packet.release()

        enqueue_payloads(fake_socket.queue, payload_factory, [second_spec])
        start = time.perf_counter()
        second_packet = client.recv()
        elapsed_ms = (time.perf_counter() - start) * 1_000
        assert second_packet is not None
        assert value_extractor(second_packet) == expected_second
        assert elapsed_ms < max_elapsed_ms
        if release_packets:
            second_packet.release()
        assert wait_calls, "wait_for_read_ready should be consulted for pending packet"
        assert wait_calls[0] <= max_wait_timeout
    finally:
        client.close()


async def assert_async_udp_reordering(
    *,
    client_factory: Callable[[], Awaitable[Any]],
    payload_factory: Callable[..., UDPPayload],
    send_specs: Iterable[Any],
    expected_values: list[Any],
    value_extractor: Callable[[Any], Any],
    release_packets: bool = False,
) -> None:
    """Assert that an async UDP client reorders packets delivered out of order."""

    client = await client_factory()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.bind(("127.0.0.1", 0))
    try:
        _, port = client.address
        target = ("127.0.0.1", port)
        await asyncio.sleep(0)
        for spec in send_specs:
            sender.sendto(_resolve_payload_spec(payload_factory, spec), target)

        results = []
        for _ in range(len(expected_values)):
            packet = await client.recv()
            if packet is not None:
                results.append(packet)

        assert [value_extractor(packet) for packet in results] == expected_values
        stats = client.statistics
        assert stats["delivered"] == len(expected_values)
        assert stats["reordered"] >= 1
        if release_packets:
            for packet in results:
                packet.release()
    finally:
        sender.close()
        await client.close()


async def assert_async_client_close_wakes(
    *,
    client_factory: Callable[[], Awaitable[Any]],
) -> None:
    """Assert that awaiting receivers wake when an async UDP client closes."""

    client = await client_factory()
    try:
        recv_task = asyncio.create_task(client.recv())
        await asyncio.sleep(0)
        await client.close()
        with pytest.raises(RuntimeError):
            await recv_task
    finally:
        await client.close()


def pad_outgauge_field(value: str, size: int, *, encoding: str = "latin-1") -> bytes:
    """Pad an OutGauge string field to the expected byte length."""

    return value.encode(encoding).ljust(size, b"\x00")


def build_outgauge_payload(
    packet_id: int,
    time_value: int,
    *,
    car: str = "XFG",
    player_name: str = "Driver",
    plate: str = "",
    track: str = "BL1",
    layout: str = "",
    flags: int = 0,
    gear: int = 3,
    plid: int = 0,
    speed: float = 50.0,
    rpm: float = 4_000.0,
    turbo: float = 0.0,
    eng_temp: float = 80.0,
    fuel: float = 30.0,
    oil_pressure: float = 0.0,
    oil_temp: float = 90.0,
    dash_lights: int = 0,
    show_lights: int = 0,
    throttle: float = 0.5,
    brake: float = 0.1,
    clutch: float = 0.0,
    display1: str = "",
    display2: str = "",
) -> bytes:
    """Construct a minimal OutGauge datagram for the requested packet id/time."""

    return outgauge_module._PACK_STRUCT.pack(
        time_value,
        pad_outgauge_field(car, 4),
        pad_outgauge_field(player_name, 16),
        pad_outgauge_field(plate, 8),
        pad_outgauge_field(track, 6),
        pad_outgauge_field(layout, 6),
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
        pad_outgauge_field(display1, 16),
        pad_outgauge_field(display2, 16),
        packet_id,
    )


def build_outsim_payload(
    time_ms: int,
    *,
    base_values: Iterable[float] | None = None,
) -> bytes:
    """Construct a minimal OutSim datagram for the requested timestamp."""

    values = list(base_values) if base_values is not None else [0.0] * 15
    return outsim_module._BASE_STRUCT.pack(time_ms, *values)
