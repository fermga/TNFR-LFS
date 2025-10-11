"""Reusable helpers for UDP-focused tests."""

from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Iterable, List

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
