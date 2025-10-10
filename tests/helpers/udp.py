"""Reusable helpers for UDP-focused tests."""

from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Iterable, List

UDPPayload = bytes | tuple[bytes, tuple[str, int]]


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
