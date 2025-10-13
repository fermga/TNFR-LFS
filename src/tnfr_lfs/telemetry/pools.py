"""Reusable packet pools for UDP ingestion."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from typing import Callable, Deque, Generic, Iterator, TypeVar

__all__ = ["PacketPool", "PoolItem"]


T = TypeVar("T", bound="PoolItem")


class PoolItem:
    """Mixin providing lifecycle hooks for pooled objects."""

    __slots__ = ("_pool", "_in_use")

    def __init__(self) -> None:
        self._pool: PacketPool[PoolItem] | None = None
        self._in_use = False

    def release(self) -> None:
        """Return the instance to its pool."""

        if not self._in_use:
            return
        pool = self._pool
        self._in_use = False
        if pool is not None:
            pool._release(self)

    def _attach(self, pool: "PacketPool[PoolItem]") -> None:
        self._pool = pool
        self._in_use = True

    def _reset(self) -> None:  # pragma: no cover - documentation default
        """Reset the internal state before returning to the pool."""


class PacketPool(Generic[T]):
    """Simple reusable-object pool with optional size limits."""

    __slots__ = ("_factory", "_items", "_max_size")

    def __init__(self, factory: Callable[[], T], *, max_size: int | None = None) -> None:
        self._factory = factory
        self._items: Deque[T] = deque()
        self._max_size = max_size

    def acquire(self) -> T:
        try:
            item = self._items.pop()
        except IndexError:
            item = self._factory()
        item._attach(self)  # type: ignore[attr-defined]
        return item

    @contextmanager
    def get(self) -> Iterator[T]:
        item = self.acquire()
        try:
            yield item
        finally:
            item.release()

    def _release(self, item: PoolItem) -> None:
        item._reset()
        if self._max_size is not None and len(self._items) >= self._max_size:
            return
        self._items.append(item)  # type: ignore[arg-type]

