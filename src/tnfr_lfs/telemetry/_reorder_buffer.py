"""Utilities for managing monotonic arrival buffers.

This module provides a lightweight circular buffer that keeps the stored
packets ordered by their sequence key (``time`` for OutSim and
``packet_id`` for OutGauge).  The implementation uses a preallocated
storage area and binary search to minimise per-packet allocations while
still allowing arbitrary insert positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterator, Optional, TypeVar


T = TypeVar("T")


DEFAULT_REORDER_BUFFER_SIZE = 64


@dataclass(frozen=True)
class EvictedEntry(Generic[T]):
    """Record describing an entry discarded due to capacity limits."""

    arrival: float
    packet: T
    key: int


class CircularReorderBuffer(Generic[T]):
    """Circular buffer that maintains packets ordered by an integer key."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("CircularReorderBuffer requires a positive capacity")
        self._capacity = capacity
        self._arrivals: list[float] = [0.0] * capacity
        self._packets: list[Optional[T]] = [None] * capacity
        self._keys: list[int] = [0] * capacity
        self._start = 0
        self._size = 0
        self._keys_present: set[int] = set()

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:  # pragma: no cover - delegating to __len__
        return self._size > 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def clear(self) -> None:
        self._start = 0
        self._size = 0
        self._keys_present.clear()

    def contains_key(self, key: int) -> bool:
        return key in self._keys_present

    def peek_oldest(self) -> Optional[tuple[float, T]]:
        if not self._size:
            return None
        idx = self._start
        packet = self._packets[idx]
        assert packet is not None  # for type-checkers
        return self._arrivals[idx], packet

    def pop_oldest(self) -> Optional[tuple[float, T]]:
        if not self._size:
            return None
        idx = self._start
        arrival = self._arrivals[idx]
        packet = self._packets[idx]
        self._packets[idx] = None
        self._start = (self._start + 1) % self._capacity
        self._size -= 1
        self._keys_present.discard(self._keys[idx])
        if packet is None:  # pragma: no cover - defensive guard
            raise RuntimeError("CircularReorderBuffer stored None packet")
        return arrival, packet

    def insert(self, arrival: float, packet: T, key: int) -> tuple[int, Optional[EvictedEntry[T]]]:
        """Insert ``packet`` using ``key`` to keep the buffer sorted.

        Returns the position where the packet was inserted and, when the
        buffer was already full, information about the evicted entry.
        """

        position = self._find_insert_position(key)
        evicted: Optional[EvictedEntry[T]] = None
        if self._size == self._capacity:
            evicted_arrival, evicted_packet, evicted_key = self._pop_oldest_raw()
            if position > 0:
                position -= 1
            if evicted_packet is not None:
                evicted = EvictedEntry(evicted_arrival, evicted_packet, evicted_key)

        self._shift_right_from(position)
        idx = self._logical_index(position)
        self._arrivals[idx] = arrival
        self._packets[idx] = packet
        self._keys[idx] = key
        self._size += 1
        self._keys_present.add(key)
        return position, evicted

    def __iter__(self) -> Iterator[tuple[float, T, int]]:
        for index in range(self._size):
            logical = self._logical_index(index)
            packet = self._packets[logical]
            if packet is None:  # pragma: no cover - defensive guard
                raise RuntimeError("CircularReorderBuffer stored None packet")
            yield self._arrivals[logical], packet, self._keys[logical]

    def _logical_index(self, offset: int) -> int:
        return (self._start + offset) % self._capacity

    def _find_insert_position(self, key: int) -> int:
        low, high = 0, self._size
        while low < high:
            mid = (low + high) // 2
            mid_key = self._keys[self._logical_index(mid)]
            if key < mid_key:
                high = mid
            else:
                low = mid + 1
        return low

    def _shift_right_from(self, position: int) -> None:
        if position >= self._size:
            return
        for index in range(self._size, position, -1):
            src = self._logical_index(index - 1)
            dest = self._logical_index(index)
            self._arrivals[dest] = self._arrivals[src]
            self._packets[dest] = self._packets[src]
            self._keys[dest] = self._keys[src]

    def _pop_oldest_raw(self) -> tuple[float, Optional[T], int]:
        idx = self._start
        arrival = self._arrivals[idx]
        packet = self._packets[idx]
        key = self._keys[idx]
        self._packets[idx] = None
        self._start = (self._start + 1) % self._capacity
        self._size -= 1
        self._keys_present.discard(key)
        return arrival, packet, key

