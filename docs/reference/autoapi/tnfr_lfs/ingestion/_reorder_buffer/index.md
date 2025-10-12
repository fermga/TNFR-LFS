# `tnfr_lfs.ingestion._reorder_buffer` module
Utilities for managing monotonic arrival buffers.

This module provides a lightweight circular buffer that keeps the stored
packets ordered by their sequence key (``time`` for OutSim and
``packet_id`` for OutGauge).  The implementation uses a preallocated
storage area and binary search to minimise per-packet allocations while
still allowing arbitrary insert positions.

## Classes
### `EvictedEntry` (Generic[T])
Record describing an entry discarded due to capacity limits.

### `CircularReorderBuffer` (Generic[T])
Circular buffer that maintains packets ordered by an integer key.

#### Methods
- `capacity(self) -> int`
- `clear(self) -> None`
- `contains_key(self, key: int) -> bool`
- `peek_oldest(self) -> Optional[tuple[float, T]]`
- `pop_oldest(self) -> Optional[tuple[float, T]]`
- `insert(self, arrival: float, packet: T, key: int) -> tuple[int, Optional[EvictedEntry[T]]]`
  - Insert ``packet`` using ``key`` to keep the buffer sorted.

Returns the position where the packet was inserted and, when the
buffer was already full, information about the evicted entry.

## Attributes
- `T = TypeVar('T')`
- `DEFAULT_REORDER_BUFFER_SIZE = 64`

