# `tnfr_lfs.telemetry.pools` module
Reusable packet pools for UDP ingestion.

## Classes
### `PoolItem`
Mixin providing lifecycle hooks for pooled objects.

#### Methods
- `release(self) -> None`
  - Return the instance to its pool.

### `PacketPool` (Generic[T])
Simple reusable-object pool with optional size limits.

#### Methods
- `acquire(self) -> T`
- `get(self) -> Iterator[T]`

## Attributes
- `T = TypeVar('T', bound='PoolItem')`

