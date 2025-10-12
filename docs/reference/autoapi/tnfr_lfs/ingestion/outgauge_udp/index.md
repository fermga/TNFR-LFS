# `tnfr_lfs.ingestion.outgauge_udp` module
OutGauge UDP client implementation.

The client understands Live for Speed's OutGauge datagrams and preserves a
small buffer keyed by :attr:`OutGaugePacket.packet_id` so that late
packets can be reinserted in-order.  Duplicate datagrams are discarded and
suspected gaps are tracked via :attr:`OutGaugeUDPClient.statistics`,
allowing monitoring code to differentiate between packet loss and
successful recoveries.

## Classes
### `FrozenOutGaugePacket`

### `OutGaugePacket` (PoolItem)
Representation of a decoded OutGauge datagram.

#### Methods
- `freeze(self) -> FrozenOutGaugePacket`
- `from_bytes(cls, payload: bytes, *, freeze: bool = False) -> 'OutGaugePacket | FrozenOutGaugePacket'`

### `OutGaugeUDPClient`
Non-blocking UDP client for OutGauge telemetry.

#### Methods
- `address(self) -> Tuple[str, int]`
- `timeouts(self) -> int`
- `ignored_hosts(self) -> int`
- `statistics(self) -> dict[str, int]`
  - Return packet accounting metrics collected by the client.
- `recv(self) -> Optional[OutGaugePacket]`
- `close(self) -> None`
- `drain_ready(self) -> list[OutGaugePacket]`

### `AsyncOutGaugeUDPClient`
Asynchronous UDP client delivering :class:`OutGaugePacket` objects.

#### Methods
- `address(self) -> Tuple[str, int]`
- `timeouts(self) -> int`
- `ignored_hosts(self) -> int`
- `statistics(self) -> dict[str, int]`
- `drain_ready(self) -> list[OutGaugePacket]`

## Attributes
- `logger = logging.getLogger(__name__)`

