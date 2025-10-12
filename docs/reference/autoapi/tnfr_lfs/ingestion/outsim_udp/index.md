# `tnfr_lfs.ingestion.outsim_udp` module
OutSim UDP client implementation.

The Live for Speed simulator broadcasts physics-oriented telemetry using
the OutSim protocol.  Packets are transmitted as a binary structure over
UDP and contain orientation, acceleration and position data.  This
module provides a small client capable of decoding the official packet
layout while operating in non-blocking mode so that it can be safely
polled from high-frequency loops.

``OutSimUDPClient`` keeps a short reordering buffer keyed by the
``OutSimPacket.time`` field, automatically deduplicating, re-sequencing
and tracking suspected gaps.  Applications may call
:meth:`OutSimUDPClient.drain_ready` to extract the current buffer without
blocking and inspect :attr:`OutSimUDPClient.statistics` to monitor
packet loss or recovery counters emitted while the client operates.

## Classes
### `FrozenOutSimDriverInputs`

### `FrozenOutSimWheelState`

### `FrozenOutSimPacket`

### `OutSimDriverInputs`

#### Methods
- `clear(self) -> None`
- `set_values(self, throttle: float, brake: float, clutch: float, handbrake: float, steer: float) -> None`
- `freeze(self) -> FrozenOutSimDriverInputs`

### `OutSimWheelState`

#### Methods
- `clear(self) -> None`
- `set_values(self, slip_ratio: float, slip_angle: float, long_force: float, lat_force: float, load: float, suspension_deflection: float) -> None`
- `freeze(self) -> FrozenOutSimWheelState`

### `OutSimPacket` (PoolItem)
Representation of a decoded OutSim datagram.

#### Methods
- `inputs(self) -> Optional[OutSimDriverInputs]`
- `inputs(self, value: Optional[OutSimDriverInputs]) -> None`
- `wheels(self) -> Tuple[OutSimWheelState, OutSimWheelState, OutSimWheelState, OutSimWheelState]`
- `freeze(self) -> FrozenOutSimPacket`
- `from_bytes(cls, payload: bytes, *, freeze: bool = False) -> 'OutSimPacket | FrozenOutSimPacket'`
  - Deserialize a byte payload following the official layout.

### `OutSimUDPClient`
Non-blocking UDP client that yields :class:`OutSimPacket` objects.

#### Methods
- `address(self) -> Tuple[str, int]`
  - Return the bound socket address.
- `timeouts(self) -> int`
  - Number of calls that exhausted the retry budget.
- `ignored_hosts(self) -> int`
  - Number of datagrams dropped due to unexpected hosts.
- `statistics(self) -> dict[str, int]`
  - Return packet accounting metrics collected by the client.
- `recv(self) -> Optional[OutSimPacket]`
  - Attempt to receive a packet, returning ``None`` on timeout.
- `drain_ready(self) -> list[OutSimPacket]`
  - Return all packets ready for immediate consumption.

The method never blocks; it simply flushes the buffered packets whose
arrival timestamps have aged beyond ``reorder_grace`` or which are
already followed by newer datagrams.
- `close(self) -> None`
  - Close the underlying socket.

### `AsyncOutSimUDPClient`
Asynchronous UDP client delivering :class:`OutSimPacket` objects.

#### Methods
- `address(self) -> Tuple[str, int]`
- `timeouts(self) -> int`
- `ignored_hosts(self) -> int`
- `statistics(self) -> dict[str, int]`
- `drain_ready(self) -> list[OutSimPacket]`

## Attributes
- `logger = logging.getLogger(__name__)`
- `OUTSIM_MAX_PACKET_SIZE = _BASE_STRUCT.size + struct.calcsize('<I') + _INPUT_STRUCT.size + 4 * _WHEEL_STRUCT.size`

