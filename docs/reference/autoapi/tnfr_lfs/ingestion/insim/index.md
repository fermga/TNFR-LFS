# `tnfr_lfs.ingestion.insim` module
Minimal InSim TCP client used to control Live for Speed overlays.

## Classes
### `ButtonLayout`
Simple description of an overlay button region.

#### Methods
- `clamp(self) -> 'ButtonLayout'`
  - Clamp coordinates to the limits accepted by ``IS_BTN`` packets.

### `ButtonEvent`
Simplified view of ``IS_BTC`` button control packets.

### `InSimClient`
Very small InSim TCP client focused on button overlays.

#### Methods
- `connected(self) -> bool`
- `connect(self) -> None`
- `close(self) -> None`
- `send_keepalive(self) -> None`
- `subscribe_controls(self) -> None`
  - Request control packets (button events) from Live for Speed.
- `poll_button(self, timeout: float = 0.0) -> Optional[ButtonEvent]`
  - Poll for a pending ``IS_BTC`` packet without blocking.

Returns ``None`` if no packet is available within ``timeout`` seconds
or if the received packet is not a button click event (``TypeIn`` != 0).
- `send_button(self, text: str, layout: ButtonLayout | None = None) -> None`
  - Render or update a button overlay using ``IS_BTN`` packets.
- `send_command(self, command: str) -> None`
  - Send a chat command (e.g. ``/press``) to Live for Speed.
- `clear_button(self, layout: ButtonLayout | None = None) -> None`
  - Remove an overlay button using the ``BTN_STYLE_CLEAR`` flag.

### `OverlayManager`
Helper that keeps a button overlay alive via InSim.

#### Methods
- `connect(self) -> None`
- `show(self, lines: Sequence[str] | str) -> None`
- `hide(self) -> None`
- `clear(self) -> None`
  - Public alias for :meth:`hide` used by long-running loops.
- `tick(self) -> None`
- `poll_button(self, timeout: float = 0.0) -> Optional[ButtonEvent]`
  - Proxy to :meth:`InSimClient.poll_button`.
- `close(self) -> None`

### `MacroStep`
Single command scheduled for future execution.

### `MacroQueue`
Schedule ``/press`` command sequences with spacing safeguards.

#### Methods
- `clear(self) -> None`
- `enqueue_press(self, key: str, *, spacing: float | None = None) -> None`
- `enqueue_press_sequence(self, keys: Sequence[str], *, spacing: float | None = None) -> None`
- `tick(self) -> int`
  - Dispatch ready commands.

Returns the number of commands sent during the tick.
- `pending(self) -> Sequence[str]`

