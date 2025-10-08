"""Minimal InSim TCP client used to control Live for Speed overlays."""

from __future__ import annotations

import select
import socket
import struct
from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Callable, Deque, Optional, Sequence


@dataclass(slots=True)
class ButtonLayout:
    """Simple description of an overlay button region."""

    left: int = 80
    top: int = 20
    width: int = 40
    height: int = 16
    ucid: int = 0
    inst: int = 0
    click_id: int = 1
    style: int = 0x10
    type_in: int = 0

    def clamp(self) -> "ButtonLayout":
        """Clamp coordinates to the limits accepted by ``IS_BTN`` packets."""

        def _clamp(value: int) -> int:
            return max(0, min(200, int(value)))

        return ButtonLayout(
            left=_clamp(self.left),
            top=_clamp(self.top),
            width=_clamp(self.width),
            height=_clamp(self.height),
            ucid=max(0, min(255, int(self.ucid))),
            inst=max(0, min(255, int(self.inst))),
            click_id=max(0, min(255, int(self.click_id))),
            style=max(0, min(255, int(self.style))),
            type_in=max(0, min(255, int(self.type_in))),
        )


@dataclass(slots=True)
class ButtonEvent:
    """Simplified view of ``IS_BTC`` button control packets."""

    ucid: int
    click_id: int
    inst: int
    type_in: int
    typed_char: Optional[str] = None
    flags: int = 0


class InSimClient:
    """Very small InSim TCP client focused on button overlays."""

    ISP_ISI = 1
    ISP_VER = 2
    ISP_TINY = 3
    ISP_BTN = 18
    ISP_BTC = 19
    ISP_MST = 21

    TINY_NONE = 0
    TINY_ALIVE = 1
    TINY_SUBT_BTC = 6

    BTN_STYLE_CLEAR = 0x80
    MAX_BUTTON_TEXT = 240

    INSIM_VERSION = 9
    ISF_LOCAL = 0x80

    ISI_STRUCT = struct.Struct("<BBBB16sBBHHH")
    VER_STRUCT = struct.Struct("<BBBBH")
    TINY_STRUCT = struct.Struct("<BBBB")
    BTN_HEADER_STRUCT = struct.Struct("<BBBBBBBBBBBB")
    BTC_STRUCT = struct.Struct("<BBBBBBBBHH")
    MST_STRUCT = struct.Struct("<BBBB64s")

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 29999,
        timeout: float = 1.0,
        keepalive_interval: float = 5.0,
        app_name: str = "TNFR-LFS",
        prefix: str = "!",
        request_id: int = 1,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.keepalive_interval = max(0.5, float(keepalive_interval))
        self.app_name = app_name[:16]
        self.prefix = prefix or "!"
        self.request_id = max(0, min(255, int(request_id)))
        self._socket: Optional[socket.socket] = None

    # ------------------------------------------------------------------
    # Context management helpers
    # ------------------------------------------------------------------
    def __enter__(self) -> "InSimClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> None:
        if self._socket is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._send_handshake(sock)
        self._await_version(sock)
        self._socket = sock

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
        finally:
            self._socket = None

    def send_keepalive(self) -> None:
        self._ensure_connected()
        packet = self.TINY_STRUCT.pack(
            self.TINY_STRUCT.size,
            self.ISP_TINY,
            self.request_id,
            self.TINY_NONE,
        )
        self._socket.sendall(packet)  # type: ignore[union-attr]

    def subscribe_controls(self) -> None:
        """Request control packets (button events) from Live for Speed."""

        self._ensure_connected()
        packet = self.TINY_STRUCT.pack(
            self.TINY_STRUCT.size,
            self.ISP_TINY,
            self.request_id,
            self.TINY_SUBT_BTC,
        )
        self._socket.sendall(packet)  # type: ignore[union-attr]

    def poll_button(self, timeout: float = 0.0) -> Optional[ButtonEvent]:
        """Poll for a pending ``IS_BTC`` packet without blocking.

        Returns ``None`` if no packet is available within ``timeout`` seconds
        or if the received packet is not a button click event (``TypeIn`` != 0).
        """

        self._ensure_connected()
        sock = self._socket
        if sock is None:
            raise RuntimeError("InSim client is not connected")
        readable, _, _ = select.select([sock], [], [], timeout)
        if not readable:
            return None

        header = self._recv_exact(sock, 1)
        if not header:
            return None
        size = header[0]
        payload = self._recv_exact(sock, size - 1)
        data = header + payload

        if size != self.BTC_STRUCT.size or data[1] != self.ISP_BTC:
            return None

        (
            _size,
            _type,
            _reqi,
            ucid,
            click_id,
            inst,
            type_in,
            typed,
            unicode_char,
            flags,
        ) = self.BTC_STRUCT.unpack(data)

        if type_in != 0:
            return None

        typed_char: Optional[str]
        if unicode_char:
            typed_char = chr(unicode_char)
        elif typed:
            typed_char = chr(typed)
        else:
            typed_char = None

        return ButtonEvent(
            ucid=ucid,
            click_id=click_id,
            inst=inst,
            type_in=type_in,
            typed_char=typed_char,
            flags=flags,
        )

    def send_button(self, text: str, layout: ButtonLayout | None = None) -> None:
        """Render or update a button overlay using ``IS_BTN`` packets."""

        self._ensure_connected()
        button = (layout or ButtonLayout()).clamp()
        encoded = text.encode("utf8")
        if len(encoded) > self.MAX_BUTTON_TEXT - 1:
            raise ValueError(
                "IS_BTN text exceeds maximum length of "
                f"{self.MAX_BUTTON_TEXT - 1} bytes"
            )
        if not encoded:
            encoded = b""
        packet_text = encoded + b"\0"
        header = self.BTN_HEADER_STRUCT.pack(
            self.BTN_HEADER_STRUCT.size + len(packet_text),
            self.ISP_BTN,
            self.request_id,
            button.ucid,
            button.click_id,
            button.inst,
            button.style,
            button.type_in,
            button.left,
            button.top,
            button.width,
            button.height,
        )
        packet = header + packet_text
        self._socket.sendall(packet)  # type: ignore[union-attr]

    def send_command(self, command: str) -> None:
        """Send a chat command (e.g. ``/press``) to Live for Speed."""

        self._ensure_connected()
        message = (command or "").strip()
        if not message:
            return
        encoded = message.encode("latin1", "ignore")[:63]
        payload = encoded + b"\0" * (64 - len(encoded))
        packet = self.MST_STRUCT.pack(
            self.MST_STRUCT.size,
            self.ISP_MST,
            self.request_id,
            0,
            payload,
        )
        self._socket.sendall(packet)  # type: ignore[union-attr]

    def clear_button(self, layout: ButtonLayout | None = None) -> None:
        """Remove an overlay button using the ``BTN_STYLE_CLEAR`` flag."""

        button = layout or ButtonLayout()
        cleared = ButtonLayout(
            left=button.left,
            top=button.top,
            width=button.width,
            height=button.height,
            ucid=button.ucid,
            inst=button.inst,
            click_id=button.click_id,
            style=self.BTN_STYLE_CLEAR,
            type_in=button.type_in,
        )
        self.send_button("", layout=cleared)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _ensure_connected(self) -> None:
        if self._socket is None:
            raise RuntimeError("InSim client is not connected")

    def _send_handshake(self, sock: socket.socket) -> None:
        padded_name = self.app_name.encode("utf8").ljust(16, b"\0")
        prefix = self.prefix.encode("latin1", "ignore")[:1] or b"!"
        keepalive_ms = int(self.keepalive_interval * 1000)
        packet = self.ISI_STRUCT.pack(
            self.ISI_STRUCT.size,
            self.ISP_ISI,
            self.request_id,
            0,
            padded_name,
            self.INSIM_VERSION,
            prefix[0],
            0,
            self.ISF_LOCAL,
            keepalive_ms,
        )
        sock.sendall(packet)

    def _await_version(self, sock: socket.socket) -> None:
        header = self._recv_exact(sock, 1)
        if not header:
            raise ConnectionError("Empty response while waiting for IS_VER")
        size = header[0]
        payload = self._recv_exact(sock, size - 1)
        data = header + payload
        unpacked = self.VER_STRUCT.unpack(data)
        if unpacked[1] != self.ISP_VER:
            raise ConnectionError("Unexpected packet while waiting for IS_VER")
        if unpacked[4] != self.INSIM_VERSION:
            raise ConnectionError("Unsupported InSim version")

    def _recv_exact(self, sock: socket.socket, count: int) -> bytes:
        data = bytearray()
        while len(data) < count:
            chunk = sock.recv(count - len(data))
            if not chunk:
                raise ConnectionError("Connection closed while receiving data")
            data.extend(chunk)
        return bytes(data)


class OverlayManager:
    """Helper that keeps a button overlay alive via InSim."""

    MAX_BUTTON_TEXT = InSimClient.MAX_BUTTON_TEXT

    def __init__(
        self,
        client: InSimClient,
        *,
        layout: ButtonLayout | None = None,
        title: str = "TNFR × LFS",
    ) -> None:
        self.client = client
        self.layout = (layout or ButtonLayout()).clamp()
        self.title = title
        self._next_keepalive = 0.0
        self._visible = False

    def __enter__(self) -> "OverlayManager":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()

    def connect(self) -> None:
        if not self.client.connected:
            self.client.connect()
        self.client.subscribe_controls()
        self._schedule_keepalive()

    def show(self, lines: Sequence[str] | str) -> None:
        if isinstance(lines, str):
            payload = lines
        else:
            payload = "\n".join(line.strip() for line in lines)
        text = f"{self.title}\n{payload}" if payload else self.title
        self.client.send_button(text, self.layout)
        self._visible = True
        self._schedule_keepalive()

    def hide(self) -> None:
        if not self._visible:
            return
        self.client.clear_button(self.layout)
        self._visible = False
        self._schedule_keepalive()

    def clear(self) -> None:
        """Public alias for :meth:`hide` used by long-running loops."""

        self.hide()

    def tick(self) -> None:
        if not self.client.connected:
            return
        now = monotonic()
        if now >= self._next_keepalive:
            self.client.send_keepalive()
            self._schedule_keepalive(now)

    def poll_button(self, timeout: float = 0.0) -> Optional[ButtonEvent]:
        """Proxy to :meth:`InSimClient.poll_button`."""

        return self.client.poll_button(timeout)

    def close(self) -> None:
        try:
            self.hide()
        finally:
            self.client.close()

    def _schedule_keepalive(self, base: Optional[float] = None) -> None:
        reference = base if base is not None else monotonic()
        self._next_keepalive = reference + self.client.keepalive_interval


@dataclass(slots=True)
class MacroStep:
    """Single command scheduled for future execution."""

    ready_at: float
    command: str


class MacroQueue:
    """Schedule ``/press`` command sequences with spacing safeguards."""

    def __init__(
        self,
        sender: Callable[[str], None],
        *,
        min_interval: float = 0.35,
        time_fn: Callable[[], float] = monotonic,
    ) -> None:
        self._sender = sender
        self._min_interval = max(0.05, float(min_interval))
        self._time_fn: Callable[[], float] = time_fn
        self._steps: Deque[MacroStep] = deque()
        self._last_scheduled = 0.0

    def __len__(self) -> int:
        return len(self._steps)

    def clear(self) -> None:
        self._steps.clear()
        self._last_scheduled = 0.0

    def enqueue_press(self, key: str, *, spacing: float | None = None) -> None:
        self.enqueue_press_sequence([key], spacing=spacing)

    def enqueue_press_sequence(
        self, keys: Sequence[str], *, spacing: float | None = None
    ) -> None:
        if not keys:
            return
        base_spacing = max(self._min_interval, float(spacing) if spacing else self._min_interval)
        now = self._time_fn()
        base_time = max(now, self._last_scheduled)
        for index, key in enumerate(keys):
            command = f"/press {key.strip()}"
            ready_at = base_time + index * base_spacing
            self._steps.append(MacroStep(ready_at=ready_at, command=command))
        self._last_scheduled = base_time + (len(keys) * base_spacing)

    def tick(self) -> int:
        """Dispatch ready commands.

        Returns the number of commands sent during the tick.
        """

        dispatched = 0
        now = self._time_fn()
        while self._steps and self._steps[0].ready_at <= now:
            step = self._steps.popleft()
            self._sender(step.command)
            dispatched += 1
        if not self._steps:
            self._last_scheduled = now
        return dispatched

    def pending(self) -> Sequence[str]:
        return tuple(step.command for step in self._steps)


__all__ = [
    "ButtonLayout",
    "ButtonEvent",
    "InSimClient",
    "MacroQueue",
    "OverlayManager",
]
