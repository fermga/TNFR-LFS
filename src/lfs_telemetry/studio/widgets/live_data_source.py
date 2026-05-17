"""Central poller for ``live.json``.

A single :class:`LiveDataSource` is shared by every overlay module so we
read the JSON file once per tick and broadcast the snapshot via Qt
signals. This keeps the radar, delta bar, lap-info and gear/RPM modules
fully independent — each one can be toggled on/off and dragged to its
own screen position without affecting the others.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal


class LiveDataSource(QObject):
    """Polls ``live.json`` at a fixed interval and emits the parsed dict."""

    snapshot_changed = Signal(dict)
    """Emitted on every successful read with the full snapshot dict."""

    available_changed = Signal(bool)
    """Emitted when the file appears or disappears."""

    def __init__(self, interval_ms: int = 100, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._path: Path | None = None
        self._available = False
        self._last_snap: dict[str, Any] = {}
        # Cache (mtime_ns, size) of the last successful read so we can
        # skip the JSON parse when the file has not changed between
        # ticks (typical case: LFS not writing new samples).
        self._last_stat: tuple[int, int] | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def snapshot(self) -> dict[str, Any]:
        return self._last_snap

    def set_path(self, path: Path | None) -> None:
        new_path = Path(path) if path else None
        if new_path == self._path:
            return
        self._path = new_path
        self._last_snap = {}
        self._last_stat = None
        self.snapshot_changed.emit(self._last_snap)
        self._update_available(False)

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        self._last_stat = None
        self._update_available(False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_available(self, value: bool) -> None:
        if value != self._available:
            self._available = value
            self.available_changed.emit(value)

    def _tick(self) -> None:
        path = self._path
        if path is None:
            self._update_available(False)
            return
        try:
            st = path.stat()
        except OSError:
            self._last_stat = None
            self._update_available(False)
            return
        stat_key = (st.st_mtime_ns, st.st_size)
        if stat_key == self._last_stat:
            # File untouched since last successful read; nothing to do.
            self._update_available(True)
            return
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            self._update_available(False)
            return
        if not text.strip():
            return
        try:
            snap = json.loads(text)
        except ValueError:
            return
        if not isinstance(snap, dict):
            return
        self._last_stat = stat_key
        self._last_snap = snap
        self._update_available(True)
        self.snapshot_changed.emit(snap)


__all__ = ["LiveDataSource"]
