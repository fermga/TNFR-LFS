"""Workspace state for the MoTeC-style viewer.

Wraps :func:`discover_captures` with a lap-cache so the Dash callbacks
don't reload the same CSV on every interaction. All methods are
lock-free and intended to be used from a single Dash worker process.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..telemetry import (
    CaptureInfo,
    LapTelemetry,
    StintTelemetry,
    discover_captures,
)


@dataclass
class WorkspaceState:
    """In-memory view of a captures folder + lap cache."""

    workspace: Path
    pattern: str = "*.csv"
    recursive: bool = True
    _captures: list[CaptureInfo] = field(default_factory=list)
    _lap_cache: dict[Path, LapTelemetry] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------

    def refresh(self) -> list[CaptureInfo]:
        """Re-scan the workspace and return the catalog."""
        with self._lock:
            self._captures = discover_captures(
                self.workspace, self.pattern, recursive=self.recursive
            )
            return list(self._captures)

    @property
    def captures(self) -> list[CaptureInfo]:
        if not self._captures:
            return self.refresh()
        return list(self._captures)

    def find(self, path: str | Path) -> CaptureInfo | None:
        target = Path(path).resolve()
        for info in self.captures:
            if Path(info.path).resolve() == target:
                return info
        return None

    # ------------------------------------------------------------------
    # Lap loading (cached)
    # ------------------------------------------------------------------

    def load_lap(self, path: str | Path) -> LapTelemetry:
        """Return :class:`LapTelemetry` for ``path``, caching the result."""
        key = Path(path).resolve()
        with self._lock:
            cached = self._lap_cache.get(key)
            if cached is not None:
                return cached
        lap = LapTelemetry.from_csv(key)
        with self._lock:
            self._lap_cache[key] = lap
        return lap

    def load_laps(self, paths: Iterable[str | Path]) -> list[LapTelemetry]:
        return [self.load_lap(p) for p in paths]

    def stint(self, paths: Iterable[str | Path]) -> StintTelemetry:
        return StintTelemetry.from_laps(self.load_laps(paths))

    def clear_cache(self) -> None:
        """Drop both the catalog and the lap cache.

        Called when the user switches workspace folder so that the
        next access re-scans from disk.
        """
        with self._lock:
            self._lap_cache.clear()
            self._captures = []

    # ------------------------------------------------------------------
    # Catalog rendering helpers (used by callbacks)
    # ------------------------------------------------------------------

    def catalog_rows(self) -> list[dict]:
        """Rows for the captures DataTable."""
        rows: list[dict] = []
        for info in self.captures:
            rows.append({
                "path": str(info.path),
                "file": Path(info.path).name,
                "car": info.car or "",
                "track": info.track or "",
                "samples": info.samples,
                "lap_time_s": (
                    round(info.lap_time_s, 3) if info.lap_time_s is not None else None
                ),
                "distance_m": (
                    round(info.distance_m, 1) if info.distance_m is not None else None
                ),
                "size_kb": round(info.file_size_bytes / 1024.0, 1),
            })
        return rows
