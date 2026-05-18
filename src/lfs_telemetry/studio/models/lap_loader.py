"""Background lap-loader.

CSV parse + enrichment is fast (~150 ms cold, ~25 ms warm-disk-cached)
but still long enough to stutter the UI thread on a click. We push every
load to a :class:`QThreadPool` worker and emit ``lap_loaded`` from the
GUI thread when the result is ready. The worker re-uses
:meth:`WorkspaceState.load_lap` so the in-memory and on-disk caches are
shared with the rest of the app.

Errors are caught and emitted on a sibling signal so the status bar can
display them without crashing the GUI.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class _LoaderSignals(QObject):
    lap_loaded = Signal(Path, object)   # (path, LapTelemetry)
    lap_failed = Signal(Path, str)      # (path, error message)


class _LoaderTask(QRunnable):
    """One-shot loader run on a thread-pool worker."""

    def __init__(
        self,
        loader: LapLoader,
        path: Path,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._loader = loader
        self._path = path

    @Slot()
    def run(self) -> None:  # type: ignore[override]
        try:
            lap = self._loader.workspace.load_lap(self._path)
        except Exception as exc:  # broad on purpose: any exc → status bar
            self._loader.signals.lap_failed.emit(self._path, str(exc))
            return
        self._loader.signals.lap_loaded.emit(self._path, lap)


class LapLoader(QObject):
    """Submit lap-load jobs to a background pool and broadcast results.

    A small dedicated pool (default 2 workers) keeps loads concurrent
    enough to overlap I/O while preventing storms when the user clicks
    through 25 captures in rapid succession.
    """

    def __init__(self, workspace, max_workers: int = 2, parent=None) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self.signals = _LoaderSignals(self)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(1, int(max_workers)))

    # Public signals (forwarded for ergonomic ``loader.lap_loaded.connect(...)``).
    @property
    def lap_loaded(self) -> Signal:
        return self.signals.lap_loaded

    @property
    def lap_failed(self) -> Signal:
        return self.signals.lap_failed

    def request(self, path: str | Path) -> None:
        """Queue a lap load. Idempotent re-requests are coalesced by the cache."""
        self._pool.start(_LoaderTask(self, Path(path)))

    def request_many(self, paths) -> None:
        for p in paths:
            self.request(p)

    def shutdown(self, wait_ms: int = 2000) -> None:
        """Drain the pool on app close (best effort)."""
        self._pool.clear()
        self._pool.waitForDone(int(wait_ms))


__all__ = ["LapLoader"]
