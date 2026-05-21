"""Shared CLI helpers: flag parsing, stream hardening, signal handler."""
from __future__ import annotations

import argparse
import contextlib
import io
import sys

from ..telemetry.constants import (
    INSIM_DEFAULT_PORT,
    OUTGAUGE_DEFAULT_PORT,
    OUTSIM_DEFAULT_PORT,
)
from ..telemetry.protocol.packets import OSO_ALL
from . import _state


def _request_stop(*_args) -> None:
    _state.STOP_REQUESTED = True
    print("[capture] stop requested, flushing…", file=sys.stderr)
    # Wake up the asyncio loop even if no UDP samples are arriving:
    # cancel the awaited operation so the `async for` raises and the
    # `except BaseException` path runs cleanly.
    loop = _state.CAPTURE_LOOP
    task = _state.CAPTURE_TASK
    if loop is not None and task is not None and not task.done():
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(task.cancel)


def _add_lfs_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--outsim-port", type=int, default=OUTSIM_DEFAULT_PORT,
    )
    parser.add_argument(
        "--outgauge-port", type=int, default=OUTGAUGE_DEFAULT_PORT,
    )
    parser.add_argument(
        "--outsim-opts", type=lambda s: int(s, 0), default=OSO_ALL,
        help="OutSim Opts hex flags (default 0x1ff = full extended packets)")
    parser.add_argument(
        "--insim-host", default=None,
        help="enable InSim TCP client by host (e.g. 127.0.0.1)")
    parser.add_argument(
        "--insim-port", type=int, default=INSIM_DEFAULT_PORT,
    )
    parser.add_argument("--insim-admin", default="",
                        help="LFS admin password (if required)")
    parser.add_argument(
        "--car", default=None,
        help="LFS car short name (FOX, FO8, BF1, MRT). Overrides auto-detect.")


class _ResilientTextStream(io.TextIOBase):
    """Wrap a text stream so writes never raise.

    PyInstaller windowed bundles attach ``sys.stdout`` / ``sys.stderr``
    to a special handle that can fail with ``OSError [Errno 22] Invalid
    argument`` after long-running sessions or large cumulative writes
    (known PyInstaller + Windows windowed-mode issue). The capture
    subprocess logs a lot of diagnostics through these streams, so a
    single failed write must not crash the loop and lose the in-flight
    capture. We swallow OSError / ValueError on write, flush and close.
    """

    def __init__(self, inner: io.TextIOBase | None) -> None:
        self._inner = inner

    def writable(self) -> bool:  # type: ignore[override]
        return True

    def write(self, s: str) -> int:  # type: ignore[override]
        if self._inner is None:
            return len(s)
        try:
            return self._inner.write(s)
        except (OSError, ValueError):
            return len(s)

    def flush(self) -> None:  # type: ignore[override]
        if self._inner is None:
            return
        with contextlib.suppress(OSError, ValueError):
            self._inner.flush()

    def isatty(self) -> bool:  # type: ignore[override]
        if self._inner is None:
            return False
        try:
            return bool(self._inner.isatty())
        except (OSError, ValueError):
            return False



def _harden_std_streams() -> None:
    """Replace ``sys.stdout`` / ``sys.stderr`` with resilient wrappers.

    Only acts when the bundled Studio launches the CLI as a child
    process (frozen build on Windows). Idempotent: re-wrapping an
    already-resilient stream is a no-op.
    """
    if not getattr(sys, "frozen", False):
        return
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if isinstance(stream, _ResilientTextStream):
            continue
        setattr(sys, name, _ResilientTextStream(stream))
