"""Shared helpers for non-blocking UDP socket polling."""

from __future__ import annotations

import select
import socket
import time
from typing import Optional

__all__ = ["wait_for_read_ready"]


def wait_for_read_ready(
    sock: socket.socket,
    *,
    timeout: float,
    deadline: Optional[float],
) -> bool:
    """Return ``True`` if a socket is ready for reading before ``deadline``.

    Parameters
    ----------
    sock:
        The UDP socket registered for non-blocking reads.
    timeout:
        Maximum wait duration supplied for the readiness probe.
    deadline:
        Optional monotonic timestamp after which waiting should stop.  ``None``
        indicates that the caller should retry immediately without blocking.
    """

    if timeout <= 0.0:
        # No waiting was requested; signal the caller to retry immediately.
        return True

    wait_time = timeout
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False
        wait_time = min(timeout, remaining)

    try:
        readable, _, _ = select.select([sock], [], [], wait_time)
    except (OSError, ValueError):
        return False
    return bool(readable)
