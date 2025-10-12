# `tnfr_lfs.ingestion._socket_poll` module
Shared helpers for non-blocking UDP socket polling.

## Functions
- `wait_for_read_ready(sock: socket.socket, *, timeout: float, deadline: Optional[float]) -> bool`
  - Return ``True`` if a socket is ready for reading before ``deadline``.

Parameters
----------
sock:
    The UDP socket registered for non-blocking reads.
timeout:
    Maximum wait duration supplied for the readiness probe.
deadline:
    Optional monotonic timestamp after which waiting should stop.  ``None``
    indicates that the caller should retry immediately without blocking.

