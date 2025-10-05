"""Unit tests for the minimal InSim client."""

from __future__ import annotations

from tnfr_lfs.acquisition.insim import InSimClient


class _DummySocket:
    def __init__(self) -> None:
        self.data = bytearray()

    def sendall(self, payload: bytes) -> None:  # pragma: no cover - trivial passthrough
        self.data.extend(payload)


def test_keepalive_uses_tiny_none_subtype() -> None:
    client = InSimClient()
    dummy = _DummySocket()
    client._socket = dummy  # type: ignore[assignment]

    client.send_keepalive()

    expected = client.TINY_STRUCT.pack(
        client.TINY_STRUCT.size,
        client.ISP_TINY,
        client.request_id,
        client.TINY_NONE,
    )
    assert bytes(dummy.data) == expected
