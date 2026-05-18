"""Live integration tests against a running LFS instance.

These tests are skipped by default. To enable them:

  1. Launch Live for Speed and configure ``cfg.txt`` with::

         OutSim Mode 2
         OutSim Opts 1ff
         OutSim Delay 1
         OutSim IP 127.0.0.1
         OutSim Port 30000

         OutGauge Mode 1
         OutGauge Delay 1
         OutGauge IP 127.0.0.1
         OutGauge Port 30001

     Then enable InSim at runtime (no cfg.txt key exists for this)::

         /insim 29999

     or launch LFS with ``LFS.exe /insim=29999``.

  2. Enter a session (single player or join an InSim host with admin).
  3. Run::

         set LFS_TELEMETRY_LIVE_TEST=1
         pytest tests/test_live_lfs_integration.py -v

Optional env vars:
  LFS_TELEMETRY_HOST           — InSim host (default 127.0.0.1)
  LFS_TELEMETRY_INSIM_PORT     — default 29999
  LFS_TELEMETRY_OUTSIM_PORT    — default 30000
  LFS_TELEMETRY_OUTGAUGE_PORT  — default 30001
  LFS_TELEMETRY_ADMIN          — admin password if InSim host requires it
  LFS_TELEMETRY_TIMEOUT        — seconds to wait for first packet (default 10)
"""

from __future__ import annotations

import asyncio
import os

import pytest

from lfs_telemetry.telemetry.live import LiveTelemetry
from lfs_telemetry.telemetry.protocol.insim import InSimClient
from lfs_telemetry.telemetry.protocol.packets import (
    OSO_ALL,
    InSimNewPlayer,
    InSimState,
    InSimVersion,
    outsim2_size,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("LFS_TELEMETRY_LIVE_TEST"),
    reason="set LFS_TELEMETRY_LIVE_TEST=1 with LFS running locally to enable",
)


def _host() -> str:
    return os.getenv("LFS_TELEMETRY_HOST", "127.0.0.1")


def _insim_port() -> int:
    return int(os.getenv("LFS_TELEMETRY_INSIM_PORT", "29999"))


def _outsim_port() -> int:
    return int(os.getenv("LFS_TELEMETRY_OUTSIM_PORT", "30000"))


def _outgauge_port() -> int:
    return int(os.getenv("LFS_TELEMETRY_OUTGAUGE_PORT", "30001"))


def _timeout() -> float:
    return float(os.getenv("LFS_TELEMETRY_TIMEOUT", "10"))


@pytest.mark.asyncio
async def test_live_outsim_outgauge_fused_sample_arrives() -> None:
    """We should receive at least one fused sample (OutSim + OutGauge)."""
    timeout = _timeout()
    async with LiveTelemetry(
        outsim_port=_outsim_port(),
        outgauge_port=_outgauge_port(),
        outsim_opts=OSO_ALL,
    ) as live:
        async def _first() -> object:
            async for sample in live.samples():
                if sample.is_complete:
                    return sample
            return None
        sample = await asyncio.wait_for(_first(), timeout=timeout)
    assert sample is not None
    assert sample.is_complete
    assert sample.outsim is not None
    assert sample.outgauge is not None
    assert sample.outgauge.car  # car name string non-empty


@pytest.mark.asyncio
async def test_live_outsim_pack2_carries_wheels() -> None:
    """When OutSim Opts=0x1ff, every packet must be 280 bytes with 4 wheels."""
    expected = outsim2_size(OSO_ALL)
    assert expected == 280
    timeout = _timeout()
    async with LiveTelemetry(
        outsim_port=_outsim_port(),
        outgauge_port=_outgauge_port(),
        outsim_opts=OSO_ALL,
    ) as live:
        async def _first_with_wheels() -> object:
            async for sample in live.samples():
                if sample.outsim2 is not None and sample.outsim2.wheels:
                    return sample
            return None
        sample = await asyncio.wait_for(
            _first_with_wheels(), timeout=timeout)
    assert sample is not None
    pkt2 = sample.outsim2
    assert pkt2 is not None
    assert pkt2.wheels is not None
    assert len(pkt2.wheels) == 4
    # At least one wheel should be on the ground while moving.
    touching = [w.touching for w in pkt2.wheels]
    assert any(t for t in touching), \
        f"no wheel reports touching road (got {touching})"


@pytest.mark.asyncio
async def test_live_insim_handshake_returns_version() -> None:
    """InSim should reply to ISI with IS_VER and version >= 9."""
    timeout = _timeout()
    client = InSimClient(
        host=_host(),
        port=_insim_port(),
        admin_password=os.getenv("LFS_TELEMETRY_ADMIN", ""),
        request_hlv=False,
        request_mci=False,
    )
    await client.start()
    try:
        async def _wait_for_ver() -> InSimVersion | None:
            async for evt in client.events():
                if isinstance(evt, InSimVersion):
                    return evt
            return None
        ver = await asyncio.wait_for(_wait_for_ver(), timeout=timeout)
    finally:
        await client.stop()
    assert ver is not None
    assert ver.insim_version >= 9, \
        f"InSim version {ver.insim_version} < 9 (server too old)"
    assert ver.product  # non-empty product name


@pytest.mark.asyncio
async def test_live_insim_state_populates_track() -> None:
    """The IS_STA snapshot should populate RaceContext.track."""
    timeout = _timeout()
    client = InSimClient(
        host=_host(),
        port=_insim_port(),
        admin_password=os.getenv("LFS_TELEMETRY_ADMIN", ""),
        request_hlv=False,
        request_mci=False,
    )
    await client.start()
    try:
        async def _wait_for_state() -> InSimState | None:
            async for evt in client.events():
                if isinstance(evt, InSimState):
                    return evt
            return None
        sta = await asyncio.wait_for(_wait_for_state(), timeout=timeout)
    finally:
        await client.stop()
    assert sta is not None
    assert sta.track  # e.g. "BL1"
    assert client.context.track == sta.track


@pytest.mark.asyncio
async def test_live_insim_player_event_arrives() -> None:
    """Joining an active session should yield at least one IS_NPL within the
    timeout window. Skip if the local session has no players yet."""
    timeout = _timeout()
    client = InSimClient(
        host=_host(),
        port=_insim_port(),
        admin_password=os.getenv("LFS_TELEMETRY_ADMIN", ""),
        request_hlv=False,
        request_mci=False,
    )
    await client.start()
    try:
        async def _wait_for_npl() -> InSimNewPlayer | None:
            async for evt in client.events():
                if isinstance(evt, InSimNewPlayer):
                    return evt
            return None
        try:
            npl = await asyncio.wait_for(_wait_for_npl(), timeout=timeout)
        except TimeoutError:
            pytest.skip("no IS_NPL within timeout (no players in session?)")
    finally:
        await client.stop()
    assert npl is not None
    assert npl.player_name  # e.g. "Driver"
    assert npl.car_name     # e.g. "FOX"


@pytest.mark.asyncio
async def test_live_full_pipeline_with_insim_attaches_context() -> None:
    """End-to-end: LiveTelemetry + InSim should attach race_context to
    samples after the IS_STA snapshot arrives."""
    timeout = _timeout() * 2
    async with LiveTelemetry(
        outsim_port=_outsim_port(),
        outgauge_port=_outgauge_port(),
        outsim_opts=OSO_ALL,
        insim_host=_host(),
        insim_port=_insim_port(),
        insim_admin_password=os.getenv("LFS_TELEMETRY_ADMIN", ""),
    ) as live:
        async def _first_with_ctx() -> object:
            async for sample in live.samples():
                if (
                    sample.is_complete
                    and sample.race_context is not None
                    and sample.race_context.track
                ):
                    return sample
            return None
        sample = await asyncio.wait_for(_first_with_ctx(), timeout=timeout)
    assert sample is not None
    assert sample.race_context is not None
    assert sample.race_context.track
