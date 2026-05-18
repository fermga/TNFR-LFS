"""Async InSim TCP client.

Connects to an LFS instance over TCP, sends an IS_ISI handshake, dispatches
incoming packets to typed dataclasses and accumulates a thread-safe
:class:`RaceContext` snapshot that the rest of the pipeline can consume.

Usage::

    async with InSimClient(host="127.0.0.1", port=29999) as client:
        async for event in client.events():
            if isinstance(event, InSimNewPlayer):
                print(client.context.snapshot())

LFS InSim notes (v9):

* Each TCP packet starts with a single ``Size`` byte (in bytes since v9).
* Connections close if no packet is received within ~70 s; we keep the link
  alive by sending an empty ``IS_TINY`` (TINY_NONE) every 30 s.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .packets import (
    FUEL_SCALE,
    ISF_CON,
    ISF_HLV,
    ISF_LOCAL,
    ISF_MCI,
    ISF_NLP,
    ISF_OBH,
    ISP_CCH,
    ISP_CIM,
    ISP_CNL,
    ISP_CON,
    ISP_CSC,
    ISP_FIN,
    ISP_FLG,
    ISP_HLV,
    ISP_LAP,
    ISP_MAL,
    ISP_MCI,
    ISP_NCN,
    ISP_NLP,
    ISP_NPL,
    ISP_OBH,
    ISP_PEN,
    ISP_PFL,
    ISP_PIT,
    ISP_PLA,
    ISP_PLL,
    ISP_PLP,
    ISP_PSF,
    ISP_RES,
    ISP_RST,
    ISP_SLC,
    ISP_SMALL,
    ISP_SPX,
    ISP_STA,
    ISP_TOC,
    ISP_VER,
    TINY_NCN,
    TINY_NONE,
    TINY_NPL,
    TINY_SST,
    InSimCameraChange,
    InSimCarContact,
    InSimCarStateChanged,
    InSimConnectionLeft,
    InSimFinish,
    InSimFlag,
    InSimHotLapValid,
    InSimInterfaceMode,
    InSimLap,
    InSimMCI,
    InSimModsAllowed,
    InSimNewConnection,
    InSimNewPlayer,
    InSimNodeLap,
    InSimObjectHit,
    InSimPenalty,
    InSimPit,
    InSimPitLane,
    InSimPitStopFinish,
    InSimPlayerFlags,
    InSimPlayerLeaves,
    InSimPlayerTelepit,
    InSimRaceStart,
    InSimResult,
    InSimSelectedCar,
    InSimSmall,
    InSimSplit,
    InSimState,
    InSimTakeOverCar,
    InSimVersion,
    InSimVoteAction,
    build_isi_packet,
    build_msl_packet,
    build_tiny_packet,
    decode_host_flags,
    decode_pit_work,
    penalty_name,
)


@dataclass(slots=True)
class PitStopRecord:
    """A fused pit-stop record combining IS_PIT (start) and IS_PSF (end).

    LFS sends one IS_PIT when the driver enters the pit box (with the work
    request, fuel addition and post-stop tyre compounds) and one IS_PSF when
    the stop completes (with the wall-clock duration). Keeping them in a
    single dataclass per PLID makes endurance/strategy code trivial.
    """

    player_id: int
    laps_done: int
    fuel_add: int                                # 0..255 (raw IS_PIT field)
    penalty: int                                 # PENALTY_* served by the stop
    num_stops: int                               # cumulative stops so far
    tyres: tuple[int, int, int, int]             # compounds after the stop
    work: int                                    # PSE_* bitfield
    work_labels: tuple[str, ...]                 # decode_pit_work(work)
    flags: int                                   # PIF_* snapshot at IS_PIT
    stop_time_ms: int | None = None              # filled when IS_PSF arrives

    @property
    def completed(self) -> bool:
        return self.stop_time_ms is not None


_LOG = logging.getLogger(__name__)
_KEEPALIVE_INTERVAL_S = 30.0
# Period at which we re-request IS_STA via TINY_SST. LFS does send IS_STA
# spontaneously on state changes (track / weather / race start) but a
# dropped packet or a session change while we were briefly disconnected
# would otherwise leave ``RaceContext.track`` stale forever, which is
# exactly the symptom users see when the radar / racing line keep
# showing the previous track. A small periodic refresh is the simplest
# robust fix and matches what helicorsa / Detect&Monitor do.
_STATE_REFRESH_INTERVAL_S = 5.0


@dataclass(slots=True)
class RaceContext:
    """Rolling snapshot of race-level state derived from InSim packets."""

    lfs_version: str | None = None
    insim_version: int | None = None
    track: str | None = None
    weather: int | None = None
    wind: int | None = None
    race_laps: int | None = None
    qual_minutes: int | None = None
    race_in_progress: int | None = None
    view_player_id: int | None = None
    # Per-PLID player info, populated from IS_NPL.
    players: dict[int, InSimNewPlayer] = field(default_factory=dict)
    # Last lap / split for each PLID.
    last_lap_ms: dict[int, int] = field(default_factory=dict)
    last_split_ms: dict[int, dict[int, int]] = field(default_factory=dict)
    # Latest hot-lap-valid violation per PLID.
    last_hlv: dict[int, InSimHotLapValid] = field(default_factory=dict)
    # Number of completed laps per PLID (incremented on every IS_LAP).
    lap_count: dict[int, int] = field(default_factory=dict)
    # Per-PLID list of completed lap times (ms).
    lap_times_ms: dict[int, list[int]] = field(default_factory=dict)
    # Per-PLID list of {split: split_time_ms} dicts, in order of completion.
    split_times_ms: dict[int, list[dict[int, int]]] = field(default_factory=dict)
    # Per-PLID buffered splits being built up before the next IS_LAP arrives.
    _pending_splits: dict[int, dict[int, int]] = field(default_factory=dict)
    # Per-PLID list of fuel% at end of each lap (from IS_LAP fuel200/FUEL_SCALE).
    lap_fuel_pct: dict[int, list[float]] = field(default_factory=dict)
    # Last MCI snapshot (multi-car traffic).
    last_mci: InSimMCI | None = None
    # Last NLP snapshot (compact node/lap for every car).
    last_nlp: InSimNodeLap | None = None
    # Rolling list of HLV (hot-lap-valid lost) events with full payload.
    hlv_events: list[InSimHotLapValid] = field(default_factory=list)
    # Rolling list of OBH (object hit) events.
    obh_events: list[InSimObjectHit] = field(default_factory=list)
    # Rolling list of pit stops (IS_PIT fused with IS_PSF).
    pit_stops: list[PitStopRecord] = field(default_factory=list)
    # Per-PLID pending IS_PIT awaiting its matching IS_PSF.
    _pending_pits: dict[int, PitStopRecord] = field(default_factory=dict)
    # Connections on the host keyed by UCID (populated from IS_NCN/IS_CNL).
    connections: dict[int, InSimNewConnection] = field(default_factory=dict)
    # Car currently selected by each connection (UCID -> car id), from IS_SLC.
    # ``""`` means the connection has no car selected.
    selected_cars: dict[int, str] = field(default_factory=dict)
    # Latest mods-allowed list from the host (None = unknown / unrestricted).
    allowed_mods: tuple[str, ...] | None = None
    # Latest pit-lane fact per PLID (PITLANE_EXIT / ENTER / DT / SG / ...).
    pit_lane: dict[int, int] = field(default_factory=dict)
    # Last vote action seen on the host (None = no vote pending / cleared).
    last_vote_action: int | None = None
    # ---- additions for IS_CON/FIN/RES/TOC/PEN/FLG/PFL/PLP/PLL/CCH/CIM/CSC ----
    # Host race-rule flags from IS_RST (HOSTF_* bitfield).
    host_flags: int | None = None
    # Rolling log of car-to-car contacts (IS_CON).
    car_contacts: list[InSimCarContact] = field(default_factory=list)
    # Per-PLID provisional finish (IS_FIN, before result is confirmed).
    finishes: dict[int, InSimFinish] = field(default_factory=dict)
    # Per-PLID confirmed race result (IS_RES).
    results: dict[int, InSimResult] = field(default_factory=dict)
    # Driver-swap log (IS_TOC) — critical for endurance multi-driver stints.
    driver_swaps: list[InSimTakeOverCar] = field(default_factory=list)
    # Per-PLID penalty log; the dict is the *current* penalty per PLID and
    # ``penalty_events`` is the chronological audit trail.
    penalties: dict[int, int] = field(default_factory=dict)
    penalty_events: list[InSimPenalty] = field(default_factory=list)
    # Per-PLID active flags (yellow/blue), plus chronological log.
    flag_state: dict[int, dict[int, int]] = field(default_factory=dict)
    flag_events: list[InSimFlag] = field(default_factory=list)
    # Latest PIF_* flags per PLID, updated by IS_NPL and IS_PFL.
    player_flags: dict[int, int] = field(default_factory=dict)
    # Players currently in pits via telepit (IS_PLP). IS_PLA enter/exit and
    # IS_PIT/PSF live in separate sets.
    tele_pits: set[int] = field(default_factory=set)
    # PLIDs that left the race (IS_PLL) since last RST.
    departures: list[int] = field(default_factory=list)
    # Latest in-game camera per PLID (IS_CCH); useful to suspend telemetry
    # when the driver leaves cockpit view.
    camera: dict[int, int] = field(default_factory=dict)
    # Latest interface mode per UCID (IS_CIM) — (mode, sub_mode, sel_type).
    interface_mode: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    # Engine start/stop log (IS_CSC). Each entry: (player_id, action, time_ms).
    engine_events: list[tuple[int, int, int]] = field(default_factory=list)

    def update(self, packet: object) -> None:
        if isinstance(packet, InSimVersion):
            self.lfs_version = packet.version
            self.insim_version = packet.insim_ver
        elif isinstance(packet, InSimState):
            self.race_in_progress = packet.race_in_progress
            self.qual_minutes = packet.qual_minutes
            self.race_laps = packet.race_laps
            self.track = packet.track
            self.weather = packet.weather
            self.wind = packet.wind
            self.view_player_id = packet.view_plid
        elif isinstance(packet, InSimRaceStart):
            self.race_laps = packet.race_laps
            self.qual_minutes = packet.qual_minutes
            self.track = packet.track
            self.weather = packet.weather
            self.wind = packet.wind
            self.host_flags = packet.flags
            # IS_RST means the race was (re)started — typically via
            # /restart, /track, exit-to-pits-and-rejoin, or moving to a
            # different host. Per-driver lap counters and timing state
            # from the previous run are no longer valid; if we keep
            # them around the capture loop's ``cur_lap > last_lap_count``
            # check stops firing (because the new lap 1 is < the old
            # last lap count) and laps stop being recorded. Clearing
            # everything here is the same approach used by
            # Detect&Monitor and matches LFS' own behaviour.
            self.last_lap_ms.clear()
            self.last_split_ms.clear()
            self.lap_count.clear()
            self.lap_times_ms.clear()
            self.split_times_ms.clear()
            self._pending_splits.clear()
            self.lap_fuel_pct.clear()
            self.last_hlv.clear()
            self.hlv_events.clear()
            self.obh_events.clear()
            self.pit_stops.clear()
            self._pending_pits.clear()
            self.car_contacts.clear()
            self.finishes.clear()
            self.results.clear()
            self.driver_swaps.clear()
            self.penalties.clear()
            self.penalty_events.clear()
            self.flag_state.clear()
            self.flag_events.clear()
            self.tele_pits.clear()
            self.departures.clear()
            self.engine_events.clear()
        elif isinstance(packet, InSimNewPlayer):
            self.players[packet.player_id] = packet
            self.player_flags[packet.player_id] = packet.flags
        elif isinstance(packet, InSimLap):
            self.last_lap_ms[packet.player_id] = packet.lap_time_ms
            self.lap_count[packet.player_id] = (
                self.lap_count.get(packet.player_id, 0) + 1
            )
            self.lap_times_ms.setdefault(packet.player_id, []).append(
                packet.lap_time_ms)
            self.lap_fuel_pct.setdefault(packet.player_id, []).append(
                packet.fuel200 / FUEL_SCALE)
            pending = self._pending_splits.pop(packet.player_id, {})
            self.split_times_ms.setdefault(packet.player_id, []).append(pending)
        elif isinstance(packet, InSimSplit):
            self.last_split_ms.setdefault(packet.player_id, {})[packet.split] = (
                packet.split_time_ms
            )
            self._pending_splits.setdefault(packet.player_id, {})[packet.split] = (
                packet.split_time_ms
            )
        elif isinstance(packet, InSimHotLapValid):
            self.last_hlv[packet.player_id] = packet
            self.hlv_events.append(packet)
        elif isinstance(packet, InSimObjectHit):
            self.obh_events.append(packet)
        elif isinstance(packet, InSimPit):
            rec = PitStopRecord(
                player_id=packet.player_id,
                laps_done=packet.laps_done,
                fuel_add=packet.fuel_add,
                penalty=packet.penalty,
                num_stops=packet.num_stops,
                tyres=packet.tyres,
                work=packet.work,
                work_labels=tuple(decode_pit_work(packet.work)),
                flags=packet.flags,
            )
            self._pending_pits[packet.player_id] = rec
            self.pit_stops.append(rec)
        elif isinstance(packet, InSimPitStopFinish):
            rec = self._pending_pits.pop(packet.player_id, None)
            if rec is not None:
                rec.stop_time_ms = packet.stop_time_ms
            else:
                # No matching IS_PIT seen (e.g. we joined mid-stop) — still
                # record the duration so callers can count stops correctly.
                self.pit_stops.append(PitStopRecord(
                    player_id=packet.player_id, laps_done=0,
                    fuel_add=0, penalty=0, num_stops=0,
                    tyres=(0, 0, 0, 0), work=0,
                    work_labels=(), flags=0,
                    stop_time_ms=packet.stop_time_ms,
                ))
        elif isinstance(packet, InSimMCI):
            self.last_mci = packet
        elif isinstance(packet, InSimNodeLap):
            self.last_nlp = packet
        elif isinstance(packet, InSimNewConnection):
            self.connections[packet.connection_id] = packet
        elif isinstance(packet, InSimConnectionLeft):
            self.connections.pop(packet.connection_id, None)
            self.selected_cars.pop(packet.connection_id, None)
        elif isinstance(packet, InSimSelectedCar):
            self.selected_cars[packet.connection_id] = packet.car_name
        elif isinstance(packet, InSimModsAllowed):
            self.allowed_mods = packet.mod_ids
        elif isinstance(packet, InSimPitLane):
            self.pit_lane[packet.player_id] = packet.fact
        elif isinstance(packet, InSimVoteAction):
            self.last_vote_action = packet.action
        elif isinstance(packet, InSimCarContact):
            self.car_contacts.append(packet)
        elif isinstance(packet, InSimFinish):
            self.finishes[packet.player_id] = packet
        elif isinstance(packet, InSimResult):
            self.results[packet.player_id] = packet
        elif isinstance(packet, InSimTakeOverCar):
            self.driver_swaps.append(packet)
        elif isinstance(packet, InSimPenalty):
            self.penalties[packet.player_id] = packet.new_penalty
            self.penalty_events.append(packet)
        elif isinstance(packet, InSimFlag):
            self.flag_events.append(packet)
            slot = self.flag_state.setdefault(packet.player_id, {})
            if packet.off_on:
                slot[packet.flag] = packet.car_behind
            else:
                slot.pop(packet.flag, None)
        elif isinstance(packet, InSimPlayerFlags):
            self.player_flags[packet.player_id] = packet.flags
        elif isinstance(packet, InSimPlayerTelepit):
            self.tele_pits.add(packet.player_id)
        elif isinstance(packet, InSimPlayerLeaves):
            self.departures.append(packet.player_id)
            # IS_PLL frees the slot — drop transient per-PLID state but keep
            # accumulated lap/result history for post-race analysis.
            self.tele_pits.discard(packet.player_id)
            self.player_flags.pop(packet.player_id, None)
            self.camera.pop(packet.player_id, None)
        elif isinstance(packet, InSimCameraChange):
            self.camera[packet.player_id] = packet.camera
        elif isinstance(packet, InSimInterfaceMode):
            self.interface_mode[packet.connection_id] = (
                packet.mode, packet.sub_mode, packet.sel_type,
            )
        elif isinstance(packet, InSimCarStateChanged):
            self.engine_events.append(
                (packet.player_id, packet.action, packet.time_ms))

    def snapshot(self) -> dict[str, object]:
        return {
            "lfs_version": self.lfs_version,
            "insim_version": self.insim_version,
            "track": self.track,
            "weather": self.weather,
            "wind": self.wind,
            "race_laps": self.race_laps,
            "qual_minutes": self.qual_minutes,
            "race_in_progress": self.race_in_progress,
            "view_player_id": self.view_player_id,
            "num_players": len(self.players),
            "view_player": (
                self.players.get(self.view_player_id).car_name
                if self.view_player_id in self.players else None
            ),
            "view_lap_count": self.lap_count.get(self.view_player_id, 0),
            "view_last_lap_ms": self.last_lap_ms.get(self.view_player_id),
            "host_flags": self.host_flags,
            "num_results": len(self.results),
            "num_finishes": len(self.finishes),
            "num_driver_swaps": len(self.driver_swaps),
            "num_penalty_events": len(self.penalty_events),
            "num_flag_events": len(self.flag_events),
            "num_car_contacts": len(self.car_contacts),
            "num_engine_events": len(self.engine_events),
            "view_camera": self.camera.get(self.view_player_id),
            "view_penalty": self.penalties.get(self.view_player_id),
            "view_player_flags": self.player_flags.get(self.view_player_id),
            "view_in_tele_pit": self.view_player_id in self.tele_pits,
            "host_flags_decoded": (
                decode_host_flags(self.host_flags)
                if self.host_flags is not None else []
            ),
            "view_penalty_name": penalty_name(
                self.penalties.get(self.view_player_id, 0)),
            "view_last_pit_work": (
                list(self.pit_stops[-1].work_labels)
                if self.pit_stops
                and self.pit_stops[-1].player_id == self.view_player_id
                else []
            ),
            "num_pit_stops": len(self.pit_stops),
        }


# Map packet type → (parser, fixed_size_or_None).
_PACKET_PARSERS: dict[int, callable] = {
    ISP_VER: InSimVersion.parse,
    ISP_STA: InSimState.parse,
    ISP_RST: InSimRaceStart.parse,
    ISP_NCN: InSimNewConnection.parse,
    ISP_CNL: InSimConnectionLeft.parse,
    ISP_NPL: InSimNewPlayer.parse,
    ISP_PLP: InSimPlayerTelepit.parse,
    ISP_PLL: InSimPlayerLeaves.parse,
    ISP_SLC: InSimSelectedCar.parse,
    ISP_MAL: InSimModsAllowed.parse,
    ISP_PLA: InSimPitLane.parse,
    ISP_CCH: InSimCameraChange.parse,
    ISP_CIM: InSimInterfaceMode.parse,
    ISP_CSC: InSimCarStateChanged.parse,
    ISP_PEN: InSimPenalty.parse,
    ISP_TOC: InSimTakeOverCar.parse,
    ISP_FLG: InSimFlag.parse,
    ISP_PFL: InSimPlayerFlags.parse,
    ISP_FIN: InSimFinish.parse,
    ISP_RES: InSimResult.parse,
    ISP_SMALL: InSimSmall.parse,
    ISP_LAP: InSimLap.parse,
    ISP_SPX: InSimSplit.parse,
    ISP_PIT: InSimPit.parse,
    ISP_PSF: InSimPitStopFinish.parse,
    ISP_HLV: InSimHotLapValid.parse,
    ISP_MCI: InSimMCI.parse,
    ISP_NLP: InSimNodeLap.parse,
    ISP_OBH: InSimObjectHit.parse,
    ISP_CON: InSimCarContact.parse,
}


class InSimClient:
    """Async InSim TCP client.

    The client emits typed events through :meth:`events` and continuously
    updates :attr:`context`. It auto-sends a ``TINY_NONE`` keepalive every
    30 s and requests current players (``TINY_NPL``) right after handshake.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 29999,
        *,
        admin_password: str = "",
        request_mci: bool = False,
        request_nlp: bool = False,
        request_hlv: bool = True,
        request_obh: bool = True,
        request_con: bool = True,
        mci_interval_ms: int = 100,
        nlp_interval_ms: int = 100,
        client_name: str = "lfs-telemetry",
        connect_retry_interval_s: float = 0.0,
    ) -> None:
        self.host = host
        self.port = port
        self.admin_password = admin_password
        self.request_mci = request_mci
        self.request_nlp = request_nlp
        self.request_hlv = request_hlv
        self.request_obh = request_obh
        self.request_con = request_con
        self.mci_interval_ms = mci_interval_ms
        self.nlp_interval_ms = nlp_interval_ms
        self.client_name = client_name
        self.connect_retry_interval_s = float(connect_retry_interval_s)

        self.context = RaceContext()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._tasks: list[asyncio.Task] = []
        self._queue: asyncio.Queue[object] = asyncio.Queue(2048)

    # -- lifecycle ----------------------------------------------------------

    async def __aenter__(self) -> InSimClient:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    async def start(self) -> None:
        self._reader, self._writer = await self._connect_with_retry()
        flags = ISF_LOCAL
        if self.request_hlv:
            flags |= ISF_HLV
        if self.request_obh:
            flags |= ISF_OBH
        if self.request_con:
            flags |= ISF_CON
        if self.request_mci:
            flags |= ISF_MCI
        if self.request_nlp:
            flags |= ISF_NLP
        # If both MCI and NLP are requested, LFS uses a single Interval; we
        # bias to MCI's value (richer payload). Otherwise pick whichever is set.
        if self.request_mci:
            interval = self.mci_interval_ms
        elif self.request_nlp:
            interval = self.nlp_interval_ms
        else:
            interval = 0
        isi = build_isi_packet(
            udp_port=0,
            flags=flags,
            prefix="!",
            interval_ms=interval,
            admin_password=self.admin_password,
            iname=self.client_name,
        )
        self._writer.write(isi)
        await self._writer.drain()
        # Ask LFS to send IS_STA (current state: track, weather, race) and
        # IS_NPL for every existing player. Without these requests we'd only
        # see context updates after the next state change in LFS.
        #
        # LFS prints "InSim : TINY_xxx with no ReqI" in red top-right when
        # an info request arrives with ReqI=0, so every TINY below uses
        # ReqI=1. TINY_SLC and TINY_MAL are skipped on purpose: they are
        # only meaningful when connected to a multiplayer host as admin,
        # and LFS prints another red warning ("only for multiplayer
        # hosts") if we send them from a singleplayer / non-admin session.
        self._writer.write(build_tiny_packet(TINY_SST, req_i=1))
        self._writer.write(build_tiny_packet(TINY_NCN, req_i=1))
        self._writer.write(build_tiny_packet(TINY_NPL, req_i=1))
        await self._writer.drain()
        loop = asyncio.get_running_loop()
        self._tasks = [
            loop.create_task(self._read_loop()),
            loop.create_task(self._keepalive_loop()),
            loop.create_task(self._state_refresh_loop()),
        ]

    async def _connect_with_retry(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open the InSim TCP socket, optionally retrying on failure.

        With ``connect_retry_interval_s == 0`` (default) this is a single
        attempt that propagates any exception. With a positive interval
        the call retries indefinitely on ``ConnectionRefusedError`` and
        other transport errors so the capture process can be started
        before LFS is open or before ``/insim`` is enabled.
        """
        if self.connect_retry_interval_s <= 0:
            return await asyncio.open_connection(self.host, self.port)
        import sys as _sys
        attempts = 0
        while True:
            try:
                return await asyncio.open_connection(self.host, self.port)
            except (ConnectionRefusedError, ConnectionResetError,
                    ConnectionAbortedError, OSError) as exc:
                attempts += 1
                if attempts == 1 or attempts % 10 == 0:
                    print(
                        f"[insim] waiting for LFS on "
                        f"{self.host}:{self.port} "
                        f"({type(exc).__name__}; attempt {attempts})",
                        file=_sys.stderr,
                    )
                await asyncio.sleep(self.connect_retry_interval_s)

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                _LOG.debug(
                    "InSim background task raised on shutdown: %s", exc,
                )
        self._tasks.clear()
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception as exc:  # noqa: BLE001
                _LOG.debug("InSim writer close failed: %s", exc)
        self._writer = None
        self._reader = None

    async def send_message(self, text: str) -> bool:
        """Display ``text`` on the local LFS screen via IS_MSL.

        Returns ``True`` if the packet was queued, ``False`` if the
        connection is not open. Never raises.
        """
        if self._writer is None:
            return False
        try:
            self._writer.write(build_msl_packet(text))
            await self._writer.drain()
            return True
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("send_message failed: %s", exc)
            return False

    # -- consumption --------------------------------------------------------

    async def events(self) -> AsyncIterator[object]:
        """Yield parsed InSim packets as they arrive."""
        while True:
            yield await self._queue.get()

    # -- internals ----------------------------------------------------------

    async def _read_loop(self) -> None:
        assert self._reader is not None
        reader = self._reader
        pkt_count = 0
        while True:
            header = await reader.readexactly(1)
            # InSim v9+: Size byte represents packet length / 4.
            size = header[0] * 4
            if size < 4:
                continue
            rest = await reader.readexactly(size - 1)
            packet = header + rest
            ptype = packet[1]
            pkt_count += 1
            _LOG.info("InSim RX #%d type=%d size=%d", pkt_count, ptype, size)
            parser = _PACKET_PARSERS.get(ptype)
            if parser is None:
                _LOG.info("InSim RX type=%d not parsed (no handler)", ptype)
                continue
            try:
                evt = parser(packet)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("InSim parse error type=%d size=%d: %s",
                             ptype, size, exc)
                continue
            self.context.update(evt)
            _LOG.info("InSim ctx after type=%d: view_plid=%s "
                      "race_in_progress=%s lap_count=%s",
                      ptype, self.context.view_player_id,
                      self.context.race_in_progress,
                      dict(self.context.lap_count))
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(evt)

    async def _keepalive_loop(self) -> None:
        assert self._writer is not None
        try:
            while True:
                await asyncio.sleep(_KEEPALIVE_INTERVAL_S)
                self._writer.write(build_tiny_packet(TINY_NONE))
                await self._writer.drain()
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _state_refresh_loop(self) -> None:
        """Periodically re-request IS_STA so ``track`` stays in sync.

        Without this, a user who switches to a different track without
        restarting LFS (or if the spontaneous IS_STA from LFS was lost)
        keeps seeing the previous track in the live overlay forever.
        """
        assert self._writer is not None
        try:
            while True:
                await asyncio.sleep(_STATE_REFRESH_INTERVAL_S)
                self._writer.write(build_tiny_packet(TINY_SST, req_i=1))
                await self._writer.drain()
        except asyncio.CancelledError:
            raise
        except Exception:
            return
