"""Command-line interface for lfs-telemetry.

Subcommands:
    capture          Listen to LFS UDP and write a CSV stint capture.
    calibrate        Auto-measure car mass + weight distribution from rest.
    reslice          Re-slice an aggregate capture CSV into per-lap CSVs.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import os
import sys
from pathlib import Path

from .telemetry import live_publisher
from .telemetry.car_calibration import CarSpecStore, RestCalibrator
from .telemetry.constants import (
    INSIM_DEFAULT_PORT,
    OUTGAUGE_DEFAULT_PORT,
    OUTSIM_DEFAULT_PORT,
)
from .telemetry.fuel_tracker import FuelTracker
from .telemetry.lap_slicer import reslice_csv, write_per_lap_files
from .telemetry.live import LiveTelemetry, TelemetrySample
from .telemetry.node_delta import NodeDeltaTracker
from .telemetry.predict import SplitPredictor
from .telemetry.protocol.packets import DL_PITSPEED, OSO_ALL, WHEEL_ORDER
from .telemetry.replay import write_csv_replay
from .telemetry.session_naming import session_tag as _session_tag
from .telemetry.track.loader import find_racing_line_csv

# Module-level stop flag set by SIGBREAK / SIGINT handlers. The capture
# loop polls it every sample so a Stop from the Studio (CTRL_BREAK_EVENT)
# unwinds the loop cleanly and the post-loop CSV writer still runs.
_STOP_REQUESTED = False
_CAPTURE_LOOP: asyncio.AbstractEventLoop | None = None
_CAPTURE_TASK: asyncio.Task | None = None


def _request_stop(*_args) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("[capture] stop requested, flushing…", file=sys.stderr)
    # Wake up the asyncio loop even if no UDP samples are arriving:
    # cancel the awaited operation so the `async for` raises and the
    # `except BaseException` path runs cleanly.
    loop = _CAPTURE_LOOP
    task = _CAPTURE_TASK
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


def main(argv: list[str] | None = None) -> int:
    _harden_std_streams()
    parser = argparse.ArgumentParser(prog="lfs-telemetry")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cap = sub.add_parser("capture", help="record LFS UDP telemetry to CSV")
    p_cap.add_argument("output", type=Path, help="output CSV path")
    _add_lfs_flags(p_cap)
    p_cap.add_argument("--seconds", type=float, default=0.0,
                       help="stop after N seconds (0 = no time limit)")
    p_cap.add_argument(
        "--laps", type=int, default=0,
        help="stop after N completed flying laps (0 = no lap limit). "
             "Requires --insim-host. The lap that's already in progress when "
             "capture starts is skipped (treated as out-lap).")
    p_cap.add_argument(
        "--warmup-laps", type=int, default=0,
        help="extra full laps to discard at the start (in addition to the "
             "in-progress out-lap). Useful for tyre warm-up.")
    p_cap.add_argument(
        "--trim-out-lap", action="store_true", default=False,
        help="discard samples taken before the first lap completion. "
             "OFF by default: the out-lap is kept so the user decides "
             "later which data to use.")
    p_cap.add_argument(
        "--per-lap", action="store_true",
        help="also write one CSV per lap next to the aggregate output. "
             "Files are tagged with capture timestamp + car + track to avoid "
             "overwriting earlier sessions, e.g. "
             "stint_20260514-153012_FBM_BL1_lap01.csv. "
             "Requires --insim-host (uses IS_LAP boundaries).")
    p_cap.add_argument(
        "--no-aggregate", action="store_true",
        help="with --per-lap, skip writing the combined CSV.")
    p_cap.add_argument(
        "--include-out-lap", action="store_true",
        help="with --per-lap, also write the out-lap (from capture start "
             "to first start/finish crossing) as _lap00.csv. Disables "
             "--warmup-laps so every completed lap is preserved.")
    p_cap.add_argument(
        "--wait-on-track", action="store_true",
        help="keep retrying the InSim TCP connection until LFS is up, "
             "and discard incoming samples until the car actually starts "
             "moving (>3 m/s) so the recording is aligned with race "
             "start / pit exit. Implied by --include-out-lap.")
    p_cap.add_argument(
        "--debug-insim", action="store_true",
        help="log every InSim packet (state, lap, npl, mci...) to stderr "
             "and emit a heartbeat with current RaceContext every 5 s.")
    p_cap.add_argument(
        "--stop-file", type=Path, default=None,
        help="path to a sentinel file; when it appears, capture stops "
             "cleanly (used by the Studio Capture tab to stop the "
             "child process when the parent has no console).")
    p_cap.add_argument(
        "--live-file", type=Path, default=None,
        help="path to a JSON snapshot file refreshed at ~10 Hz with "
             "the current race state (position, lap times, delta vs "
             "best, fuel, traffic, radar). Consumed by the Studio "
             "Live tab to drive the in-game-style overlay. Implies "
             "InSim MCI subscription.")
    p_cap.add_argument(
        "--no-csv", action="store_true",
        help="overlay-only mode: connect to LFS and refresh the live "
             "snapshot (requires --live-file), but do NOT buffer "
             "samples in memory and do NOT write any per-lap or "
             "aggregate CSV at the end. Useful when the user only "
             "wants the in-game overlay without recording telemetry.")

    p_cal = sub.add_parser(
        "calibrate",
        help="auto-measure car mass + weight distribution from rest telemetry")
    _add_lfs_flags(p_cal)
    p_cal.add_argument("--timeout", type=float, default=120.0,
                       help="give up if no rest window detected in N seconds")
    p_cal.add_argument("--store", type=Path, default=None,
                       help="custom JSON store path (default ~/.lfs-telemetry/cars.json)")
    p_cal.add_argument("--show", action="store_true",
                       help="just print existing store contents and exit")

    p_res = sub.add_parser(
        "reslice",
        help="re-slice an aggregate capture CSV into clean line-to-line "
             "per-lap CSVs using the canonical current_lap_dist_m wraparound.")
    p_res.add_argument("input", type=Path,
                       help="aggregate CSV produced by `lfs-telemetry capture`")
    p_res.add_argument("--out-dir", type=Path, default=None,
                       help="output directory (default: same as input)")
    p_res.add_argument("--stem", default=None,
                       help="filename stem (default: input filename stem)")
    p_res.add_argument("--suffix", default=".csv",
                       help="output suffix (default: .csv)")
    p_res.add_argument("--session-tag", default="",
                       help="optional tag inserted between stem and lapNN")
    p_res.add_argument("--min-drop-m", type=float, default=100.0,
                       help="minimum negative jump in current_lap_dist_m to "
                            "count as a line crossing (default 100 m)")

    p_raf = sub.add_parser(
        "raf-import",
        help="convert an LFS RAF (Replay Analyser File) v2 into one CSV "
             "per detected lap, using the app's standard schema so they "
             "can be loaded by the Studio for cross-driver comparison.")
    p_raf.add_argument("input", type=Path, help=".raf file produced by LFS")
    p_raf.add_argument("--out-dir", type=Path, default=None,
                       help="output directory "
                            "(default: <input>_raf_laps next to the file)")
    p_raf.add_argument("--keep-outlap", action="store_true",
                       help="also export the lead-in partial lap "
                            "(before the first start/finish crossing)")
    p_raf.add_argument("--min-samples", type=int, default=100,
                       help="discard lap segments shorter than this many "
                            "samples (default 100)")
    p_raf.add_argument("--inspect", type=int, default=0, metavar="N",
                       help="diagnostic: parse but do NOT write CSVs; "
                            "dump the header and the first N decoded "
                            "data blocks to stdout (use 5-10 to debug "
                            "a RAF that produces wrong telemetry)")

    args = parser.parse_args(argv)

    # On Windows, the Studio's Capture tab stops the child by sending
    # CTRL_BREAK_EVENT to the new process group. Python does not turn
    # SIGBREAK into KeyboardInterrupt by default (only SIGINT/Ctrl-C),
    # so the asyncio loop would die with STATUS_CONTROL_C_EXIT and
    # never reach the per-lap CSV writing block. Instead of raising
    # KeyboardInterrupt (which would abandon the coroutine while it's
    # suspended on an await), we set a stop flag that the capture loop
    # polls each sample, so it unwinds cleanly and writes the CSVs.
    if sys.platform == "win32":
        import signal as _signal
        for _name in ("SIGBREAK", "SIGINT"):
            _sig = getattr(_signal, _name, None)
            if _sig is None:
                continue
            with contextlib.suppress(ValueError, OSError):
                _signal.signal(_sig, _request_stop)

    if args.cmd == "capture":
        return asyncio.run(_cmd_capture(args))
    if args.cmd == "calibrate":
        return asyncio.run(_cmd_calibrate(args))
    if args.cmd == "reslice":
        return _cmd_reslice(args)
    if args.cmd == "raf-import":
        return _cmd_raf_import(args)
    parser.error(f"unknown command: {args.cmd}")
    return 2


# ---------------------------------------------------------------------------


async def _cmd_capture(args: argparse.Namespace) -> int:
    global _CAPTURE_LOOP, _CAPTURE_TASK
    _CAPTURE_LOOP = asyncio.get_running_loop()
    _CAPTURE_TASK = asyncio.current_task()
    samples: list[TelemetrySample] = []
    use_laps = args.laps > 0
    per_lap = args.per_lap
    # Overlay-only mode: keep the InSim/OutSim pipeline + live.json
    # publisher running, but don't retain samples in memory and don't
    # emit any CSV when the capture stops.
    no_csv = bool(getattr(args, "no_csv", False))
    # --include-out-lap means: keep every sample, no warmup skipping,
    # no out-lap trimming. The slicer will emit _lap00 for the out-lap.
    if getattr(args, "include_out_lap", False):
        if int(getattr(args, "warmup_laps", 0) or 0) > 0:
            print(
                "[capture] --include-out-lap overrides --warmup-laps; "
                "every completed lap will be kept.",
                file=sys.stderr,
            )
        args.trim_out_lap = False
        args.warmup_laps = 0
        args.wait_on_track = True
    wait_on_track = bool(getattr(args, "wait_on_track", False))
    insim_retry_s = 2.0 if wait_on_track else 0.0
    armed = not wait_on_track  # if not waiting, every sample counts immediately
    arm_speed_mps = 3.0  # ~10.8 km/h
    if args.debug_insim:
        import logging
        logging.basicConfig(level=logging.INFO,
                            format="[insim-log] %(message)s",
                            stream=sys.stderr)
        logging.getLogger("lfs_telemetry.telemetry.protocol.insim").setLevel(logging.INFO)
    if use_laps and not args.insim_host:
        print("[capture] --laps requires --insim-host (need IS_LAP packets)",
              file=sys.stderr)
        return 2
    if per_lap and not args.insim_host:
        print("[capture] --per-lap requires --insim-host", file=sys.stderr)
        return 2
    if args.seconds <= 0 and not use_laps:
        print("[capture] no stop condition — capturing until Ctrl-C. "
              "Use --seconds N or --laps N to set a limit.", file=sys.stderr)
    print(
        f"[capture] OutSim={args.outsim_port}, OutGauge={args.outgauge_port}, "
        f"OutSimOpts=0x{args.outsim_opts:x}, "
        f"InSim={args.insim_host or 'off'}:{args.insim_port}. "
        f"laps={args.laps or 'off'} seconds={args.seconds or 'off'} "
        f"per-lap={'on' if per_lap else 'off'}.",
        file=sys.stderr,
    )

    # Lap-state for the active driver. When --laps is used we trim everything
    # captured before lap_count first transitions 0 -> 1 (out-lap), then keep
    # samples until lap_count reaches start_lap + warmup_laps + laps.
    target_plid: int | None = None
    last_lap_count = -1
    flying_lap_started = False
    flying_lap_start_idx = 0
    completed_flying_laps = 0
    laps_to_skip = max(0, args.warmup_laps)
    # Per-lap slices: list of (lap_number, list_of_samples, lap_ms_or_None)
    lap_slices: list[tuple[int, list[TelemetrySample], int | None]] = []

    # Streaming per-lap output: when --per-lap is on, write each lap's
    # CSV the moment it closes (canonical line crossing) instead of
    # batching everything at the end. Lets the user open laps from the
    # current session in the Studio without having to stop/refresh. The
    # final ``_flush_capture`` becomes a no-op for already-written laps
    # but still emits the aggregate CSV.
    out_dir = args.output.parent
    stem = args.output.stem
    suffix = args.output.suffix or ".csv"
    streaming_session_tag: str = ""
    written_lap_indices: set[int] = set()

    def _ensure_session_tag(reference_sample: TelemetrySample | None) -> str:
        """Compute the session tag once and reuse it for the whole capture."""
        nonlocal streaming_session_tag
        if streaming_session_tag:
            return streaming_session_tag
        # Use the current buffer so car/track are populated; ``samples``
        # is already at least one element by the time the first lap
        # closes. If somehow empty, fall back to the reference sample.
        pool = samples if samples else (
            [reference_sample] if reference_sample is not None else []
        )
        streaming_session_tag = _session_tag(pool)
        return streaming_session_tag

    def _write_lap_atomic(
        lap_index: int,
        slice_samples: list[TelemetrySample],
        lap_ms: int | None,
    ) -> None:
        """Write one lap CSV atomically (tmp + os.replace) to avoid
        readers (Studio refresh, catalog) ever seeing a partial file.
        """
        if not slice_samples:
            return
        if lap_index in written_lap_indices:
            return
        tag = _ensure_session_tag(slice_samples[0])
        tag_part = f"_{tag}" if tag else ""
        path = out_dir / f"{stem}{tag_part}_lap{lap_index:02d}{suffix}"
        tmp = path.with_name(path.name + ".part")
        try:
            rows = write_csv_replay(tmp, slice_samples)
            os.replace(tmp, path)
        except BaseException as exc:  # noqa: BLE001
            print(
                f"[capture] streaming lap{lap_index:02d} write failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            with contextlib.suppress(OSError):
                tmp.unlink()
            return
        written_lap_indices.add(lap_index)
        lap_s = (lap_ms / 1000.0) if lap_ms else None
        tag_dur = f" ({lap_s:.3f}s)" if lap_s else ""
        print(
            f"[capture] wrote {rows} rows to {path} (streaming){tag_dur}",
            file=sys.stderr,
        )

    def _close_lap(end_idx_exclusive: int, lap_ms: int | None) -> None:
        """Snapshot samples[start_idx:end_idx] as a completed flying lap
        and stream it to disk immediately so the user can inspect
        completed laps without stopping the capture.
        """
        nonlocal flying_lap_start_idx
        if not per_lap or not flying_lap_started:
            return
        slice_ = samples[flying_lap_start_idx:end_idx_exclusive]
        if slice_:
            lap_index = completed_flying_laps + 1
            lap_slices.append((lap_index, list(slice_), lap_ms))
            _write_lap_atomic(lap_index, list(slice_), lap_ms)
        flying_lap_start_idx = end_idx_exclusive

    live_file = getattr(args, "live_file", None)
    try:
        async with LiveTelemetry(
            args.outsim_port, args.outgauge_port,
            outsim_opts=args.outsim_opts,
            insim_host=args.insim_host,
            insim_port=args.insim_port,
            insim_admin_password=args.insim_admin,
            insim_connect_retry_interval_s=insim_retry_s,
            # MCI is required for the live overlay's traffic/radar
            # modules and is cheap (one packet per 100 ms). We also
            # leave it on when no live overlay is configured so the
            # recorded session captures opponent positions for later
            # offline analysis.
            insim_request_mci=True,
            insim_mci_interval_ms=100,
        ) as live:
            stop_file = getattr(args, "stop_file", None)
            stop_watcher_task: asyncio.Task[None] | None = None
            if stop_file is not None:
                main_task = asyncio.current_task()

                async def _watch_stop_file(
                    path: Path,
                    target: asyncio.Task | None,
                ) -> None:
                    global _STOP_REQUESTED
                    while not _STOP_REQUESTED:
                        try:
                            if path.exists():
                                _STOP_REQUESTED = True
                                print(
                                    f"[capture] stop-file detected: {path}",
                                    file=sys.stderr,
                                )
                                # Cancel the main capture task so the
                                # blocking ``async for sample in
                                # live.samples()`` unblocks even if no
                                # more UDP packets arrive (InSim can
                                # stall after a parse error). The
                                # CancelledError handler below falls
                                # through to ``_flush_capture`` so the
                                # CSVs are written.
                                if target is not None and not target.done():
                                    target.cancel()
                                return
                        except OSError:
                            pass
                        await asyncio.sleep(0.25)

                stop_watcher_task = asyncio.create_task(
                    _watch_stop_file(stop_file, main_task)
                )
            if wait_on_track:
                print(
                    "[capture] InSim ready. Waiting for car to start "
                    "moving (race start / pit exit) before recording...",
                    file=sys.stderr,
                )
            deadline = (None if args.seconds <= 0
                        else asyncio.get_running_loop().time() + args.seconds)
            last_heartbeat = asyncio.get_running_loop().time()
            last_seen_lap_count: dict[int, int] = {}
            # Live snapshot publisher state (only used when --live-file).
            last_live_publish_t: float = 0.0
            last_lap_seen_at: float | None = None
            # LFS-internal lap-start clock (OutGauge.time_ms at the moment
            # IS_LAP arrived). Preferred over monotonic wall-clock because
            # OutGauge ticks at 50 Hz with the LFS engine's own timer, so
            # the per-tick elapsed (used to feed node_delta_tracker and
            # split_predictor) is free of InSim network jitter and asyncio
            # scheduling latency. Mirrors Detect&Monitor's approach (which
            # also requires OutGauge alongside InSim for delta features).
            lap_start_og_time_ms: int | None = None
            node_delta_tracker = NodeDeltaTracker()
            split_predictor = SplitPredictor(n_splits=3)
            fuel_tracker = FuelTracker(window=3)
            # Per-track racing-line cache: maps LFS path-node index ->
            # cumulative arclength in metres. Lets traffic.py compute
            # on-track gaps (instead of straight-line euclidean) which
            # is critical in curvy sections. Lazy-loaded the first
            # time we see a track name and reloaded on track changes.
            _rl_cache_track: str | None = None
            _rl_node_to_s_m: list[float] = []
            _rl_total_length_m: float = 0.0
            last_seen_split_idx = 0
            async for sample in live.samples():
                # ---- arm-on-motion gate -----------------------------
                if not armed:
                    speed = 0.0
                    if sample.outsim is not None:
                        vx, vy, vz = sample.outsim.vel
                        speed = (vx * vx + vy * vy + vz * vz) ** 0.5
                    if speed >= arm_speed_mps:
                        armed = True
                        # Anchor the LFS-internal lap clock to *now*. Without
                        # this, ``lap_start_og_time_ms`` stays None until the
                        # first IS_LAP fires, so during lap 1 we record no
                        # nodes into NodeDeltaTracker and the live overlay's
                        # delta bar never gets a PB to compare against (it
                        # would only start working from lap 3+). Anchoring on
                        # arm means lap 1 already feeds the tracker, so the
                        # delta bar lights up on lap 2 — matching what users
                        # expect from in-game delta overlays.
                        og_arm = sample.outgauge
                        if og_arm is not None and lap_start_og_time_ms is None:
                            lap_start_og_time_ms = int(og_arm.time_ms)
                            last_lap_seen_at = asyncio.get_running_loop().time()
                        print(
                            f"[capture] armed: car moving "
                            f"({speed * 3.6:.1f} km/h) — recording starts now",
                            file=sys.stderr,
                        )
                    else:
                        # Stationary in pits / on grid / countdown — drop.
                        continue

                if not no_csv:
                    samples.append(sample)

                # ------------- diagnostic heartbeat ----------------
                if args.debug_insim and sample.race_context is not None:
                    rc = sample.race_context
                    # Detect any lap_count change for any PLID.
                    for plid, lc in rc.lap_count.items():
                        if lc != last_seen_lap_count.get(plid, 0):
                            llms = rc.last_lap_ms.get(plid)
                            print(f"[debug] IS_LAP plid={plid} laps={lc}"
                                  f" last={llms}ms", file=sys.stderr)
                            last_seen_lap_count[plid] = lc
                    now = asyncio.get_running_loop().time()
                    if now - last_heartbeat >= 5.0:
                        last_heartbeat = now
                        print(
                            f"[debug] hb view_plid={rc.view_player_id} "
                            f"track={rc.track} race_in_progress={rc.race_in_progress} "
                            f"players={list(rc.players.keys())} "
                            f"lap_count={dict(rc.lap_count)}",
                            file=sys.stderr,
                        )

                if sample.race_context is not None:
                    rc = sample.race_context
                    # Pick the active PLID. Prefer the one LFS reports as
                    # "viewed". If view_plid hasn't propagated yet, fall back
                    # to whichever PLID is racking up laps.
                    #
                    # NOTE: also re-latch if ``view_player_id`` changes
                    # mid-session (driver swap, late IS_NPL, spectator → car).
                    # The previous "latch once and forget" logic left
                    # ``target_plid`` stuck on a stale/empty PLID, which made
                    # ``cur_lap`` permanently 0 and silently disabled both the
                    # delta-vs-PB overlay and the fuel-laps-remaining widget.
                    rc_view = rc.view_player_id
                    if (
                        rc_view is not None
                        and rc_view in rc.players
                        and rc_view != target_plid
                    ):
                        prev = target_plid
                        target_plid = rc_view
                        last_lap_count = rc.lap_count.get(target_plid, 0)
                        if prev is None:
                            print(f"[capture] tracking PLID {target_plid} "
                                  f"(via view_plid, lap_count={last_lap_count})",
                                  file=sys.stderr)
                        else:
                            print(f"[capture] view_plid changed "
                                  f"{prev} -> {target_plid} "
                                  f"(lap_count={last_lap_count}); resyncing",
                                  file=sys.stderr)
                    elif target_plid is None and rc.lap_count:
                        # First PLID with completed laps = our driver.
                        # IS_NPL never arrived (common): we discover the
                        # PLID only AFTER the first IS_LAP. That IS_LAP is
                        # almost certainly the out-lap completion we just
                        # missed — anchor flying laps starting now.
                        target_plid = max(rc.lap_count, key=rc.lap_count.get)
                        last_lap_count = rc.lap_count.get(target_plid, 0)
                        if use_laps or per_lap:
                            if args.trim_out_lap:
                                samples = [sample]
                                flying_lap_start_idx = 0
                            else:
                                flying_lap_start_idx = len(samples) - 1
                            flying_lap_started = True
                        print(f"[capture] tracking PLID {target_plid} "
                              f"(no IS_NPL; assuming first {last_lap_count} "
                              f"lap(s) were out-lap, starting flying now)",
                              file=sys.stderr)
                    # Only act on lap transitions once we know who we're tracking.
                    cur_lap = (rc.lap_count.get(target_plid, 0)
                               if target_plid is not None else 0)
                    # Detect a race restart / track change. When LFS sends
                    # IS_RST, RaceContext clears its per-PLID lap counters,
                    # so ``cur_lap`` drops below our locally-tracked
                    # ``last_lap_count``. Without this guard the
                    # ``cur_lap > last_lap_count`` check below would never
                    # fire again for the new run and laps would silently
                    # stop being recorded — exactly the bug seen when
                    # users restart the car (Shift+S) or change track.
                    if (
                        target_plid is not None
                        and cur_lap < last_lap_count
                    ):
                        print(
                            f"[capture] lap counter reset detected "
                            f"({last_lap_count} -> {cur_lap}); resyncing",
                            file=sys.stderr,
                        )
                        last_lap_count = cur_lap
                        if use_laps or per_lap:
                            # Drop the now-orphaned in-progress samples and
                            # restart the flying-lap window from this sample.
                            if args.trim_out_lap:
                                samples = [sample]
                                flying_lap_start_idx = 0
                            else:
                                flying_lap_start_idx = len(samples) - 1
                            flying_lap_started = False
                        last_lap_seen_at = None
                        lap_start_og_time_ms = None
                        node_delta_tracker.reset_lap()
                        split_predictor.reset_lap()
                        last_seen_split_idx = 0
                    if target_plid is not None and cur_lap > last_lap_count:
                        # A lap just completed (the sample we just appended is
                        # the FIRST of the next lap). Snapshot index excludes it.
                        last_lap_count = cur_lap
                        last_lap_seen_at = asyncio.get_running_loop().time()
                        og_at_lap = sample.outgauge
                        lap_start_og_time_ms = (
                            int(og_at_lap.time_ms)
                            if og_at_lap is not None else None
                        )
                        lap_ms = (rc.last_lap_ms.get(target_plid)
                                  if target_plid is not None else None)
                        # Promote the lap we just finished into the per-node
                        # PB table for the continuous delta bar.
                        if lap_ms is not None:
                            node_delta_tracker.complete_lap(int(lap_ms))
                            split_predictor.observe_lap(int(lap_ms))
                            # Fuel: use the most recent OutGauge fuel%.
                            og_for_fuel = sample.outgauge
                            if og_for_fuel is not None:
                                fuel_tracker.observe_lap(
                                    og_for_fuel.fuel * 100.0
                                )
                            last_seen_split_idx = 0
                        else:
                            node_delta_tracker.reset_lap()
                            split_predictor.reset_lap()
                            last_seen_split_idx = 0
                        # CSV slicing / per-lap output — only when the user
                        # actually asked for it (--per-lap or --laps N). The
                        # live overlay always benefits from the trackers
                        # above regardless of CSV emission.
                        if use_laps or per_lap:
                            if not flying_lap_started:
                                # First lap completion = end of out-lap. Drop the
                                # in-progress lap from the buffer and re-anchor.
                                # When --include-out-lap is on, stream the
                                # out-lap (lap00) to disk BEFORE we trim the
                                # buffer / re-anchor, so the user can inspect
                                # it mid-session.
                                outlap_end_idx = len(samples) - 1
                                if (
                                    per_lap
                                    and getattr(args, "include_out_lap", False)
                                    and outlap_end_idx > 0
                                    and 0 not in written_lap_indices
                                ):
                                    _write_lap_atomic(
                                        0,
                                        list(samples[:outlap_end_idx]),
                                        lap_ms,
                                    )
                                if args.trim_out_lap:
                                    samples = [sample]
                                    flying_lap_start_idx = 0
                                else:
                                    flying_lap_start_idx = len(samples) - 1
                                flying_lap_started = True
                                print(
                                    f"[capture] out-lap complete, starting flying laps "
                                    f"(skipping {laps_to_skip} warmup, "
                                    f"recording {args.laps or 'all'})",
                                    file=sys.stderr)
                            else:
                                if laps_to_skip > 0:
                                    laps_to_skip -= 1
                                    samples = [sample]      # drop warmup lap
                                    flying_lap_start_idx = 0
                                    print(f"[capture] warmup lap done, "
                                          f"{laps_to_skip} more to skip",
                                          file=sys.stderr)
                                else:
                                    # Close the just-finished flying lap.
                                    _close_lap(len(samples) - 1, lap_ms)
                                    completed_flying_laps += 1
                                    lap_s = (lap_ms / 1000.0) if lap_ms else None
                                    print(
                                        f"[capture] flying lap "
                                        f"{completed_flying_laps}"
                                        + (f"/{args.laps}" if use_laps else "")
                                        + " complete"
                                        + (f" ({lap_s:.3f}s)" if lap_s else ""),
                                        file=sys.stderr)
                                    if use_laps and completed_flying_laps >= args.laps:
                                        break

                if len(samples) % 200 == 0:
                    print(f"[capture] {len(samples)} samples", file=sys.stderr)
                # Live snapshot for the Studio overlay (~10 Hz).
                if live_file is not None:
                    now_loop = asyncio.get_running_loop().time()
                    if now_loop - last_live_publish_t >= 0.1:
                        last_live_publish_t = now_loop
                        og = sample.outgauge
                        # Estimate "current lap so far" from elapsed wall
                        # time since the last IS_LAP. RaceContext doesn't
                        # expose it directly so we approximate.
                        cur_lap_ms: int | None = None
                        rc_live = live.race_context
                        # Refresh the per-track racing-line arclength
                        # cache when LFS reports a new circuit.
                        track_now = (
                            rc_live.track if rc_live is not None else None
                        )
                        if track_now and track_now != _rl_cache_track:
                            _rl_cache_track = track_now
                            _rl_node_to_s_m = []
                            _rl_total_length_m = 0.0
                            try:
                                csv_path = find_racing_line_csv(track_now)
                                if csv_path is not None:
                                    import csv as _csv
                                    with csv_path.open(
                                        "r", newline="", encoding="utf-8",
                                    ) as fh:
                                        reader = _csv.DictReader(fh)
                                        for row in reader:
                                            try:
                                                _rl_node_to_s_m.append(
                                                    float(row["s_m"])
                                                )
                                            except (KeyError, TypeError,
                                                    ValueError):
                                                continue
                                    if len(_rl_node_to_s_m) >= 2:
                                        # Close the loop: assume the
                                        # gap from last node back to
                                        # node 0 equals the average
                                        # inter-node spacing (good
                                        # enough for wraparound
                                        # bookkeeping).
                                        avg = (
                                            _rl_node_to_s_m[-1]
                                            / max(len(_rl_node_to_s_m) - 1, 1)
                                        )
                                        _rl_total_length_m = (
                                            _rl_node_to_s_m[-1] + avg
                                        )
                            except Exception as exc:  # noqa: BLE001
                                print(
                                    f"[capture] racing-line load failed "
                                    f"for {track_now}: "
                                    f"{type(exc).__name__}: {exc}",
                                    file=sys.stderr,
                                )
                                _rl_node_to_s_m = []
                                _rl_total_length_m = 0.0
                        if (
                            rc_live is not None
                            and rc_live.view_player_id is not None
                        ):
                            og_for_clock = sample.outgauge
                            if (
                                lap_start_og_time_ms is not None
                                and og_for_clock is not None
                            ):
                                # LFS-anchored: subtract OutGauge timestamps
                                # (uint32 ms wrap handled implicitly within
                                # one lap — wrap period is ~49 days).
                                cur_lap_ms = int(
                                    og_for_clock.time_ms
                                    - lap_start_og_time_ms
                                ) & 0xFFFFFFFF
                            elif last_lap_seen_at is not None:
                                # Fallback when OutGauge isn't available
                                # (e.g. user didn't configure cfg.txt).
                                cur_lap_ms = int(
                                    (now_loop - last_lap_seen_at) * 1000.0
                                )
                        # Per-node continuous delta vs PB.
                        node_delta_value: int | None = None
                        node_speed_delta_value: float | None = None
                        ghost_node_value: int | None = None
                        if (
                            cur_lap_ms is not None
                            and rc_live is not None
                            and rc_live.last_mci is not None
                            and rc_live.view_player_id is not None
                        ):
                            view_plid = rc_live.view_player_id
                            view_car = next(
                                (c for c in rc_live.last_mci.cars
                                 if c.player_id == view_plid),
                                None,
                            )
                            if view_car is not None:
                                node_delta_tracker.record(
                                    node=view_car.node,
                                    elapsed_ms=cur_lap_ms,
                                    speed_ms=float(view_car.speed_ms),
                                )
                                node_delta_value = (
                                    node_delta_tracker.delta_ms(
                                        node=view_car.node,
                                        elapsed_ms=cur_lap_ms,
                                    )
                                )
                                node_speed_delta_value = (
                                    node_delta_tracker.speed_delta_ms(
                                        node=view_car.node,
                                        speed_ms=float(view_car.speed_ms),
                                    )
                                )
                                ghost_node_value = (
                                    node_delta_tracker.ghost_node_at(
                                        elapsed_ms=cur_lap_ms,
                                    )
                                )
                        # Pick up any new IS_SPX split crossings into the
                        # SplitPredictor for predicted-lap / SPB metrics.
                        predicted_lap_value: int | None = None
                        spb_value: int | None = None
                        if (
                            rc_live is not None
                            and rc_live.view_player_id is not None
                        ):
                            splits = rc_live.last_split_ms.get(
                                rc_live.view_player_id, {}
                            )
                            if splits:
                                max_split = max(splits)
                                if max_split > last_seen_split_idx:
                                    for s_idx in range(
                                        last_seen_split_idx + 1,
                                        max_split + 1,
                                    ):
                                        if s_idx in splits:
                                            split_predictor.observe_split(
                                                s_idx, splits[s_idx]
                                            )
                                    last_seen_split_idx = max_split
                            if cur_lap_ms is not None:
                                predicted_lap_value = (
                                    split_predictor.predicted_lap_ms(
                                        elapsed_ms=cur_lap_ms,
                                        last_split_idx=last_seen_split_idx,
                                    )
                                )
                            spb_value = split_predictor.spb_ms()
                        # Fuel laps remaining.
                        og_now = sample.outgauge
                        cur_fuel_pct = (
                            og_now.fuel * 100.0 if og_now is not None else None
                        )
                        fuel_tracker.observe_fuel(cur_fuel_pct)
                        fuel_laps_value = fuel_tracker.laps_remaining(
                            cur_fuel_pct
                        )
                        # Lateral / longitudinal G + max wheel slip from
                        # extended OutSim packets when present.
                        accel_lat_v: float | None = None
                        accel_lon_v: float | None = None
                        if sample.outsim is not None:
                            ax, ay, _az = sample.outsim.accel
                            # LFS local car frame: X = right, Y = forward
                            # (matches the radar projection convention).
                            accel_lat_v = float(ax)
                            accel_lon_v = float(ay)
                        max_slip_v: float | None = None
                        max_slip_ratio_v: float | None = None
                        tyres_live: list[
                            dict[str, float | str | bool | None]
                        ] = []
                        os2 = getattr(sample, "outsim2", None)
                        if os2 is not None and os2.wheels:
                            max_slip_v = max(
                                w.slip_fraction for w in os2.wheels
                            )
                            max_slip_ratio_v = max(
                                abs(w.slip_ratio) for w in os2.wheels
                            )
                            tyres_live = [
                                {
                                    "corner": corner,
                                    "temp_c": float(w.air_temp_c),
                                    "slip_frac": float(w.slip_fraction),
                                    "slip_ratio": float(w.slip_ratio),
                                    "load_n": float(w.vertical_load_n),
                                    "tan_slip": float(w.tan_slip_angle),
                                    "fx_n": float(w.x_force_n),
                                    "fy_n": float(w.y_force_n),
                                    "touching": bool(w.touching),
                                }
                                for corner, w in zip(WHEEL_ORDER, os2.wheels, strict=True)
                            ]
                            # Prefer extended OutSim's clutch/handbrake
                            # over OutGauge-only when both available.
                        clutch_v = (
                            og_now.clutch if og_now is not None else None
                        )
                        handbrake_v: float | None = None
                        if os2 is not None:
                            if os2.handbrake is not None:
                                handbrake_v = float(os2.handbrake)
                            if os2.clutch is not None:
                                clutch_v = float(os2.clutch)
                        try:
                            snap = live_publisher.build_snapshot(
                                rc_live,
                                armed=armed,
                                samples_count=len(samples),
                                current_lap_ms=cur_lap_ms,
                                last_sample_speed_ms=(
                                    (sample.outsim.vel[0] ** 2
                                     + sample.outsim.vel[1] ** 2
                                     + sample.outsim.vel[2] ** 2) ** 0.5
                                    if sample.outsim is not None else None
                                ),
                                last_sample_rpm=(og.rpm if og is not None else None),
                                last_sample_gear=(og.gear if og is not None else None),
                                last_sample_fuel_pct=(
                                    og.fuel * 100.0 if og is not None else None
                                ),
                                last_sample_throttle=(
                                    og.throttle if og is not None else None
                                ),
                                last_sample_brake=(og.brake if og is not None else None),
                                last_sample_clutch=clutch_v,
                                last_sample_handbrake=handbrake_v,
                                last_sample_accel_lat_ms2=accel_lat_v,
                                last_sample_accel_lon_ms2=accel_lon_v,
                                last_sample_max_slip=max_slip_v,
                                last_sample_max_slip_ratio=max_slip_ratio_v,
                                last_sample_tyres=tyres_live,
                                monotonic_ts=now_loop,
                                delta_to_best_ms=node_delta_value,
                                predicted_lap_ms=predicted_lap_value,
                                spb_ms=spb_value,
                                fuel_laps_remaining=fuel_laps_value,
                                fuel_burn_pct_per_lap=(
                                    fuel_tracker.avg_burn_pct_per_lap
                                ),
                                ghost_node=ghost_node_value,
                                last_sample_pit_limiter=(
                                    bool(og.show_lights & DL_PITSPEED)
                                    if og is not None else None
                                ),
                                speed_delta_ms_vs_best=(
                                    node_speed_delta_value
                                ),
                                node_to_s_m=_rl_node_to_s_m,
                                track_length_m=_rl_total_length_m,
                            )
                            live_publisher.write_snapshot_atomic(
                                live_file, snap
                            )
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"[capture] live snapshot write failed: "
                                f"{type(exc).__name__}: {exc}",
                                file=sys.stderr,
                            )
                if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                    break
                if _STOP_REQUESTED:
                    print("[capture] stop flag set, ending capture loop",
                          file=sys.stderr)
                    break
    except KeyboardInterrupt:
        print("[capture] stopped by user", file=sys.stderr)
    except asyncio.CancelledError:
        print("[capture] stopped by user (cancelled)", file=sys.stderr)
    except BaseException as exc:  # noqa: BLE001
        # Anything else (network error, etc.): log and proceed to flush
        # whatever we already buffered. We never want to lose 4 laps
        # because the 5th got interrupted.
        print(f"[capture] aborted: {type(exc).__name__}: {exc}",
              file=sys.stderr)
    finally:
        if 'stop_watcher_task' in locals() and stop_watcher_task is not None:
            stop_watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await stop_watcher_task

    if (use_laps or per_lap) and not flying_lap_started:
        print("[capture] WARNING: no IS_LAP packet received. Drove past "
              "start/finish? In LFS use S1 lap mode, not /restart loops.",
              file=sys.stderr)

    if no_csv:
        print("[capture] overlay-only mode: no CSV written.",
              file=sys.stderr)
    else:
        _flush_capture(
            args, samples, per_lap,
            written_lap_indices=written_lap_indices,
            session_tag_override=streaming_session_tag,
        )
    return 0


def _flush_capture(
    args: argparse.Namespace,
    samples: list[TelemetrySample],
    per_lap: bool,
    *,
    written_lap_indices: set[int] | None = None,
    session_tag_override: str = "",
) -> None:
    """Write per-lap and/or aggregate CSVs from buffered samples.

    Safe to call from both the normal completion path and from an
    interrupt handler: it never raises (errors are logged instead).

    When the capture loop streamed laps to disk as they completed
    (the default for ``--per-lap``), pass the indices via
    ``written_lap_indices`` so we don't rewrite the same files at
    flush time. ``session_tag_override`` keeps the streaming and
    final filenames in lockstep.
    """
    if not samples:
        print("[capture] no samples buffered, nothing to write.",
              file=sys.stderr)
        return
    skip = written_lap_indices or set()
    try:
        if per_lap:
            out_dir = args.output.parent
            stem = args.output.stem
            suffix = args.output.suffix or ".csv"
            session_tag = session_tag_override or _session_tag(samples)
            written = write_per_lap_files(
                samples,
                out_dir=out_dir,
                stem=stem,
                suffix=suffix,
                session_tag=session_tag,
                include_out_lap=getattr(args, "include_out_lap", False),
                skip_lap_indices=skip,
            )
            new_writes = [w for w in written if w[2] > 0]
            if not written:
                print("[capture] WARNING: no full lap recovered from buffer "
                      "(need at least 2 line crossings in current_lap_dist_m).",
                      file=sys.stderr)
            elif not new_writes:
                print(
                    f"[capture] flush: {len(skip)} lap(s) already streamed; "
                    "nothing new to write.",
                    file=sys.stderr,
                )
            for path, lap, n in new_writes:
                tag = f" ({lap.lap_ms / 1000.0:.3f}s)" if lap.lap_ms else ""
                print(f"[capture] wrote {n:5d} rows to {path}"
                      f" [d_max={lap.distance_m:.1f}m,"
                      f" dur={lap.duration_s:.2f}s]"
                      f"{tag}", file=sys.stderr)

        if not (per_lap and args.no_aggregate):
            rows = write_csv_replay(args.output, samples)
            print(f"[capture] wrote {rows} rows to {args.output}",
                  file=sys.stderr)
    except BaseException as exc:  # noqa: BLE001
        print(f"[capture] flush failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)


def _cmd_reslice(args: argparse.Namespace) -> int:
    """Re-slice a previously captured aggregate CSV into clean per-lap
    files using the canonical ``current_lap_dist_m`` wraparound.
    """
    src = args.input
    if not src.exists():
        print(f"[reslice] {src} does not exist", file=sys.stderr)
        return 2
    out_dir = args.out_dir if args.out_dir is not None else src.parent
    stem = args.stem if args.stem else src.stem
    suffix = args.suffix
    written = reslice_csv(
        src,
        out_dir=out_dir,
        stem=stem,
        suffix=suffix,
        session_tag=args.session_tag,
        min_drop_m=args.min_drop_m,
    )
    if not written:
        print("[reslice] no full lap recovered (need >= 2 line crossings).",
              file=sys.stderr)
        return 1
    for path, lap, n in written:
        tag = f" ({lap.lap_ms / 1000.0:.3f}s)" if lap.lap_ms else ""
        print(f"[reslice] wrote {n:5d} rows to {path}"
              f" [d_max={lap.distance_m:.1f}m, dur={lap.duration_s:.2f}s]"
              f"{tag}")
    return 0


def _raf_inspect(src: Path, n: int) -> int:
    """Diagnostic dump of a RAF file: header + first ``n`` decoded blocks.

    Used to debug RAFs whose decoded telemetry looks wrong
    (constant speed, tiny map, all-zero inputs, etc.). Prints both
    the raw little-endian bytes of each block prefix and the values
    that :func:`parse_raf` extracts, so offset/endianness mismatches
    against the official spec can be spotted at a glance.
    """
    from .telemetry.raf import parse_raf, split_into_laps

    head, rows = parse_raf(src)
    print("[raf-inspect] header:")
    print(f"  raf_version        = {head.raf_version}")
    print(f"  update_interval_ms = {head.update_interval_ms}")
    print(f"  header_size        = {head.header_size}")
    print(f"  block_size         = {head.block_size}")
    print(f"  wheel_block_size   = {head.wheel_block_size}")
    print(f"  wheel_block_offset = {head.wheel_block_offset}")
    print(f"  num_blocks         = {head.num_blocks}")
    print(f"  short_track        = {head.short_track_name!r}")
    print(f"  track_ruler_len_m  = {head.track_ruler_length_m:.2f}")
    print(f"  splits_ms          = {head.splits_ms}")
    print(f"  rows decoded       = {len(rows)}")
    if rows:
        idx_vals = [r['indexed_distance_m'] for r in rows]
        car_vals = [r['_car_distance_m'] for r in rows]
        print(f"  indexed_distance_m : min={min(idx_vals):.2f} "
              f"max={max(idx_vals):.2f} "
              f"first={idx_vals[0]:.2f} last={idx_vals[-1]:.2f}")
        print(f"  _car_distance_m    : min={min(car_vals):.2f} "
              f"max={max(car_vals):.2f} "
              f"first={car_vals[0]:.2f} last={car_vals[-1]:.2f}")
        speeds = [r['speed_ms'] for r in rows]
        print(f"  speed_ms           : min={min(speeds):.2f} "
              f"max={max(speeds):.2f}")
        laps = split_into_laps(head, rows)
        print(f"  laps detected      = {len(laps)} "
              f"(sizes: {[len(lap) for lap in laps[:10]]}"
              f"{'...' if len(laps) > 10 else ''})")
    raw = src.read_bytes()
    print(f"\n[raf-inspect] first {n} block(s) — raw + decoded:")
    for i in range(min(n, len(rows))):
        off = head.header_size + i * head.block_size
        block = raw[off:off + head.block_size]
        r = rows[i]
        print(f"\n  --- block {i} @ file offset {off} ---")
        print(f"  raw[0:32]  = {block[:32].hex(' ')}")
        print(f"  raw[32:64] = {block[32:64].hex(' ')}")
        print(f"  throttle={r['throttle']:.3f} brake={r['brake']:.3f} "
              f"clutch={r['clutch']:.3f} handbrake={r['input_handbrake']:.3f}")
        print(f"  steer={r['input_steer']:.3f}rad gear={r['gear']} "
              f"speed={r['speed_ms']:.3f}m/s rpm={r['rpm']:.0f}")
        print(f"  accel x/y/z = {r['accel_x']:.2f} {r['accel_y']:.2f} "
              f"{r['accel_z']:.2f} m/s\u00b2")
        print(f"  pos x/y/z   = {r['pos_x']:.2f} {r['pos_y']:.2f} "
              f"{r['pos_z']:.2f} m")
        print(f"  heading={r['heading']:.3f}rad pitch={r['pitch']:.3f} "
              f"roll={r['roll']:.3f}")
        print(f"  index_dist={r['indexed_distance_m']:.2f}m "
              f"car_dist={r['_car_distance_m']:.2f}m")
    return 0


def _cmd_raf_import(args: argparse.Namespace) -> int:
    """Convert an LFS RAF replay-analyser file into per-lap CSVs."""
    from .telemetry.raf import parse_raf_header, raf_to_lap_csvs

    src: Path = args.input
    if not src.exists():
        print(f"[raf-import] {src} does not exist", file=sys.stderr)
        return 2
    try:
        head = parse_raf_header(src.read_bytes()[:1024])
    except ValueError as exc:
        print(f"[raf-import] {src}: {exc}", file=sys.stderr)
        return 2
    print(
        f"[raf-import] {src.name}: player={head.player!r} "
        f"car={head.car!r} track={head.track!r} "
        f"({head.num_blocks} samples @ {head.update_interval_ms} ms)",
    )
    if args.inspect:
        return _raf_inspect(src, args.inspect)
    try:
        written = raf_to_lap_csvs(
            src,
            out_dir=args.out_dir,
            skip_outlap=not args.keep_outlap,
            min_samples_per_lap=args.min_samples,
        )
    except ValueError as exc:
        print(f"[raf-import] failed: {exc}", file=sys.stderr)
        return 1
    if not written:
        print(
            "[raf-import] no full lap recovered "
            "(replay too short or only out-lap).",
            file=sys.stderr,
        )
        return 1
    for p in written:
        print(f"[raf-import] wrote {p}")
    return 0


async def _cmd_calibrate(args: argparse.Namespace) -> int:
    store = CarSpecStore(args.store)
    if args.show:
        store.load()
        cars = store.all()
        if not cars:
            print(f"[calibrate] store is empty: {store.path}")
            return 0
        print(f"[calibrate] {store.path}")
        for cid, cal in sorted(cars.items()):
            print(f"  {cid:6s} mass={cal.mass_kg:6.1f} kg  "
                  f"front={cal.weight_dist_front*100:5.1f}%  "
                  f"left={cal.left_fraction*100:5.1f}%  "
                  f"(n={cal.sample_count})")
        return 0

    print(
        "[calibrate] Park the car on a flat track surface, idle gear (or "
        "clutch in), no throttle/brake. The calibration takes ~1 second.",
        file=sys.stderr)
    cal_engine = RestCalibrator()
    deadline = asyncio.get_running_loop().time() + args.timeout
    detected: list[str] = []
    last_diag_t = 0.0
    try:
        async with LiveTelemetry(
            args.outsim_port, args.outgauge_port,
            outsim_opts=args.outsim_opts,
            insim_host=args.insim_host,
            insim_port=args.insim_port,
            insim_admin_password=args.insim_admin,
        ) as live:
            async for sample in live.samples():
                og = sample.outgauge
                car = (og.car if og else None) or "?"
                if car not in detected:
                    detected.append(car)
                    print(f"[calibrate] detected car id: {car}", file=sys.stderr)
                cal = cal_engine.feed(sample)
                if cal is not None:
                    store.put(cal)
                    store.save()
                    print(f"[calibrate] OK  car={cal.car_id}  "
                          f"mass={cal.mass_kg:.1f} kg  "
                          f"front={cal.weight_dist_front*100:.1f}%  "
                          f"left={cal.left_fraction*100:.1f}%  "
                          f"(n={cal.sample_count})  -> {store.path}")
                    return 0
                now = asyncio.get_running_loop().time()
                if now - last_diag_t >= 2.0:
                    last_diag_t = now
                    reason = cal_engine.diagnose()
                    if reason:
                        print(f"[calibrate] waiting: {reason}", file=sys.stderr)
                if now >= deadline:
                    print("[calibrate] timeout — car never reached rest. "
                          "Make sure you are stationary on the track with "
                          "throttle and brake released.", file=sys.stderr)
                    return 1
    except KeyboardInterrupt:
        print("[calibrate] stopped by user", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
