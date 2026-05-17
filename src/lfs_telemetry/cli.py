"""Command-line interface for lfs-telemetry.

Subcommands:
    capture          Listen to LFS UDP and write a CSV stint capture.
    calibrate        Auto-measure car mass + weight distribution from rest.
    reslice          Re-slice an aggregate capture CSV into per-lap CSVs.
    advise           Run the TNFR Setup Advisor on a captured stint.
    compare-stints   Compare two stints (before / after a setup change).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from .telemetry.live import LiveTelemetry, TelemetrySample
from .telemetry import live_publisher
from .telemetry.fuel_tracker import FuelTracker
from .telemetry.node_delta import NodeDeltaTracker
from .telemetry.predict import SplitPredictor
from .telemetry.replay import write_csv_replay
from .telemetry.lap_slicer import reslice_csv, write_per_lap_files
from .telemetry.protocol.packets import OSO_ALL, WHEEL_ORDER
from .telemetry.car_calibration import CarSpecStore, RestCalibrator


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
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:
            pass


def _add_lfs_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--outsim-port", type=int, default=30000)
    parser.add_argument("--outgauge-port", type=int, default=30001)
    parser.add_argument(
        "--outsim-opts", type=lambda s: int(s, 0), default=OSO_ALL,
        help="OutSim Opts hex flags (default 0x1ff = full extended packets)")
    parser.add_argument(
        "--insim-host", default=None,
        help="enable InSim TCP client by host (e.g. 127.0.0.1)")
    parser.add_argument("--insim-port", type=int, default=29999)
    parser.add_argument("--insim-admin", default="",
                        help="LFS admin password (if required)")
    parser.add_argument(
        "--car", default=None,
        help="LFS car short name (FOX, FO8, BF1, MRT). Overrides auto-detect.")


def main(argv: list[str] | None = None) -> int:
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
        "--trim-out-lap", action="store_true", default=True,
        help="discard samples taken before the first lap completion "
             "(default ON when --laps is used).")
    p_cap.add_argument(
        "--no-trim-out-lap", action="store_false", dest="trim_out_lap",
        help="keep the in-progress out-lap in the output CSV.")
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
             "to first start/finish crossing) as _lap00.csv. Implies "
             "--no-trim-out-lap and disables --warmup-laps so every "
             "completed lap is preserved.")
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

    p_adv = sub.add_parser(
        "advise",
        help="run the TNFR Setup Advisor on a captured stint and write a "
             "JSON / Markdown report.")
    p_adv.add_argument(
        "--car", required=True,
        help="LFS car short name used to resolve <CAR>_CAR_info.bin if "
             "--baseline is not given (e.g. FBM, FOX, BF1).")
    p_adv.add_argument(
        "--track", required=True,
        help="LFS track short code (e.g. BL1, AS3, KY2R). Used to seed "
             "the per-track network and the stint signature.")
    p_adv.add_argument(
        "--laps", nargs="+", type=Path, required=True,
        metavar="LAP_CSV",
        help="2+ per-lap CSV files (at least 5 consecutive valid laps "
             "are required for an actual proposal).")
    p_adv.add_argument(
        "--baseline", type=Path, default=None,
        help="explicit path to <CAR>_CAR_info.bin. If omitted, the "
             "standard asset search path is used (assets/source/cars/).")
    p_adv.add_argument(
        "--seed", type=int, default=20260516,
        help="deterministic seed for the network builders (default "
             "20260516).")
    p_adv.add_argument(
        "--output", type=Path, default=None,
        help="output base path (without extension). The literal token "
             "'<timestamp>' is replaced with YYYYMMDD-HHMMSS. If "
             "omitted, the report is written to stdout.")
    p_adv.add_argument(
        "--format", choices=("json", "md", "both"), default="both",
        help="report format(s) to emit (default: both).")

    p_cmp = sub.add_parser(
        "compare-stints",
        help="compare two captured stints (typically before/after a setup "
             "change) and write a JSON / Markdown diff report.")
    p_cmp.add_argument(
        "--car", required=True,
        help="LFS car short name used to resolve <CAR>_CAR_info.bin if "
             "no --baseline-* override is given.")
    p_cmp.add_argument(
        "--track", required=True,
        help="LFS track short code (e.g. BL1, AS3, KY2R).")
    p_cmp.add_argument(
        "--before", nargs="+", type=Path, required=True,
        metavar="LAP_CSV",
        help="per-lap CSVs for the BEFORE stint.")
    p_cmp.add_argument(
        "--after", nargs="+", type=Path, required=True,
        metavar="LAP_CSV",
        help="per-lap CSVs for the AFTER stint.")
    p_cmp.add_argument(
        "--baseline-before", type=Path, default=None,
        help="explicit <CAR>_CAR_info.bin captured at the time of the "
             "BEFORE stint.")
    p_cmp.add_argument(
        "--baseline-after", type=Path, default=None,
        help="explicit <CAR>_CAR_info.bin captured at the time of the "
             "AFTER stint.")
    p_cmp.add_argument(
        "--seed", type=int, default=20260516,
        help="deterministic seed for the network builders (default "
             "20260516).")
    p_cmp.add_argument(
        "--output", type=Path, default=None,
        help="output base path (without extension). '<timestamp>' is "
             "replaced with YYYYMMDD-HHMMSS. If omitted, stdout.")
    p_cmp.add_argument(
        "--format", choices=("json", "md", "both"), default="both",
        help="report format(s) to emit (default: both).")

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
            try:
                _signal.signal(_sig, _request_stop)
            except (ValueError, OSError):
                pass

    if args.cmd == "capture":
        return asyncio.run(_cmd_capture(args))
    if args.cmd == "calibrate":
        return asyncio.run(_cmd_calibrate(args))
    if args.cmd == "reslice":
        return _cmd_reslice(args)
    if args.cmd == "advise":
        return _cmd_advise(args)
    if args.cmd == "compare-stints":
        return _cmd_compare_stints(args)
    parser.error(f"unknown command: {args.cmd}")
    return 2


# ---------------------------------------------------------------------------


def _safe_tag(value: str | None) -> str:
    """Sanitize a free-form string so it is safe for filenames."""
    if not value:
        return "unknown"
    cleaned = "".join(c if c.isalnum() else "_" for c in str(value))
    return cleaned.strip("_") or "unknown"


def _session_tag(samples: list["TelemetrySample"]) -> str:
    """Build ``YYYYMMDD-HHMMSS_CAR_TRACK`` for unique per-lap filenames."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    car: str | None = None
    track: str | None = None
    for s in samples:
        if car is None and s.outgauge and s.outgauge.car:
            car = s.outgauge.car
        if track is None and s.race_context and s.race_context.track:
            track = s.race_context.track
        if car and track:
            break
    return f"{ts}_{_safe_tag(car)}_{_safe_tag(track)}"


async def _cmd_capture(args: argparse.Namespace) -> int:
    global _CAPTURE_LOOP, _CAPTURE_TASK
    _CAPTURE_LOOP = asyncio.get_running_loop()
    _CAPTURE_TASK = asyncio.current_task()
    samples: list[TelemetrySample] = []
    use_laps = args.laps > 0
    per_lap = args.per_lap
    # --include-out-lap means: keep every sample, no warmup skipping,
    # no out-lap trimming. The slicer will emit _lap00 for the out-lap.
    if getattr(args, "include_out_lap", False):
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

    def _close_lap(end_idx_exclusive: int, lap_ms: int | None) -> None:
        """Snapshot samples[start_idx:end_idx] as a completed flying lap."""
        nonlocal flying_lap_start_idx
        if not per_lap or not flying_lap_started:
            return
        slice_ = samples[flying_lap_start_idx:end_idx_exclusive]
        if slice_:
            lap_slices.append((completed_flying_laps + 1, list(slice_), lap_ms))
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
            insim_request_mci=live_file is not None,
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
                        print(
                            f"[capture] armed: car moving "
                            f"({speed * 3.6:.1f} km/h) — recording starts now",
                            file=sys.stderr,
                        )
                    else:
                        # Stationary in pits / on grid / countdown — drop.
                        continue

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

                if (use_laps or per_lap) and sample.race_context is not None:
                    rc = sample.race_context
                    # Pick the active PLID. Prefer the one LFS reports as
                    # "viewed". If view_plid hasn't propagated yet, fall back
                    # to whichever PLID is racking up laps.
                    if target_plid is None:
                        if rc.view_player_id is not None and rc.view_player_id in rc.players:
                            target_plid = rc.view_player_id
                            last_lap_count = rc.lap_count.get(target_plid, 0)
                            print(f"[capture] tracking PLID {target_plid} "
                                  f"(via view_plid, lap_count={last_lap_count})",
                                  file=sys.stderr)
                        elif rc.lap_count:
                            # First PLID with completed laps = our driver.
                            # IS_NPL never arrived (common): we discover the
                            # PLID only AFTER the first IS_LAP. That IS_LAP is
                            # almost certainly the out-lap completion we just
                            # missed — anchor flying laps starting now.
                            target_plid = max(rc.lap_count, key=rc.lap_count.get)
                            last_lap_count = rc.lap_count.get(target_plid, 0)
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
                        if not flying_lap_started:
                            # First lap completion = end of out-lap. Drop the
                            # in-progress lap from the buffer and re-anchor.
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
                                )
                                node_delta_value = (
                                    node_delta_tracker.delta_ms(
                                        node=view_car.node,
                                        elapsed_ms=cur_lap_ms,
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
                                for corner, w in zip(WHEEL_ORDER, os2.wheels)
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
            try:
                await stop_watcher_task
            except (asyncio.CancelledError, Exception):
                pass

    if (use_laps or per_lap) and not flying_lap_started:
        print("[capture] WARNING: no IS_LAP packet received. Drove past "
              "start/finish? In LFS use S1 lap mode, not /restart loops.",
              file=sys.stderr)

    _flush_capture(args, samples, per_lap)
    return 0


def _flush_capture(
    args: argparse.Namespace,
    samples: list[TelemetrySample],
    per_lap: bool,
) -> None:
    """Write per-lap and/or aggregate CSVs from buffered samples.

    Safe to call from both the normal completion path and from an
    interrupt handler: it never raises (errors are logged instead).
    """
    if not samples:
        print("[capture] no samples buffered, nothing to write.",
              file=sys.stderr)
        return
    try:
        if per_lap:
            out_dir = args.output.parent
            stem = args.output.stem
            suffix = args.output.suffix or ".csv"
            session_tag = _session_tag(samples)
            written = write_per_lap_files(
                samples,
                out_dir=out_dir,
                stem=stem,
                suffix=suffix,
                session_tag=session_tag,
                include_out_lap=getattr(args, "include_out_lap", False),
            )
            if not written:
                print("[capture] WARNING: no full lap recovered from buffer "
                      "(need at least 2 line crossings in current_lap_dist_m).",
                      file=sys.stderr)
            for path, lap, n in written:
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


# ---------------------------------------------------------------------------
# advise
# ---------------------------------------------------------------------------


def _cmd_advise(args: argparse.Namespace) -> int:
    """Run the TNFR Setup Advisor on a captured stint.

    Mirrors the Studio's Setup → Advisor tab so the CLI and the UI
    emit byte-identical JSON / Markdown reports for the same stint.
    """
    from .telemetry import LapTelemetry
    from .telemetry.car_info_bin import CarInfoBin, parse_car_info_bin
    from .telemetry.observables import load_car_info_bin_for
    from .tnfr_racing.advisor import SetupAdvisor
    from .tnfr_racing.serialize import result_to_json, result_to_markdown

    # --- 1. Load laps -------------------------------------------------
    lap_paths: list[Path] = [Path(p) for p in args.laps]
    missing = [p for p in lap_paths if not p.is_file()]
    if missing:
        for p in missing:
            print(f"[advise] missing lap CSV: {p}", file=sys.stderr)
        return 2
    try:
        laps = [LapTelemetry.from_csv(p, car=args.car) for p in lap_paths]
    except (OSError, ValueError) as exc:
        print(f"[advise] failed to load laps: {exc}", file=sys.stderr)
        return 1

    # --- 2. Resolve baseline -----------------------------------------
    baseline: CarInfoBin | None
    if args.baseline is not None:
        if not args.baseline.is_file():
            print(f"[advise] baseline file not found: {args.baseline}",
                  file=sys.stderr)
            return 2
        try:
            baseline = parse_car_info_bin(args.baseline)
        except (OSError, ValueError) as exc:
            print(f"[advise] failed to parse baseline {args.baseline}: "
                  f"{exc}", file=sys.stderr)
            return 1
    else:
        baseline = load_car_info_bin_for(args.car)
        if baseline is None:
            print(f"[advise] no <{args.car}>_CAR_info.bin found on the "
                  f"asset search path. Pass --baseline explicitly or set "
                  f"LFS_TELEMETRY_CAR_INFO_DIR.", file=sys.stderr)
            return 2

    # --- 3. Run advisor ----------------------------------------------
    advisor = SetupAdvisor(seed=int(args.seed))
    try:
        result = advisor.advise(laps, baseline, laps[0].car, args.track)
    except Exception as exc:  # noqa: BLE001 - surface as exit code
        print(f"[advise] advisor failed: {exc!r}", file=sys.stderr)
        return 1

    json_text = result_to_json(result)
    md_text = result_to_markdown(
        result,
        car_key=args.car.upper(),
        track_code=args.track.upper(),
        n_laps=len(laps),
        baseline=baseline,
    )

    # --- 4. Emit ------------------------------------------------------
    if args.output is None:
        if args.format in ("md", "both"):
            sys.stdout.write(md_text)
            if args.format == "both":
                sys.stdout.write("\n")
        if args.format in ("json", "both"):
            sys.stdout.write(json_text)
            sys.stdout.write("\n")
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_str = str(args.output).replace("<timestamp>", ts)
        base = Path(base_str)
        # Strip any caller-supplied extension so --format controls output.
        if base.suffix.lower() in (".json", ".md", ".markdown"):
            base = base.with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        if args.format in ("json", "both"):
            p_json = base.with_suffix(".json")
            p_json.write_text(json_text, encoding="utf-8")
            written.append(p_json)
        if args.format in ("md", "both"):
            p_md = base.with_suffix(".md")
            p_md.write_text(md_text, encoding="utf-8")
            written.append(p_md)
        for p in written:
            print(f"[advise] wrote {p}")

    # Exit code 0 even on a refusal: the user got a structured report.
    return 0


# ---------------------------------------------------------------------------
# compare-stints (Phase 9.A)
# ---------------------------------------------------------------------------


def _cmd_compare_stints(args: argparse.Namespace) -> int:
    """Compare two captured stints (before/after a setup change)."""
    from .telemetry import LapTelemetry
    from .telemetry.car_info_bin import CarInfoBin, parse_car_info_bin
    from .telemetry.observables import load_car_info_bin_for
    from .tnfr_racing.multi_stint import compare_stints
    from .tnfr_racing.serialize import (
        comparison_to_json, comparison_to_markdown,
    )

    def _load(paths: list[Path], label: str) -> list[LapTelemetry] | int:
        missing = [p for p in paths if not p.is_file()]
        if missing:
            for p in missing:
                print(f"[compare] missing {label} lap CSV: {p}",
                      file=sys.stderr)
            return 2
        try:
            return [LapTelemetry.from_csv(p, car=args.car) for p in paths]
        except (OSError, ValueError) as exc:
            print(f"[compare] failed to load {label} laps: {exc}",
                  file=sys.stderr)
            return 1

    before_laps = _load([Path(p) for p in args.before], "before")
    if isinstance(before_laps, int):
        return before_laps
    after_laps = _load([Path(p) for p in args.after], "after")
    if isinstance(after_laps, int):
        return after_laps

    def _resolve_baseline(path: Path | None) -> CarInfoBin | int:
        if path is not None:
            if not path.is_file():
                print(f"[compare] baseline file not found: {path}",
                      file=sys.stderr)
                return 2
            try:
                return parse_car_info_bin(path)
            except (OSError, ValueError) as exc:
                print(f"[compare] failed to parse baseline {path}: {exc}",
                      file=sys.stderr)
                return 1
        b = load_car_info_bin_for(args.car)
        if b is None:
            print(f"[compare] no <{args.car}>_CAR_info.bin found. Pass "
                  f"--baseline-before / --baseline-after explicitly.",
                  file=sys.stderr)
            return 2
        return b

    bl_before = _resolve_baseline(args.baseline_before)
    if isinstance(bl_before, int):
        return bl_before
    bl_after = _resolve_baseline(args.baseline_after)
    if isinstance(bl_after, int):
        return bl_after

    try:
        cmp = compare_stints(
            before_laps=before_laps,
            after_laps=after_laps,
            baseline_before=bl_before,
            baseline_after=bl_after,
            car=before_laps[0].car,
            track_code=args.track,
            seed=int(args.seed),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[compare] comparison failed: {exc!r}", file=sys.stderr)
        return 1

    json_text = comparison_to_json(cmp)
    md_text = comparison_to_markdown(
        cmp, car_key=args.car.upper(), track_code=args.track.upper(),
    )

    if args.output is None:
        if args.format in ("md", "both"):
            sys.stdout.write(md_text)
            if args.format == "both":
                sys.stdout.write("\n")
        if args.format in ("json", "both"):
            sys.stdout.write(json_text)
            sys.stdout.write("\n")
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_str = str(args.output).replace("<timestamp>", ts)
        base = Path(base_str)
        if base.suffix.lower() in (".json", ".md", ".markdown"):
            base = base.with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        if args.format in ("json", "both"):
            p_json = base.with_suffix(".json")
            p_json.write_text(json_text, encoding="utf-8")
            written.append(p_json)
        if args.format in ("md", "both"):
            p_md = base.with_suffix(".md")
            p_md.write_text(md_text, encoding="utf-8")
            written.append(p_md)
        for p in written:
            print(f"[compare] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
