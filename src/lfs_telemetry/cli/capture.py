"""``lfs-telemetry capture`` subcommand."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
from pathlib import Path

from ..telemetry import live_publisher
from ..telemetry.constants import SPEED_MS_TO_KMH
from ..telemetry.fuel_tracker import FuelTracker
from ..telemetry.lap_slicer import write_per_lap_files
from ..telemetry.live import LiveTelemetry, TelemetrySample
from ..telemetry.node_delta import NodeDeltaTracker
from ..telemetry.predict import SplitPredictor
from ..telemetry.protocol.packets import DL_PITSPEED, WHEEL_ORDER
from ..telemetry.replay import write_csv_replay
from ..telemetry.session_naming import session_tag as _session_tag
from ..telemetry.track.loader import find_racing_line_csv
from . import _state


async def _cmd_capture(args: argparse.Namespace) -> int:
    _state.CAPTURE_LOOP = asyncio.get_running_loop()
    _state.CAPTURE_TASK = asyncio.current_task()
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
        except BaseException as exc:
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
                    while not _state.STOP_REQUESTED:
                        try:
                            if path.exists():  # noqa: ASYNC240 — cheap stat in polling loop
                                _state.STOP_REQUESTED = True
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
                            f"({speed * SPEED_MS_TO_KMH:.1f} km/h) — recording starts now",
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
                            except Exception as exc:
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
                        except Exception as exc:
                            print(
                                f"[capture] live snapshot write failed: "
                                f"{type(exc).__name__}: {exc}",
                                file=sys.stderr,
                            )
                if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                    break
                if _state.STOP_REQUESTED:
                    print("[capture] stop flag set, ending capture loop",
                          file=sys.stderr)
                    break
    except KeyboardInterrupt:
        print("[capture] stopped by user", file=sys.stderr)
    except asyncio.CancelledError:
        print("[capture] stopped by user (cancelled)", file=sys.stderr)
    except BaseException as exc:
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
    except BaseException as exc:
        print(f"[capture] flush failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)

