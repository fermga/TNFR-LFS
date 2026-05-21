"""Command-line interface for lfs-telemetry.

Subcommands:
    capture          Listen to LFS UDP and write a CSV stint capture.
    calibrate        Auto-measure car mass + weight distribution from rest.
    reslice          Re-slice an aggregate capture CSV into per-lap CSVs.
    raf-import       Convert an LFS RAF replay file into per-lap CSVs.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path

from ._common import (
    _add_lfs_flags,
    _harden_std_streams,
    _request_stop,
    _ResilientTextStream,
)
from .calibrate import _cmd_calibrate
from .capture import _cmd_capture
from .raf_import import _cmd_raf_import
from .reslice import _cmd_reslice

__all__ = ["_ResilientTextStream", "main"]


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



if __name__ == "__main__":
    sys.exit(main())
