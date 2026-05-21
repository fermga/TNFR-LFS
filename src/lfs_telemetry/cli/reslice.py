"""``lfs-telemetry reslice`` subcommand."""
from __future__ import annotations

import argparse
import sys

from ..telemetry.lap_slicer import reslice_csv


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

