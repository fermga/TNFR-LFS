"""``lfs-telemetry raf-import`` subcommand."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


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

