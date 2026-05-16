"""Reconnaissance dumper for ``BLACKWOOD.wld`` (SRWORL).

The file is the only SRWORL in the canonical LFS install (`C:\\LFS\\data\\wld`).
We want to know whether it contains the static surface mesh (vertex
positions + triangle indices) which would let us recover real banking
and surface elevation per (x, y), complementing the PTH centerline.

Strategy
--------
1. Dump the header (first 256 B), then bulk-statistic the body to look
   for repeating record sizes (likely 12, 16, 24, 32, 36 bytes for
   vertex/triangle blocks).
2. Locate ASCII tokens (object names, material names, "ROAD",
   "GRASS", etc.) and report their offsets — they typically delimit
   sub-chunks in LFS resource files.
3. Sample a few f32 triplets and check whether they form plausible
   world-XYZ coordinates inside the BLACKWOOD bbox known from PIN.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

DEFAULT_WLD = Path(r"C:\LFS\data\wld\BLACKWOOD.wld")
MAGIC = b"SRWORL"


def hex_dump(data: bytes, start: int, length: int) -> str:
    end = min(start + length, len(data))
    lines: list[str] = []
    for off in range(start, end, 16):
        chunk = data[off:off + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(
            chr(b) if 0x20 <= b <= 0x7e else "." for b in chunk
        )
        lines.append(f"{off:08x}  {hex_part:<47}  {ascii_part}")
    return "\n".join(lines)


def find_ascii_tokens(data: bytes, *, min_len: int = 4,
                      max_report: int = 80) -> list[tuple[int, str]]:
    """Find printable-ASCII runs (≥ ``min_len`` chars)."""
    tokens: list[tuple[int, str]] = []
    cur_start = -1
    cur: list[int] = []
    for i, b in enumerate(data):
        if 0x20 <= b <= 0x7e:
            if cur_start < 0:
                cur_start = i
            cur.append(b)
        else:
            if cur_start >= 0 and len(cur) >= min_len:
                tokens.append((cur_start, bytes(cur).decode("ascii", "replace")))
            cur_start = -1
            cur = []
    if cur_start >= 0 and len(cur) >= min_len:
        tokens.append((cur_start, bytes(cur).decode("ascii", "replace")))
    return tokens[:max_report]


def scan_f32_triplets(data: bytes, *, max_report: int = 20,
                      bbox=(-2000, -2000, -200, 2000, 2000, 500)) -> list:
    """Walk the file looking for ranges where consecutive f32 triplets
    look like XYZ coordinates inside ``bbox``."""
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    found: list[tuple[int, int, float, float, float]] = []
    i = 12  # skip magic + flags
    n = len(data) - 36
    run_start = -1
    run_count = 0
    while i < n:
        try:
            x, y, z = struct.unpack_from("<fff", data, i)
        except struct.error:
            break
        ok = (xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax
              and not any(v != v for v in (x, y, z)))
        if ok:
            if run_start < 0:
                run_start = i
                run_count = 1
            else:
                run_count += 1
            i += 12   # advance one triplet
        else:
            if run_count >= 32:
                # Close a long run, record (start, count, last triplet).
                found.append((run_start, run_count, x, y, z))
                if len(found) >= max_report:
                    break
            run_start = -1
            run_count = 0
            i += 4    # advance one float to keep alignment lossless
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path, default=DEFAULT_WLD)
    ap.add_argument("--header-bytes", type=int, default=256)
    args = ap.parse_args(argv)

    if not args.path.exists():
        print(f"missing: {args.path}")
        return 2

    data = args.path.read_bytes()
    print(f"file = {args.path}  ({len(data):,} B)")
    if data[:6] != MAGIC:
        print(f"WARNING: magic mismatch ({data[:6]!r} != {MAGIC!r})")
    print()
    print("=== header dump ===")
    print(hex_dump(data, 0, args.header_bytes))
    print()

    print("=== ascii tokens (>=4 chars, first 80) ===")
    for off, tok in find_ascii_tokens(data):
        # Filter the obvious magic.
        if tok.startswith("SRWORL"):
            continue
        print(f"  {off:08x}  {tok!r}")
    print()

    print("=== XYZ-triplet runs inside BLACKWOOD bbox (first 20) ===")
    runs = scan_f32_triplets(data)
    for start, count, x, y, z in runs:
        print(f"  start={start:08x}  count={count:>6}  "
              f"last=({x:+8.2f},{y:+8.2f},{z:+7.2f})  "
              f"end={start + count*12:08x}")
    if runs:
        # Estimate vertex-block total length: sum of long runs (≥ 200 triplets).
        long_runs = [(s, c) for s, c, *_ in runs if c >= 200]
        total_triplets = sum(c for _, c in long_runs)
    print("  long runs (>=200 triplets):", len(long_runs),
          "totaling", total_triplets, "XYZ points")
    print()

    # ---- 36-byte record sweep ---------------------------------------
    body_start = 0x70
    body = data[body_start:]
    record_size = 36
    n_records = len(body) // record_size
    print(f"=== 36-byte record sweep: body={len(body):,} B "
          f"=> {n_records} records (remainder {len(body) % record_size} B) ===")

    # Sample first 5 and last 2 records.
    def dump_rec(idx: int) -> None:
        off = body_start + idx * record_size
        rec = data[off:off + record_size]
        u0, u1, u2 = struct.unpack_from("<III", rec, 0)
        f0, f1, f2 = struct.unpack_from("<fff", rec, 12)
        f3, f4, f5 = struct.unpack_from("<fff", rec, 24)
        print(f"  rec#{idx:>6}  @{off:08x}"
              f"  ids=({u0:08x},{u1:08x},{u2:08x})"
              f"  f012=({f0:+.4f},{f1:+.4f},{f2:+.4f})"
              f"  XYZ=({f3:+9.3f},{f4:+9.3f},{f5:+9.3f})")

    for idx in (0, 1, 2, 3, 4, 1000, 10000, 100000, n_records - 2, n_records - 1):
        if 0 <= idx < n_records:
            dump_rec(idx)
    print()

    # Bounding box on the XYZ field of every record.
    import math
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    f0s: list[float] = []
    f1s: list[float] = []
    f2s: list[float] = []
    plausible = 0
    for idx in range(n_records):
        off = body_start + idx * record_size
        try:
            f0, f1, f2, fx, fy, fz = struct.unpack_from("<ffffff", data, off + 12)
        except struct.error:
            break
        if any(math.isnan(v) or math.isinf(v) for v in (fx, fy, fz)):
            continue
        if -3000 < fx < 3000 and -3000 < fy < 3000 and -500 < fz < 500:
            xs.append(fx); ys.append(fy); zs.append(fz)
            f0s.append(f0); f1s.append(f1); f2s.append(f2)
            plausible += 1

    if plausible > 0:
        print(f"=== XYZ stats across {plausible} of {n_records} records (assuming 36-B layout) ===")
        def s(name: str, v: list[float]) -> None:
            print(f"  {name}: min={min(v):+9.3f}  max={max(v):+9.3f}  "
                  f"mean={sum(v)/len(v):+9.3f}")
        s("X     ", xs)
        s("Y     ", ys)
        s("Z     ", zs)
        s("field0", f0s)
        s("field1", f1s)
        s("field2", f2s)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
