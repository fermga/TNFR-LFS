"""Exploratory decoder for LFS .knw (AI knowledge) files.

Hypothesis: header 12 B (magic 6 + version u16 + 4 B) then N records of 24 B.
Cross-check across multiple files to identify the record schema.
"""
from __future__ import annotations

import struct
from pathlib import Path

KNW_DIR = Path(r"C:\LFS\data\knw")
MAGIC = b"LFSKNW"


def split_records(path: Path, hdr: int = 12, rec: int = 24):
    data = path.read_bytes()
    assert data[:6] == MAGIC, path
    ver = struct.unpack_from("<H", data, 6)[0]
    pad = data[8:hdr]
    body = data[hdr:]
    n_full, rem = divmod(len(body), rec)
    return ver, pad, n_full, rem, [body[i*rec:(i+1)*rec] for i in range(n_full)]


def decode_record(b: bytes) -> dict:
    """Try a couple of plausible decodings."""
    as_u32 = struct.unpack("<6I", b)
    as_f32 = struct.unpack("<6f", b)
    as_mix_uffff_u = struct.unpack("<I4fI", b)   # (u32, f×4, u32)
    as_mix_ufuuff = struct.unpack("<IfIIff", b)
    return {
        "u32x6": as_u32,
        "f32x6": as_f32,
        "u_ffff_u": as_mix_uffff_u,
        "u_f_uu_ff": as_mix_ufuuff,
    }


def dump(path: Path, *, max_rec: int = 6):
    ver, pad, n, rem, recs = split_records(path)
    print(f"\n=== {path.name} ({path.stat().st_size} B, ver={ver}, header_pad={pad.hex()}, "
          f"records={n}, leftover={rem}) ===")
    if rem != 0:
        # Try other header sizes
        for hdr in (8, 16, 20, 24, 28, 32):
            _, _, n2, rem2, _ = split_records(path, hdr=hdr)
            print(f"  alt hdr={hdr}: n={n2}, leftover={rem2}")
    for i, r in enumerate(recs[:max_rec]):
        dec = decode_record(r)
        print(f"  [{i:2d}] hex={r.hex(' ')}")
        for k, v in dec.items():
            shown = tuple(round(x, 4) if isinstance(x, float) else x for x in v)
            print(f"        {k:12s} {shown}")


def cross_field_summary(files: list[Path], field_idx: int, fmt: str):
    """Show value of one field across many files (to find arc-length etc.)."""
    print(f"\n--- field idx {field_idx} as {fmt} across {len(files)} files ---")
    for p in files:
        _, _, _, _, recs = split_records(p)
        size = struct.calcsize(fmt)
        offset = field_idx * size
        vals = [struct.unpack_from("<" + fmt, r, offset)[0] for r in recs]
        if not vals:
            continue
        fvals = [v if not isinstance(v, float) else round(v, 3) for v in vals]
        print(f"  {p.name:20s} n={len(vals):3d} "
              f"min={min(fvals)!s:>14s} max={max(fvals)!s:>14s} "
              f"first6={fvals[:6]}")


if __name__ == "__main__":
    samples = [
        KNW_DIR / "AS1_BF1.knw",
        KNW_DIR / "AS1_FBM.knw",
        KNW_DIR / "AS1_FXR.knw",
        KNW_DIR / "BL1_FBM.knw",
        KNW_DIR / "BL1_FXR.knw",
        KNW_DIR / "BL1_BF1.knw",
        KNW_DIR / "BL2_FBM.knw",
        KNW_DIR / "FE1_FBM.knw",
        KNW_DIR / "KY1_FBM.knw",
    ]
    samples = [p for p in samples if p.exists()]
    for p in samples:
        dump(p, max_rec=4)

    # Try to detect which field is monotonic per file (likely arc-length s).
    print("\n\n############ monotonic-field scan ############")
    for fmt in ("f", "I"):
        for idx in range(6):
            cross_field_summary(samples[:4], idx, fmt)
