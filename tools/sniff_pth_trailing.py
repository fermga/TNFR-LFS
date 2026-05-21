"""Empirical sniff of the 4 trailing floats per PTH node.

Goal: figure out whether the 16 bytes at node offset 28..43 encode
banking (Z components of the lateral edges) or just scalar widths.
We dump a handful of nodes per track for several tracks of different
character (banked vs flat, wide vs narrow).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

SMX = Path(r"C:\LFS\data\smx")
HEADER_BYTES = 56
NODE_BYTES = 44

TARGETS = [
    ("AS1", "Aston Cadet (has banked bowl)"),
    ("AS2", "Aston Club"),
    ("BL1", "Blackwood GP (almost flat)"),
    ("KY1", "Kyoto Oval (heavily banked)"),
    ("KY3", "Kyoto GP Long"),
    ("WE1", "Westhill National"),
]


def dump(name: str, label: str, n_samples: int = 6) -> None:
    p = SMX / f"{name}.pth"
    if not p.exists():
        print(f"  -- {name}: missing")
        return
    data = p.read_bytes()
    body = data[HEADER_BYTES:]
    n_nodes = len(body) // NODE_BYTES
    print(f"\n=== {name}  ({label})  nodes={n_nodes} ===")
    # Sample evenly.
    idxs = np.linspace(0, n_nodes - 1, n_samples).astype(int)
    print("  idx | center XYZ                 | dir XYZ                | f4 f5 f6 f7")
    for i in idxs:
        off = i * NODE_BYTES
        _flags, ix, iy, iz = struct.unpack_from("<i3i", body, off)
        dx, dy, dz = struct.unpack_from("<3f", body, off + 16)
        f4, f5, f6, f7 = struct.unpack_from("<4f", body, off + 28)
        cx, cy, cz = ix / 65536.0, iy / 65536.0, iz / 65536.0
        print(f"  {i:>4} | ({cx:+8.2f},{cy:+8.2f},{cz:+6.2f}) "
              f"| ({dx:+.3f},{dy:+.3f},{dz:+.3f}) "
              f"| {f4:+7.3f} {f5:+7.3f} {f6:+7.3f} {f7:+7.3f}")

    # Aggregate stats on f4..f7 across all nodes.
    f4s, f5s, f6s, f7s = [], [], [], []
    for i in range(n_nodes):
        off = i * NODE_BYTES
        a, b, c, d = struct.unpack_from("<4f", body, off + 28)
        f4s.append(a); f5s.append(b); f6s.append(c); f7s.append(d)
    a = np.array(f4s); b = np.array(f5s); c = np.array(f6s); d = np.array(f7s)
    print(f"  stats: f4 [{a.min():+.2f},{a.max():+.2f}] mean {a.mean():+.2f}"
          f"  | f5 [{b.min():+.2f},{b.max():+.2f}] mean {b.mean():+.2f}"
          f"  | f6 [{c.min():+.2f},{c.max():+.2f}] mean {c.mean():+.2f}"
          f"  | f7 [{d.min():+.2f},{d.max():+.2f}] mean {d.mean():+.2f}")
    # Sign sanity: are f4/f6 mostly negative (left) and f5/f7 positive (right)?
    print(f"  signs: f4<0:{(a<0).mean():.0%}  f5>0:{(b>0).mean():.0%}  "
          f"f6<0:{(c<0).mean():.0%}  f7>0:{(d>0).mean():.0%}")


def main() -> None:
    for name, label in TARGETS:
        dump(name, label)


if __name__ == "__main__":
    main()
