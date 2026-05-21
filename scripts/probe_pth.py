"""Cross-track PTH probe — verify (size-56)/44 = integer for many tracks."""
import glob
import os

DIR = r"C:\LFS\data\smx"
print(f"{'file':14} {'size':>8} {'(size-56)/44':>15} {'ok':>4}")
for f in sorted(glob.glob(os.path.join(DIR, "*.pth"))):
    sz = os.path.getsize(f)
    nodes = (sz - 56) / 44
    ok = "OK" if nodes == int(nodes) else "NO"
    print(f"{os.path.basename(f):14} {sz:8} {nodes:15.3f} {ok:>4}")
