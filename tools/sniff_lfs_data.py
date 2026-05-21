"""Quick header sniffer for LFS data files."""
from pathlib import Path

paths = [
    r"C:\LFS\data\knw\AS1_BF1.knw",
    r"C:\LFS\data\knw\AS1_FBM.knw",
    r"C:\LFS\data\knw\BL1_FBM.knw",
    r"C:\LFS\data\knw\BL1_FOX.knw",
    r"C:\LFS\data\wld\BLACKWOOD.lgh",
    r"C:\LFS\data\wld\BLACKWOOD.wld",
    r"C:\LFS\data\wld\BL.lok",
    r"C:\LFS\data\wld\AS.lok",
    r"C:\LFS\data\grids\cfg.rac",
    r"C:\LFS\data\layout\AU1_LX_week1.lyt",
    r"C:\LFS\data\hmn\hmn.hmn",
    r"C:\LFS\data\abc\lfs.abc",
]
for p in paths:
    pp = Path(p)
    if not pp.exists():
        print(f"--- MISSING: {p}")
        continue
    data = pp.read_bytes()[:96]
    hex_ = " ".join(f"{b:02X}" for b in data)
    asc = "".join(chr(b) if 32 <= b < 127 else "." for b in data)
    print(f"=== {p}  (size {pp.stat().st_size}) ===")
    print(f"HEX {hex_}")
    print(f"ASC {asc}")
    print()
