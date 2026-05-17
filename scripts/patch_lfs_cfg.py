"""Patch LFS ``cfg.txt`` with the settings LFS-Telemetry needs.

Writes the OutSim/OutGauge/InSim entries required for telemetry capture.
Idempotent: existing keys are updated in place; missing keys are appended.
Original file is backed up to cfg.txt.bak (only the first time).

Run with LFS CLOSED::

    .\\.venv\\Scripts\\python.exe scripts\\patch_lfs_cfg.py [LFS folder]

If no folder is passed, the script auto-detects common LFS install
locations (C:\\LFS, C:\\Program Files\\LFS, ...).
"""
from __future__ import annotations

import sys
from pathlib import Path

from lfs_telemetry.lfs_config import patch_cfg
from lfs_telemetry.lfs_paths import autodetect_lfs_dir


def main() -> int:
    if len(sys.argv) >= 2:
        lfs_dir = Path(sys.argv[1])
    else:
        guess = autodetect_lfs_dir()
        if guess is None:
            print(
                "Could not auto-detect LFS. "
                "Pass the install folder explicitly:\n"
                "    python scripts\\patch_lfs_cfg.py C:\\path\\to\\LFS",
            )
            return 1
        lfs_dir = guess
        print(f"Using auto-detected LFS folder: {lfs_dir}")

    try:
        result = patch_cfg(lfs_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(result.summary_text())
    print()
    print("Done. Now launch LFS and enter a session.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
