"""PyInstaller runtime hook: pin cwd to the .exe folder.

Studio resolves data files (``config/cars.json``, ``racing_lines/``,
``tracks/``) relative to the current working directory.  When the user
double-clicks the installed shortcut Windows sets cwd to ``%USERPROFILE%``
or similar, which obviously does not contain those folders.  Forcing cwd
to the directory holding ``lfs-race-engineer.exe`` keeps the lookup
logic working without touching the application code.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    try:
        os.chdir(Path(sys.executable).resolve().parent)
    except OSError:
        # Best-effort: do not block startup if chdir is denied.
        pass
