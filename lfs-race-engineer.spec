# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for **LFS Race Engineer** (PySide6 Studio).

Produces a one-folder distributable at ``dist/lfs-race-engineer/``:

    dist/lfs-race-engineer/
        lfs-race-engineer.exe
        _internal/...           (Python runtime, PySide6, scipy, ...)
        config/cars.json
        racing_lines/*.csv
        tracks/*.csv

Build with:

    pip install -e ".[studio,build]"
    pyinstaller lfs-race-engineer.spec --noconfirm --clean

Or via:

    .\\scripts\\build_app.ps1
"""
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

ROOT = Path(SPECPATH).resolve()  # noqa: F821 - SPECPATH provided by PyInstaller


# --- Hidden imports & data ----------------------------------------------------

hiddenimports: list[str] = []
datas: list[tuple[str, str]] = []

# Pull every Studio + telemetry submodule so dynamic imports survive.
hiddenimports += collect_submodules("lfs_telemetry")

# The TNFR Setup Advisor pulls modules from the external `tnfr` package
# under runtime (operator factories, dynamics helpers). `collect_submodules`
# guarantees the frozen build keeps every submodule even if the static
# importer of `lfs_telemetry.tnfr_racing.*` only references a subset.
hiddenimports += collect_submodules("tnfr")

# pyqtgraph occasionally needs runtime templates that hooks miss.
hiddenimports += collect_submodules("pyqtgraph")

# scipy lazy submodules (signal.windows, special, ndimage) used by the
# track and lap analysis pipelines.
hiddenimports += collect_submodules("scipy")

# pandas needs openpyxl metadata if the user ever exports to xlsx via
# Studio.  Optional, but cheap.
try:
    datas += copy_metadata("openpyxl")
except Exception:
    pass

# Bundle our static data folders side-by-side with the .exe (NOT inside
# _internal/) so the app's cwd-based lookups keep working.  The runtime
# hook chdir's to the .exe directory at startup.
def _bundle_dir(folder: str, pattern: str = "*") -> list[tuple[str, str]]:
    src = ROOT / folder
    if not src.exists():
        return []
    return [(str(p), folder) for p in src.glob(pattern) if p.is_file()]


datas += _bundle_dir("config", "*.json")
datas += _bundle_dir("racing_lines", "*.csv")
datas += _bundle_dir("tracks", "*.csv")
# Mod-car footprint database (seeded from Detect&Monitor) — keeps the
# radar usable for opponents driving LFS mods.
datas += _bundle_dir("assets/source/mods", "*.json")
# Stock CAR_info.bin exports the user has dropped under
# assets/source/cars/. The directory may be empty at build time; the
# Studio "Import CAR_info.bin…" button writes new exports there at
# runtime when launched from a writable working directory.
datas += _bundle_dir("assets/source/cars", "*.bin")
# Track overview .pngs are nice-to-have; uncomment to ship.
# datas += _bundle_dir("tracks", "*.png")


# --- Analysis -----------------------------------------------------------------

a = Analysis(
    [str(ROOT / "src" / "lfs_telemetry" / "studio" / "__main__.py")],
    pathex=[
        str(ROOT / "src"),
    ],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "scripts" / "pyi_runtime_chdir.py")],
    excludes=[
        # Test-only deps.
        "pytest",
        "pytest_asyncio",
        "_pytest",
        # Dash viewer is shipped as a separate CLI; not needed in the GUI.
        "dash",
        "plotly",
        "flask",
        "werkzeug",
        # Heavy optional libs we never import.
        "tkinter",
        "matplotlib",
        "PyQt5",
        "PyQt6",
        "PySide2",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="lfs-race-engineer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                 # windowed app: no terminal flashes up
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "icon.ico") if (ROOT / "assets" / "icon.ico").exists() else None,
    # PyInstaller >=6 defaults to nesting every collected file under
    # ``_internal/``; the Studio runtime (``_asset_search_dirs``) and the
    # Inno Setup script in ``installer/lfs-race-engineer.iss`` both
    # expect ``config/``, ``racing_lines/`` and ``tracks/`` sitting
    # side-by-side with the .exe. ``contents_directory='.'`` restores
    # the pre-6 flat layout so both paths keep resolving.
    contents_directory=".",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="lfs-race-engineer",
)
