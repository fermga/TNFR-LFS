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
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

ROOT = Path(SPECPATH).resolve()  # noqa: F821 - SPECPATH provided by PyInstaller


# --- Hidden imports & data ----------------------------------------------------

hiddenimports: list[str] = []
datas: list[tuple[str, str]] = []
binaries: list[tuple[str, str]] = []

# Modules deliberately *not* shipped in the installable build. The
# Setup tab and its in-app editor are unwired from ``CenterTabs`` (see
# ``src/lfs_telemetry/studio/README.md``) and must not travel inside
# the .exe until they are ready for end users.
_EXCLUDED_STUDIO_MODULES = {
    "lfs_telemetry.studio.widgets.setup_tab",
    "lfs_telemetry.studio.widgets.setup_editor_tab",
}

# Pull every Studio + telemetry submodule so dynamic imports survive,
# minus the explicitly hidden modules above.
hiddenimports += [
    name
    for name in collect_submodules("lfs_telemetry")
    if name not in _EXCLUDED_STUDIO_MODULES
]

# pyqtgraph occasionally needs runtime templates that hooks miss.
# Skip the optional ``pyqtgraph.opengl`` (PyOpenGL backend) and
# ``pyqtgraph.jupyter`` (jupyter_rfb widget) subtrees: neither
# dependency is installed and we use only the 2D PySide6 backend.
hiddenimports += collect_submodules(
    "pyqtgraph",
    filter=lambda name: not name.startswith(
        ("pyqtgraph.opengl", "pyqtgraph.jupyter")
    ),
)

# scipy lazy submodules (signal.windows, special, ndimage) used by the
# track and lap analysis pipelines. Skip optional array-API backends
# (``cupy``/``torch``/``dask``) whose top-level packages we never
# install — the walker would try to import them and warn — and the
# legacy ``scipy.special._cdflib`` shim that no longer exists as an
# importable module on scipy>=1.13.
_SCIPY_SKIP_PREFIXES = (
    "scipy._lib.array_api_compat.cupy",
    "scipy._lib.array_api_compat.torch",
    "scipy._lib.array_api_compat.dask",
)
_SCIPY_SKIP_EXACT = {"scipy.special._cdflib"}
hiddenimports += collect_submodules(
    "scipy",
    filter=lambda name: (
        name not in _SCIPY_SKIP_EXACT
        and not name.startswith(_SCIPY_SKIP_PREFIXES)
    ),
)

# pandas needs openpyxl metadata if the user ever exports to xlsx via
# Studio.  Optional, but cheap.
try:
    datas += copy_metadata("openpyxl")
except Exception:
    pass

# Optional VR mirror (SteamVR / OpenVR). The ``openvr`` Python package
# ships a native ``openvr_api.dll`` next to its module; PyInstaller's
# default analysis misses it because we only import the package
# lazily inside ``studio/vr/openvr_overlay.py``.  When the build env
# has the ``[vr]`` extra installed, bundle the module + its DLL so the
# shipped .exe can mirror overlays to SteamVR.  Without ``openvr`` the
# build still succeeds; the runtime will simply report VR unavailable.
try:
    import openvr  # noqa: F401  (presence check)
    hiddenimports += collect_submodules("openvr")
    binaries += collect_dynamic_libs("openvr")
    try:
        datas += collect_data_files("openvr", include_py_files=False)
    except Exception:
        pass
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
# User manual (English + Spanish). Exposed in-app via Help → User manual.
datas += _bundle_dir("docs", "*.md")
# Mod-car footprint database (seeded from Detect&Monitor) — keeps the
# radar usable for opponents driving LFS mods.
datas += _bundle_dir("assets/source/mods", "*.json")
# Stock CAR_info.bin exports the user has dropped under
# assets/source/cars/. The directory may be empty at build time; the
# Studio "Import CAR_info.bin…" button writes new exports there at
# runtime when launched from a writable working directory.
datas += _bundle_dir("assets/source/cars", "*.bin")
# Per-environment top-down track images (used by the Track map dock
# overlay in the Studio "Map" panel). Without these, the "Track image"
# checkbox renders nothing because the candidate-dir search never finds
# ``<ENV>.tif`` in the installed bundle.
datas += _bundle_dir("assets/tracks", "*.tif")
datas += _bundle_dir("assets/tracks", "*.png")
# Track overview .pngs are nice-to-have; uncomment to ship.
# datas += _bundle_dir("tracks", "*.png")


# --- Analysis -----------------------------------------------------------------

a = Analysis(
    [str(ROOT / "src" / "lfs_telemetry" / "studio" / "__main__.py")],
    pathex=[
        str(ROOT / "src"),
    ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(ROOT / "scripts" / "pyi_hooks")],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "scripts" / "pyi_runtime_chdir.py")],
    excludes=[
        # Setup tab + in-app garage editor are not wired into the
        # Studio UI yet (see studio/README.md). Keep them out of the
        # installable build so they cannot be loaded at runtime.
        "lfs_telemetry.studio.widgets.setup_tab",
        "lfs_telemetry.studio.widgets.setup_editor_tab",
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
        # ML / data-science deps that some env-installed packages
        # (sentence-transformers, datasets, streamlit, jax) pull into
        # the venv but the Studio runtime never imports. Without these
        # excludes PyInstaller collects ~3.6 GB of CUDA DLLs from torch
        # plus jaxlib/pyarrow/torchvision/torchaudio (see build_log).
        "torch",
        "torchvision",
        "torchaudio",
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
        "tokenizers",
        "safetensors",
        "jax",
        "jaxlib",
        "pyarrow",
        "datasets",
        "streamlit",
        "tensorflow",
        "sklearn",
        "scikit_learn",
        "cupy",
        # Optional OpenGL backend for pyqtgraph — we only use 2D plots,
        # so neither PyOpenGL nor ``pyqtgraph.opengl`` is needed. Listing
        # them silences the "Failed to collect submodules" warning that
        # ``collect_submodules('pyqtgraph')`` would otherwise emit.
        "OpenGL",
        "OpenGL_accelerate",
        "pyqtgraph.opengl",
        # Removed legacy shim on scipy>=1.13. The bundled scipy hook
        # still declares it as a hidden import, which trips PyInstaller
        # with "Hidden import not found!". Listing it here makes the
        # resolver skip it silently.
        "scipy.special._cdflib",
        # Numerical/data ecosystem we never import — these only land in
        # the bundle when the build env has user-site contamination
        # (e.g. global ``pip install numba``). The Studio runtime stays
        # on plain numpy/scipy/pandas, so excluding them is safe and
        # shaves 100s of MB from the dist.
        "numba",
        "llvmlite",
        "dask",
        "xarray",
        "netCDF4",
        "cftime",
        "statsmodels",
        "sqlalchemy",
        "asyncpg",
        "aiohttp",
        "lxml",
        "tornado",
        "zmq",
        # Dev / notebook / docs tooling. Never imported at runtime.
        "IPython",
        "jedi",
        "parso",
        "black",
        "blib2to3",
        "sphinx",
        "alabaster",
        "docutils",
        "nbformat",
        "jsonschema",
        "jsonschema_specifications",
        "coverage",
        "cryptography",
        # Audio/game libs that leak via transitive installs.
        "pygame",
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
