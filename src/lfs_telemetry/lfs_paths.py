"""Single source of truth for the user's LFS install folder.

Stores the chosen path in ``QSettings`` under ``lfs/install_dir`` so the
configure-LFS dialog, the bin importer, the live telemetry pipeline and
the first-run wizard all agree on which folder to read/write.

The module also knows how to probe sensible defaults so the user rarely
has to type a path by hand: it consults the previously saved value, the
Windows registry key LFS itself writes (``HKCU\\Software\\Live for
Speed``) and a hand-picked list of common install locations. LFS is
distributed only from lfs.net, so no Steam library lookup is needed.

GUI-aware helpers (:func:`ask_for_lfs_dir`, :func:`require_lfs_dir`)
import PySide6 lazily so this module remains importable in headless
test environments where Qt is available but no display is attached.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QSettings

from .lfs_config import (
    cfg_path_for,
    lfs_data_dir,
    lfs_setups_dir,
)
from .lfs_config import (
    is_valid_lfs_dir as _is_valid_lfs_dir,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PySide6.QtWidgets import QWidget

# QSettings coordinates -----------------------------------------------------
# Public constants — single source of truth for the whole app. The
# Studio QApplication factory and the MainWindow about-box both
# import these from here instead of redefining them locally.

QSETTINGS_ORG = "LFS-Race-Engineer"
QSETTINGS_APP = "LFS Telemetry Studio"
QSETTINGS_DOMAIN = "lfs-race-engineer.local"

# Internal aliases kept for readability inside this module.
_ORG = QSETTINGS_ORG
_APP = QSETTINGS_APP
_KEY_LFS_DIR = "lfs/install_dir"
_KEY_FIRST_RUN_DONE = "lfs/first_run_complete"

# Hand-picked locations to probe when nothing is saved yet.
_STATIC_CANDIDATES: tuple[Path, ...] = (
    Path(r"C:\LFS"),
    Path(r"C:\Program Files\LFS"),
    Path(r"C:\Program Files (x86)\LFS"),
    Path(r"D:\LFS"),
    Path(r"D:\Games\LFS"),
    Path(r"C:\Games\LFS"),
)

# ---------------------------------------------------------------------------
# Validation (re-exported so callers only need this module)
# ---------------------------------------------------------------------------

def is_valid_lfs_dir(path: Path | None) -> bool:
    """True if *path* looks like an LFS install (has LFS.exe or cfg.txt)."""
    if path is None:
        return False
    return _is_valid_lfs_dir(Path(path))


# ---------------------------------------------------------------------------
# Persistent store
# ---------------------------------------------------------------------------

def _settings() -> QSettings:
    return QSettings(_ORG, _APP)


def get_lfs_dir() -> Path | None:
    """Return the saved LFS install dir if it is still valid, else ``None``."""
    raw = _settings().value(_KEY_LFS_DIR, "", type=str)
    if not raw:
        return None
    candidate = Path(raw)
    return candidate if is_valid_lfs_dir(candidate) else None


def set_lfs_dir(path: Path) -> None:
    """Persist *path* as the user's LFS install folder."""
    _settings().setValue(_KEY_LFS_DIR, str(Path(path)))
    _static_autodetect_candidates.cache_clear()


def forget_lfs_dir() -> None:
    """Remove the saved LFS install dir from settings."""
    _settings().remove(_KEY_LFS_DIR)
    _static_autodetect_candidates.cache_clear()


def first_run_complete() -> bool:
    """True once the setup wizard has been finished at least once."""
    return bool(_settings().value(_KEY_FIRST_RUN_DONE, False, type=bool))


def mark_first_run_complete() -> None:
    """Remember that the user has been through the setup wizard."""
    _settings().setValue(_KEY_FIRST_RUN_DONE, True)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def _registry_lfs_dir() -> Path | None:
    """Look up the LFS install path via ``HKCU\\Software\\Live for Speed``.

    LFS itself writes its install folder there. Returns ``None`` on any
    failure, on non-Windows hosts, or if the path no longer looks like a
    valid LFS install.
    """
    if os.name != "nt":
        return None
    try:
        import winreg
    except ImportError:  # pragma: no cover - non-Windows
        return None
    candidate_names = (
        "Install Folder", "InstallFolder", "InstallPath", "Path",
    )
    candidate_subkeys = (
        r"Software\Live for Speed",
        r"Software\LFS",
    )
    for subkey in candidate_subkeys:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, subkey) as key:
                for name in candidate_names:
                    try:
                        value, _ = winreg.QueryValueEx(key, name)
                    except OSError:
                        continue
                    candidate = Path(str(value))
                    if is_valid_lfs_dir(candidate):
                        return candidate
        except OSError:
            continue
    return None


def autodetect_candidates() -> list[Path]:
    """Return likely LFS install dirs in priority order (validated only).

    The order is: saved value, Windows registry, hand-picked common
    locations. Duplicates are dropped after ``resolve()``; non-existing
    entries are skipped.

    The registry/static probes are cached because they touch the
    filesystem; the cache is cleared whenever :func:`set_lfs_dir` or
    :func:`forget_lfs_dir` is called (the saved value is *not* cached
    because it can change independently via QSettings).
    """
    out: list[Path] = []
    seen: set[Path] = set()

    def push(path: Path | None) -> None:
        if path is None or not is_valid_lfs_dir(path):
            return
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        out.append(path)

    push(get_lfs_dir())
    for candidate in _static_autodetect_candidates():
        push(candidate)
    return out


@lru_cache(maxsize=1)
def _static_autodetect_candidates() -> tuple[Path, ...]:
    """Cached registry + ``_STATIC_CANDIDATES`` probes.

    Pure-filesystem lookup, no QSettings. Returns the raw candidate
    list (not validated); :func:`autodetect_candidates` filters with
    :func:`is_valid_lfs_dir`. The cache survives the process lifetime
    unless explicitly cleared.
    """
    out: list[Path] = []
    reg = _registry_lfs_dir()
    if reg is not None:
        out.append(reg)
    out.extend(_STATIC_CANDIDATES)
    return tuple(out)


def autodetect_lfs_dir() -> Path | None:
    """Return the first valid candidate or ``None`` if nothing is found."""
    cands = autodetect_candidates()
    return cands[0] if cands else None


# ---------------------------------------------------------------------------
# Path helpers (thin wrappers so callers stop importing lfs_config)
# ---------------------------------------------------------------------------

def lfs_exe(lfs_dir: Path) -> Path:
    """Path to ``<lfs>/LFS.exe`` (may not exist on disk)."""
    return Path(lfs_dir) / "LFS.exe"


def cfg_path(lfs_dir: Path) -> Path:
    """Path to ``<lfs>/cfg.txt`` (may not exist on disk)."""
    return cfg_path_for(Path(lfs_dir))


def data_dir(lfs_dir: Path) -> Path:
    """Path to ``<lfs>/data`` (may not exist on disk)."""
    return lfs_data_dir(Path(lfs_dir))


def veh_dir(lfs_dir: Path) -> Path:
    """Path to ``<lfs>/data/veh`` where downloaded mod vehicles live."""
    return data_dir(lfs_dir) / "veh"


def setups_dir(lfs_dir: Path, car_key: str | None = None) -> Path:
    """Return the setups folder, optionally under a per-car subfolder."""
    return lfs_setups_dir(Path(lfs_dir), car_key)


def car_info_bin_path(lfs_dir: Path, car_key: str) -> Path:
    """Path to ``<lfs>/data/<CAR>_CAR_info.bin``."""
    return data_dir(lfs_dir) / f"{car_key.upper()}_CAR_info.bin"


# ---------------------------------------------------------------------------
# GUI helpers (lazy Qt imports happen at call time inside ask_for_lfs_dir)
# ---------------------------------------------------------------------------

def ask_for_lfs_dir(
    parent: QWidget | None = None,
    *,
    initial: Path | None = None,
    persist: bool = True,
) -> Path | None:
    """Open a directory picker and validate the choice.

    Returns the chosen folder when the user picked a valid LFS install,
    otherwise ``None`` (cancel or invalid). When *persist* is true the
    selection is written to QSettings via :func:`set_lfs_dir`.
    """
    from PySide6.QtWidgets import QFileDialog, QMessageBox  # local import

    start = str(initial) if initial else str(Path.home())
    chosen = QFileDialog.getExistingDirectory(
        parent, "Select your LFS install folder", start,
    )
    if not chosen:
        return None
    path = Path(chosen)
    if not is_valid_lfs_dir(path):
        QMessageBox.warning(
            parent, "LFS folder",
            f"{path}\n\nDoes not look like an LFS install folder "
            "(no LFS.exe or cfg.txt found).",
        )
        return None
    if persist:
        set_lfs_dir(path)
    return path


def require_lfs_dir(
    parent: QWidget | None = None,
    *,
    allow_autodetect: bool = True,
) -> Path | None:
    """Return a valid LFS install dir, prompting the user if necessary.

    Resolution order:
      1. Previously saved value (if still valid).
      2. Auto-detected default (if *allow_autodetect* is true).
         This is also persisted on success.
      3. Interactive directory picker.
    """
    saved = get_lfs_dir()
    if saved is not None:
        return saved
    if allow_autodetect:
        guess = autodetect_lfs_dir()
        if guess is not None:
            set_lfs_dir(guess)
            return guess
    return ask_for_lfs_dir(parent)


__all__ = [
    "ask_for_lfs_dir",
    "autodetect_candidates",
    "autodetect_lfs_dir",
    "car_info_bin_path",
    "cfg_path",
    "data_dir",
    "first_run_complete",
    "forget_lfs_dir",
    "get_lfs_dir",
    "is_valid_lfs_dir",
    "lfs_exe",
    "mark_first_run_complete",
    "require_lfs_dir",
    "set_lfs_dir",
    "setups_dir",
    "veh_dir",
]
