"""Centralized resolution of bundled & repo-root asset/docs/config paths.

All historical call sites used variants of the same probe order
(`cwd` → exe dir → PyInstaller ``_MEIPASS`` → package-root fallback)
to find resources that ship both inside a frozen build (where they
live next to the .exe or under ``sys._MEIPASS``) and inside a developer
checkout (where they sit at the repo root). Keeping each call site's
own copy of that logic invited drift; this module is the single source
of truth.

Note: this module is intentionally separate from :mod:`lfs_paths`,
which is specifically about the user's *LFS install* folder — a
distinct concept from the app's own bundled assets.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "candidate_asset_dirs",
    "candidate_doc_roots",
    "candidate_racing_lines_dirs",
    "candidate_search_roots",
    "car_info_bin_dirs",
    "cars_json_path",
    "find_racing_line_csv",
    "manual_doc_path",
    "mod_database_path",
]

# src/lfs_telemetry/app_paths.py → parents[2] == repo root in dev checkout.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _dedup(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve() if p.exists() else p
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def candidate_search_roots() -> list[Path]:
    """Return ordered root dirs to probe for bundled-or-repo resources.

    Order: ``cwd`` → ``sys.argv[0]`` dir → PyInstaller ``_MEIPASS`` →
    package-root (dev checkout fallback).
    """
    roots: list[Path] = [Path.cwd()]
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0:
        try:
            exe = Path(argv0).resolve().parent
        except OSError:
            exe = None
        if exe and exe.exists():
            roots.append(exe)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))
    roots.append(_PACKAGE_ROOT)
    return _dedup(roots)


def candidate_asset_dirs(
    *subpath: str, env_var: str | None = None
) -> list[Path]:
    """Search dirs/files for an asset under ``<root>/<subpath>``.

    If ``env_var`` names a set environment variable, that value is
    prepended verbatim (treated as a direct path, no subpath append).
    """
    out: list[Path] = []
    if env_var:
        env = os.environ.get(env_var)
        if env:
            out.append(Path(env))
    for r in candidate_search_roots():
        out.append(r.joinpath(*subpath))
    return out


def candidate_doc_roots() -> list[Path]:
    """Roots that may contain a ``docs/`` subdir."""
    # Match the legacy manual-dialog probe: also climb from this file
    # for a parent containing a ``docs`` folder (covers the case where
    # ``cwd`` is unrelated to the repo).
    roots = candidate_search_roots()
    for parent in Path(__file__).resolve().parents:
        if (parent / "docs").is_dir():
            roots.append(parent)
            break
    return _dedup(roots)


def manual_doc_path(lang_code: str, *, spanish_code: str) -> Path | None:
    """Locate the localised user manual; falls back to English.

    Caller passes the language code currently active in the UI and
    the constant that identifies Spanish, so this module stays
    decoupled from :mod:`lfs_telemetry.i18n`.
    """
    primary = "MANUAL.es.md" if lang_code == spanish_code else "MANUAL.en.md"
    fallback = "MANUAL.en.md"
    for root in candidate_doc_roots():
        for fname in (primary, fallback):
            p = root / "docs" / fname
            if p.is_file():
                return p
    return None


def candidate_racing_lines_dirs() -> list[Path]:
    """Search dirs for ``<dir>/<TRACK>_racing.csv``."""
    out = [r / "racing_lines" for r in candidate_search_roots()]
    return _dedup(out)


def find_racing_line_csv(track: str) -> Path | None:
    """Locate ``<dir>/<TRACK>_racing.csv`` under any candidate dir."""
    if not track:
        return None
    name = f"{track.upper()}_racing.csv"
    for base in candidate_racing_lines_dirs():
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


def mod_database_path() -> Path:
    """Resolve the on-disk mod-sizes JSON catalogue path.

    Honours ``$LFS_TELEMETRY_MOD_DB`` for tests/installers. Otherwise
    prefers the first existing ``assets/source/mods/mod_sizes.json``
    under any candidate root, falling back to the package-root path
    (which may not exist yet — callers handle that).
    """
    env = os.environ.get("LFS_TELEMETRY_MOD_DB")
    if env:
        return Path(env)
    rel = ("assets", "source", "mods", "mod_sizes.json")
    for root in candidate_search_roots():
        p = root.joinpath(*rel)
        if p.exists():
            return p
    return _PACKAGE_ROOT.joinpath(*rel)


def cars_json_path() -> Path | None:
    """First existing ``./config/cars.json`` under the search roots."""
    for p in candidate_asset_dirs(
        "config", "cars.json", env_var="LFS_TELEMETRY_CARS_JSON"
    ):
        if p.exists():
            return p
    return None


def car_info_bin_dirs() -> list[Path]:
    """Search dirs for ``<KEY>_CAR_info.bin`` exports."""
    dirs = candidate_asset_dirs(
        "assets", "source", "cars", env_var="LFS_TELEMETRY_CAR_INFO_DIR"
    )
    # LFS exports may sit at either ``assets/source/cars/`` or
    # ``assets/source/`` directly — probe the parent too.
    dirs.extend([d.parent for d in list(dirs) if d.name == "cars"])
    return dirs
