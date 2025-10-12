"""Helpers to locate bundled TNFR × LFS resources."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Iterable

__all__ = ["pack_root", "data_root", "modifiers_root", "set_pack_root_override"]

_FALLBACK_PACK_ROOT = Path(__file__).resolve().parent
_PACKAGES: tuple[str, ...] = ("tnfr_lfs.resources", "tnfr_lfs.pack")

_PACK_ROOT: Path | None = None
_DATA_ROOT: Path | None = None
_MODIFIERS_ROOT: Path | None = None
_PACK_ROOT_OVERRIDE: Path | None = None


def _iter_candidate_roots() -> Iterable[Path]:
    if _PACK_ROOT_OVERRIDE is not None:
        yield _PACK_ROOT_OVERRIDE
        return

    for package_name in _PACKAGES:
        try:
            package_path = Path(resources.files(package_name))
        except ModuleNotFoundError:
            continue
        else:
            yield package_path

    yield _FALLBACK_PACK_ROOT


def set_pack_root_override(path: Path | None) -> None:
    """Force :func:`pack_root` to return ``path`` (used in tests)."""

    global _PACK_ROOT_OVERRIDE, _PACK_ROOT, _DATA_ROOT, _MODIFIERS_ROOT

    _PACK_ROOT_OVERRIDE = Path(path) if path is not None else None
    _PACK_ROOT = None
    _DATA_ROOT = None
    _MODIFIERS_ROOT = None


def pack_root() -> Path:
    """Return the directory that contains the installed pack resources."""

    global _PACK_ROOT
    if _PACK_ROOT is None:
        for candidate in _iter_candidate_roots():
            if candidate.exists():
                _PACK_ROOT = candidate
                break
        else:  # pragma: no cover - defensive fallback
            _PACK_ROOT = _FALLBACK_PACK_ROOT
    return _PACK_ROOT


def data_root() -> Path:
    """Return the directory containing the pack's ``data`` tree."""

    global _DATA_ROOT
    if _DATA_ROOT is None:
        root = pack_root()
        candidate = root / "data"
        if candidate.exists():
            _DATA_ROOT = candidate
        else:
            _DATA_ROOT = _FALLBACK_PACK_ROOT / "data"
    return _DATA_ROOT


def modifiers_root() -> Path:
    """Return the directory containing packaged combo modifiers."""

    global _MODIFIERS_ROOT
    if _MODIFIERS_ROOT is None:
        root = pack_root()
        candidate = root / "modifiers" / "combos"
        if candidate.exists():
            _MODIFIERS_ROOT = candidate
        else:
            _MODIFIERS_ROOT = _FALLBACK_PACK_ROOT / "modifiers" / "combos"
    return _MODIFIERS_ROOT
