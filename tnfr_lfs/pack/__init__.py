"""Bundled TNFR × LFS resource pack (datasets, configuration and modifiers)."""

from __future__ import annotations

from pathlib import Path

from .._pack_resources import data_root as _data_root
from .._pack_resources import modifiers_root as _modifiers_root
from .._pack_resources import pack_root as _pack_root

__all__ = ["pack_root", "data_root", "modifiers_root"]


def pack_root() -> Path:
    """Return the filesystem location of the bundled pack."""

    return _pack_root()


def data_root() -> Path:
    """Return the directory containing bundled datasets."""

    return _data_root()


def modifiers_root() -> Path:
    """Return the directory containing bundled modifier manifests."""

    return _modifiers_root()
