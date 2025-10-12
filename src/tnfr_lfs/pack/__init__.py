"""Deprecated shim exposing bundled TNFR × LFS resources."""

from __future__ import annotations

import warnings
from pathlib import Path

from ..resources import data_root as _data_root
from ..resources import modifiers_root as _modifiers_root
from ..resources import pack_root as _pack_root

__all__ = ["pack_root", "data_root", "modifiers_root"]

_DEPRECATION_MESSAGE = (
    "tnfr_lfs.pack is deprecated; import helpers from tnfr_lfs.resources instead."
)


def _warn() -> None:
    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)


def pack_root() -> Path:
    """Return the filesystem location of the bundled pack."""

    _warn()
    return _pack_root()


def data_root() -> Path:
    """Return the directory containing bundled datasets."""

    _warn()
    return _data_root()


def modifiers_root() -> Path:
    """Return the directory containing bundled modifier manifests."""

    _warn()
    return _modifiers_root()
