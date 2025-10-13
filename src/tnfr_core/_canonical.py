"""Helpers for conditional imports of the canonical ``tnfr`` package."""

from __future__ import annotations

import os
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import Final

__all__ = [
    "CANONICAL_ENV_VALUE",
    "CANONICAL_REQUESTED",
    "TNFR_AVAILABLE",
    "TNFR_MINIMUM_VERSION",
    "CanonicalImportError",
    "import_tnfr",
    "require_tnfr",
]

TNFR_MINIMUM_VERSION: Final[str] = "2024.6"


class CanonicalImportError(ImportError):
    """Raised when canonical TNFR imports cannot be satisfied."""


_FALSE_SENTINELS: Final[set[str]] = {"", "0", "false", "no", "off"}


def _coerce_flag(value: str | None) -> bool:
    if value is None:
        return False
    normalised = value.strip().lower()
    return normalised not in _FALSE_SENTINELS


CANONICAL_ENV_VALUE = os.getenv("TNFR_CANONICAL")
CANONICAL_REQUESTED = _coerce_flag(CANONICAL_ENV_VALUE)

_TNFR_IMPORT_ERROR: ImportError | None = None

if CANONICAL_REQUESTED:
    try:
        import_module("tnfr")
    except ImportError as exc:  # pragma: no cover - exercised through configuration
        TNFR_AVAILABLE = False
        _TNFR_IMPORT_ERROR = exc
    else:  # pragma: no cover - trivial branch
        TNFR_AVAILABLE = True
else:
    TNFR_AVAILABLE = find_spec("tnfr") is not None

if not TNFR_AVAILABLE and _TNFR_IMPORT_ERROR is None:
    _TNFR_IMPORT_ERROR = ModuleNotFoundError("No module named 'tnfr'")

_MISSING_TNFR_MESSAGE = (
    "TNFR canonical mode requires the optional dependency 'tnfr' "
    f"(>= {TNFR_MINIMUM_VERSION}). Install it via "
    "'pip install \"tnfr_lfs[tnfr]\"' or add the package to your environment."
)


if CANONICAL_REQUESTED and not TNFR_AVAILABLE:
    raise CanonicalImportError(_MISSING_TNFR_MESSAGE) from _TNFR_IMPORT_ERROR


def import_tnfr() -> ModuleType:
    """Import and return the canonical :mod:`tnfr` package."""

    try:
        return import_module("tnfr")
    except ImportError as exc:  # pragma: no cover - depends on external package
        raise CanonicalImportError(_MISSING_TNFR_MESSAGE) from exc


def require_tnfr() -> None:
    """Ensure the canonical :mod:`tnfr` package is available."""

    import_tnfr()
