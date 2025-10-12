"""Deprecated aliases for :mod:`tnfr_lfs.logging.config`."""

from __future__ import annotations

import warnings

from ..logging.config import JsonFormatter, setup_logging

warnings.warn(
    "'tnfr_lfs.utils.logging' is deprecated; use 'tnfr_lfs.logging.config' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["JsonFormatter", "setup_logging"]
