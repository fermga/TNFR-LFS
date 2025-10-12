"""Deprecated aliases for :mod:`tnfr_lfs.math.conversions`."""

from __future__ import annotations

import warnings

from ..math.conversions import _safe_float

warnings.warn(
    "'tnfr_lfs.utils.numeric' is deprecated; use 'tnfr_lfs.math.conversions' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["_safe_float"]
