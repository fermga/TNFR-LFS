"""Deprecated aliases for :mod:`tnfr_lfs.common.immutables`."""

from __future__ import annotations

import warnings

from ..common.immutables import _freeze_dict, _freeze_value

warnings.warn(
    "'tnfr_lfs.utils.immutables' is deprecated; use 'tnfr_lfs.common.immutables' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["_freeze_value", "_freeze_dict"]
