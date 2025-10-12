"""Deprecated aliases for :mod:`tnfr_lfs.visualization.sparkline`."""

from __future__ import annotations

import warnings

from ..visualization.sparkline import DEFAULT_SPARKLINE_BLOCKS, render_sparkline

warnings.warn(
    "'tnfr_lfs.utils.sparkline' is deprecated; use 'tnfr_lfs.visualization.sparkline' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DEFAULT_SPARKLINE_BLOCKS", "render_sparkline"]
