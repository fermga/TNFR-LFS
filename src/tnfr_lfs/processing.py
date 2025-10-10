"""Compatibility shim for :mod:`tnfr_lfs.analysis.insights`."""

from __future__ import annotations

import warnings

from .analysis.insights import InsightsResult, compute_insights

warnings.warn(
    "'tnfr_lfs.processing' is deprecated; use 'tnfr_lfs.analysis.insights' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["InsightsResult", "compute_insights"]
