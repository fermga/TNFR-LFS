"""Compatibility layer for :mod:`tnfr_lfs.cache_settings`.

This module preserves the public interface for downstream users while
emitting a deprecation warning guiding them towards
:mod:`tnfr_core.runtime.shared`.
"""

from __future__ import annotations

from warnings import warn

from tnfr_core.runtime.shared import (
    CacheOptions,
    DEFAULT_DYNAMIC_CACHE_SIZE,
    DEFAULT_RECOMMENDER_CACHE_SIZE,
    configure_cache_from_options,
)

warn(
    "'tnfr_lfs.cache_settings' is deprecated and will be removed in a future "
    "release. Please import from 'tnfr_core.runtime.shared' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CacheOptions",
    "DEFAULT_DYNAMIC_CACHE_SIZE",
    "DEFAULT_RECOMMENDER_CACHE_SIZE",
    "configure_cache_from_options",
]
