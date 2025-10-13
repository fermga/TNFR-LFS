"""Compatibility layer for :mod:`tnfr_lfs.cache_settings`.

This module preserves the public interface for downstream users while
emitting a deprecation warning guiding them towards
:mod:`tnfr_lfs.core.cache_settings`.
"""

from __future__ import annotations

from warnings import warn

from tnfr_lfs.core.cache import configure_cache_from_options
from tnfr_lfs.core.cache_settings import (
    CacheOptions,
    DEFAULT_DYNAMIC_CACHE_SIZE,
    DEFAULT_RECOMMENDER_CACHE_SIZE,
)

warn(
    "'tnfr_lfs.cache_settings' is deprecated and will be removed in a future "
    "release. Please import from 'tnfr_lfs.core.cache_settings' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CacheOptions",
    "DEFAULT_DYNAMIC_CACHE_SIZE",
    "DEFAULT_RECOMMENDER_CACHE_SIZE",
    "configure_cache_from_options",
]
