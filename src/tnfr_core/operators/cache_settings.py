"""Compatibility layer for legacy cache configuration imports."""

from __future__ import annotations

import warnings

from tnfr_core.runtime.shared import (
    CacheOptions,
    DEFAULT_DYNAMIC_CACHE_SIZE,
    DEFAULT_RECOMMENDER_CACHE_SIZE,
    LEGACY_TELEMETRY_CACHE_KEY,
    resolve_recommender_cache_size,
)

warnings.warn(
    (
        "'tnfr_core.operators.cache_settings' is deprecated and will be removed in a "
        "future release; import from 'tnfr_core.runtime.shared' instead."
    ),
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CacheOptions",
    "DEFAULT_DYNAMIC_CACHE_SIZE",
    "DEFAULT_RECOMMENDER_CACHE_SIZE",
    "resolve_recommender_cache_size",
]
