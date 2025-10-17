"""Compatibility layer for legacy cache helper imports."""

from __future__ import annotations

import warnings

from tnfr_core.runtime.shared import (
    LRUCache,
    cached_delta_nfr_map,
    invalidate_delta_record,
    clear_delta_cache,
    cached_dynamic_multipliers,
    invalidate_dynamic_record,
    clear_dynamic_cache,
    configure_cache,
    configure_cache_from_options,
    should_use_delta_cache,
    should_use_dynamic_cache,
    delta_cache_enabled,
    dynamic_cache_enabled,
)

warnings.warn(
    (
        "'tnfr_core.operators.cache' is deprecated and will be removed in a future "
        "release; import from 'tnfr_core.runtime.shared' instead."
    ),
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "cached_delta_nfr_map",
    "invalidate_delta_record",
    "clear_delta_cache",
    "cached_dynamic_multipliers",
    "invalidate_dynamic_record",
    "clear_dynamic_cache",
    "configure_cache",
    "configure_cache_from_options",
    "should_use_delta_cache",
    "should_use_dynamic_cache",
    "delta_cache_enabled",
    "dynamic_cache_enabled",
    "LRUCache",
]
