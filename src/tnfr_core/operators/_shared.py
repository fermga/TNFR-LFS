"""Compatibility shim for runtime shared utilities."""

from __future__ import annotations

import warnings

from tnfr_core.runtime.shared import _HAS_JAX, is_module_available, jnp

warnings.warn(
    (
        "'tnfr_core.operators._shared' is deprecated and will be removed in a future "
        "release; import from 'tnfr_core.runtime.shared' instead."
    ),
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["_HAS_JAX", "jnp", "is_module_available"]
