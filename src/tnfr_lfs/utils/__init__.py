"""Deprecated utility module preserved for backward compatibility."""

from __future__ import annotations

import importlib
import warnings
from typing import Any

__all__ = [
    "immutables",
    "logging",
    "numeric",
    "sparkline",
]

_DEPRECATED_MODULES = {
    "immutables": "tnfr_lfs.common.immutables",
    "logging": "tnfr_lfs.logging.config",
    "numeric": "tnfr_lfs.math.conversions",
    "sparkline": "tnfr_lfs.visualization.sparkline",
}

_DEPRECATION_MESSAGE = (
    "'tnfr_lfs.utils.{name}' is deprecated and will be removed in a future release; "
    "import from '{target}' instead."
)


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_MODULES:
        target = _DEPRECATED_MODULES[name]
        warnings.warn(
            _DEPRECATION_MESSAGE.format(name=name, target=target),
            DeprecationWarning,
            stacklevel=2,
        )
        module = importlib.import_module(target)
        globals()[name] = module
        return module
    raise AttributeError(f"module 'tnfr_lfs.utils' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
