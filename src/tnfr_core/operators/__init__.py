"""Operator primitives exposed by :mod:`tnfr_core`."""

from __future__ import annotations

import warnings
from importlib import import_module
from types import ModuleType
from typing import Any

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from . import cache as _cache
    from . import cache_settings as _cache_settings
    from . import interfaces as _interfaces
    from . import operator_labels as _operator_labels
from . import operator_detection as _operator_detection
from . import structural as _structural
from . import structural_time as _structural_time

from .cache import *  # noqa: F401,F403
from .cache_settings import *  # noqa: F401,F403
from .interfaces import *  # noqa: F401,F403
from .operator_labels import *  # noqa: F401,F403
from .operator_detection import *  # noqa: F401,F403
from .structural import *  # noqa: F401,F403
from .structural_time import *  # noqa: F401,F403

def _exported(module: object) -> list[str]:
    names = getattr(module, "__all__", None)
    if names is not None:
        return list(names)
    return [name for name in vars(module) if not name.startswith("_")]


_EAGER_MODULES = (
    _cache,
    _cache_settings,
    _interfaces,
    _operator_labels,
    _operator_detection,
    _structural,
    _structural_time,
)


__all__ = [
    *_exported(_cache),
    *_exported(_cache_settings),
    *_exported(_interfaces),
    *_exported(_operator_labels),
    *_exported(_operator_detection),
    *_exported(_structural),
    *_exported(_structural_time),
]

_PIPELINE_EXPORTS = {"PipelineDependencies", "orchestrate_delta_metrics"}

__all__ = list(dict.fromkeys([*__all__, *_PIPELINE_EXPORTS]))

_delayed_module_name = "operators"
_delayed_all: list[str] | None = None
_delayed_module: ModuleType | None = None
_pipeline_module: ModuleType | None = None


def _load_delayed_module() -> ModuleType:
    global _delayed_all, _delayed_module
    if _delayed_module is not None:
        return _delayed_module
    module = import_module(f"{__name__}.{_delayed_module_name}")
    _delayed_module = module
    names = _exported(module)
    _delayed_all = list(names)
    for item in _delayed_all:
        if item not in __all__:
            __all__.append(item)
    return module


def __getattr__(name: str) -> Any:
    if name in _PIPELINE_EXPORTS:
        global _pipeline_module
        if _pipeline_module is None:
            _pipeline_module = import_module(f"{__name__}.pipeline")
        if hasattr(_pipeline_module, name):
            value = getattr(_pipeline_module, name)
            globals()[name] = value
            return value
    module = _load_delayed_module()
    if hasattr(module, name):
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    _load_delayed_module()
    return sorted(set(__all__) | set(globals().keys()))


del _cache
del _cache_settings
del _interfaces
del _operator_labels
del _operator_detection
del _structural
del _structural_time


def __setattr__(name: str, value: object) -> None:  # pragma: no cover - assignment helper
    for module in _EAGER_MODULES:
        if hasattr(module, name):
            setattr(module, name, value)
    if _delayed_module is not None and hasattr(_delayed_module, name):
        setattr(_delayed_module, name, value)
    globals()[name] = value
