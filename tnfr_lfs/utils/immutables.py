"""Helpers to transform mutable containers into immutable counterparts."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

__all__ = ["_freeze_value", "_freeze_dict"]


_EMPTY_MAPPING = MappingProxyType({})


def _freeze_value(value: Any) -> Any:
    """Recursively convert mutable containers into immutable counterparts."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_dict(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an immutable view for a mapping, freezing nested structures."""

    if not payload:
        return _EMPTY_MAPPING
    return MappingProxyType({str(key): _freeze_value(value) for key, value in payload.items()})
