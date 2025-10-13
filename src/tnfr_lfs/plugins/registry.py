"""Metadata registry for TNFR × LFS plugin dependencies."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import lru_cache
from typing import Dict, Tuple, Type
from tnfr_lfs.plugins.base import TNFRPlugin

OperatorName = str

_PLUGIN_REGISTRY: Dict[type[TNFRPlugin], Tuple[OperatorName, ...]] = {}

__all__ = [
    "PluginMetadataError",
    "available_operator_identifiers",
    "get_plugin_operator_requirements",
    "iter_plugin_operator_requirements",
    "plugin_metadata",
    "register_plugin_metadata",
]


class PluginMetadataError(ValueError):
    """Raised when invalid plugin metadata is supplied to the registry."""


@lru_cache(maxsize=1)
def _known_operator_identifiers() -> frozenset[OperatorName]:
    from tnfr_core import operators as _operators

    return frozenset(_operators.__all__)


def available_operator_identifiers() -> frozenset[OperatorName]:
    """Return the canonical set of operator identifiers available to plugins."""

    return _known_operator_identifiers()


def _normalise_operator_identifiers(
    operators: Sequence[OperatorName] | None,
) -> Tuple[OperatorName, ...]:
    """Validate and normalise operator identifiers."""

    if operators is None:
        raise PluginMetadataError("operators must be provided")

    normalised: list[OperatorName] = []
    seen: set[OperatorName] = set()

    known_operators = _known_operator_identifiers()

    for name in operators:
        if not isinstance(name, str) or not name:
            raise PluginMetadataError(
                "operator identifiers must be non-empty strings"
            )
        if name not in known_operators:
            raise PluginMetadataError(
                f"unknown operator identifier '{name}'"
            )
        if name in seen:
            continue
        seen.add(name)
        normalised.append(name)

    return tuple(normalised)


def register_plugin_metadata(
    plugin_cls: Type[TNFRPlugin], *, operators: Sequence[OperatorName] | None
) -> Type[TNFRPlugin]:
    """Register metadata describing the operator requirements of ``plugin_cls``."""

    if not isinstance(plugin_cls, type) or not issubclass(plugin_cls, TNFRPlugin):
        raise PluginMetadataError(
            "plugin_cls must be a subclass of TNFRPlugin"
        )

    normalised = _normalise_operator_identifiers(operators)
    existing = _PLUGIN_REGISTRY.get(plugin_cls)
    if existing is not None and existing != normalised:
        raise PluginMetadataError(
            f"plugin {plugin_cls.__name__!s} already registered with different metadata"
        )

    _PLUGIN_REGISTRY[plugin_cls] = normalised
    return plugin_cls


def plugin_metadata(*, operators: Sequence[OperatorName] | None) -> type:
    """Decorator used by plugins to declare their operator dependencies."""

    def decorator(plugin_cls: Type[TNFRPlugin]) -> Type[TNFRPlugin]:
        return register_plugin_metadata(plugin_cls, operators=operators)

    return decorator


def get_plugin_operator_requirements(
    plugin_cls: Type[TNFRPlugin],
) -> Tuple[OperatorName, ...]:
    """Return the operator identifiers registered for ``plugin_cls``."""

    try:
        return _PLUGIN_REGISTRY[plugin_cls]
    except KeyError as exc:  # pragma: no cover - error path tested
        raise LookupError(
            f"plugin {plugin_cls.__name__!s} has not been registered"
        ) from exc


def iter_plugin_operator_requirements() -> Iterator[tuple[type[TNFRPlugin], Tuple[OperatorName, ...]]]:
    """Iterate over registered plugin classes and their operator requirements."""

    return iter(_PLUGIN_REGISTRY.items())


def _clear_registry() -> None:
    """Test helper clearing the registry; not part of the public API."""

    _PLUGIN_REGISTRY.clear()
