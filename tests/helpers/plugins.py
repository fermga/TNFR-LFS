"""Test helpers for managing the plugin metadata registry."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from typing import Callable, Iterator, Tuple

from tnfr_lfs.plugins.base import TNFRPlugin
from tnfr_lfs.plugins import register_plugin_metadata
from tnfr_lfs.plugins.registry import _clear_registry

PluginRegistration = Tuple[type[TNFRPlugin], Sequence[str]]
RegisterPlugin = Callable[[type[TNFRPlugin], Sequence[str]], type[TNFRPlugin]]


@contextmanager
def plugin_registry_state(
    initial: Iterable[PluginRegistration] | None = None,
) -> Iterator[RegisterPlugin]:
    """Manage the plugin registry state for tests.

    The registry is cleared on entry and exit. Optionally, ``initial`` registrations
    can be supplied as an iterable of ``(plugin_cls, operators)`` tuples. The yielded
    callable allows tests to register additional plugin metadata during the managed
    context.
    """

    if initial is None:
        initial = ()

    def register(plugin_cls: type[TNFRPlugin], operators: Sequence[str]) -> type[TNFRPlugin]:
        return register_plugin_metadata(plugin_cls, operators=operators)

    try:
        _clear_registry()
        for plugin_cls, operators in initial:
            register(plugin_cls, operators)
        yield register
    finally:
        _clear_registry()
