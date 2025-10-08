"""Command line entry point for TNFR × LFS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import inspect
import sys
import types

from .io import load_cli_config
from .parser import build_parser
from . import workflows as _workflows_module
from . import io as _cli_io_module
from . import parser as _parser_module


_PROXY_SOURCES: tuple[types.ModuleType, ...] = (
    _workflows_module,
    _cli_io_module,
    _parser_module,
)

_ALWAYS_PROXY_NAMES: frozenset[str] = frozenset({
    "compute_session_robustness",
})


def _gather_proxy_registry() -> dict[str, tuple[types.ModuleType, ...]]:
    registry: dict[str, list[types.ModuleType]] = {}
    for module in _PROXY_SOURCES:
        for name in dir(module):
            if name.startswith("__"):
                continue
            try:
                value = getattr(module, name)
            except AttributeError:  # pragma: no cover - defensive fallback
                continue
            if not callable(value):
                continue
            should_proxy = (
                name in _ALWAYS_PROXY_NAMES
                or name.startswith("_")
                or inspect.isfunction(value)
            )
            if not should_proxy:
                continue
            registry.setdefault(name, []).append(module)
    return {key: tuple(modules) for key, modules in registry.items()}


_PROXY_REGISTRY = _gather_proxy_registry()


def _initialise_proxy_bindings(module: types.ModuleType) -> None:
    namespace = module.__dict__
    for name, origins in _PROXY_REGISTRY.items():
        if name in namespace:
            continue
        namespace[name] = getattr(origins[0], name)


class _CLIModule(types.ModuleType):
    """Proxy module that mirrors workflow helpers for monkeypatching tests."""

    def __getattr__(self, name: str):
        if name in self.__dict__:
            return self.__dict__[name]
        origins = _PROXY_REGISTRY.get(name)
        if origins:
            value = getattr(origins[0], name)
            super().__setattr__(name, value)
            return value
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        origins = _PROXY_REGISTRY.get(name)
        if not origins:
            return
        for module in origins:
            setattr(module, name, value)

    def __dir__(self) -> list[str]:
        base = set(super().__dir__())
        base.update(_PROXY_REGISTRY)
        return sorted(base)


_module = sys.modules[__name__]
_module.__class__ = _CLIModule
_initialise_proxy_bindings(_module)

def run_cli(args: Optional[Sequence[str]] = None) -> str:
    """Execute the TNFR × LFS command line interface."""

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=None,
        help="Ruta del fichero de configuración TOML a utilizar.",
    )
    config_parser.add_argument(
        "--pack-root",
        dest="pack_root",
        type=Path,
        default=None,
        help=(
            "Directorio raíz de un pack TNFR × LFS con config/ y data/. "
            "Sobrescribe paths.pack_root."
        ),
    )
    preliminary, remaining = config_parser.parse_known_args(args)
    remaining = list(remaining)
    if preliminary.pack_root is not None:
        remaining = ["--pack-root", str(preliminary.pack_root)] + remaining

    config = load_cli_config(preliminary.config_path)
    parser = build_parser(config)
    parser.set_defaults(config_path=preliminary.config_path)
    namespace = parser.parse_args(remaining)
    namespace.config = config
    namespace.config_path = (
        getattr(namespace, "config_path", None)
        or preliminary.config_path
        or config.get("_config_path")
    )
    handler: Callable[[argparse.Namespace, Mapping[str, Any]], str] = getattr(
        namespace, "handler"
    )
    return handler(namespace, config=config)


def main() -> None:  # pragma: no cover - thin wrapper
    run_cli()


if __name__ == "__main__":  # pragma: no cover - CLI invocation guard
    main()
