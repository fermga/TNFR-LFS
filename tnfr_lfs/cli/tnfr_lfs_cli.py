"""Command line entry point for TNFR × LFS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Mapping, Sequence

import sys
import types

from .io import load_cli_config, raf_to_telemetry_records
from .parser import build_parser, _add_export_argument, _validated_export
from ..core.operators import orchestrate_delta_metrics
from . import workflows as _workflows_module
from . import io as _cli_io_module


class _CLIModule(types.ModuleType):
    """Proxy module that mirrors workflow helpers for monkeypatching tests."""

    def __getattr__(self, name: str):
        if name in self.__dict__:
            return self.__dict__[name]
        if hasattr(_workflows_module, name):
            return getattr(_workflows_module, name)
        if hasattr(_cli_io_module, name):
            return getattr(_cli_io_module, name)
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        if hasattr(_workflows_module, name):
            setattr(_workflows_module, name, value)
        if hasattr(_cli_io_module, name):
            setattr(_cli_io_module, name, value)


sys.modules[__name__].__class__ = _CLIModule

def run_cli(args: Sequence[str] | None = None) -> str:
    """Execute the TNFR × LFS command line interface."""

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", dest="config_path", type=Path, default=None)
    config_parser.add_argument("--pack-root", dest="pack_root", type=Path, default=None)
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
