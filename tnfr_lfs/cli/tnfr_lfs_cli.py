"""Command line entry point for TNFR × LFS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .io import load_cli_config
from .parser import build_parser
from .workflows import *  # noqa: F401,F403 - re-export workflow helpers


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
