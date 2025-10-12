"""Command line application entry point for TNFR × LFS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from ..ingestion.config_loader import parse_cache_options
from ..core import cache as cache_helpers
from ..logging.config import setup_logging

from . import common as _common_module
from . import io as _cli_io_module
from . import workflows as _workflows_module
from .common import CliError
from .errors import log_cli_error
from .io import load_cli_config
from .parser import build_parser
from .workflows import (
    _handle_analyze,
    _handle_baseline,
    _handle_diagnose,
    _handle_osd,
    _handle_report,
    _handle_suggest,
    _handle_template,
    _handle_write_set,
)


CommandHandler = Callable[[argparse.Namespace, Mapping[str, Any]], str]


_HELPER_EXPORTS = {
    "compute_session_robustness": _workflows_module.compute_session_robustness,
    "_outsim_ping": _workflows_module._outsim_ping,
    "_outgauge_ping": _workflows_module._outgauge_ping,
    "_insim_handshake": _workflows_module._insim_handshake,
    "_check_setups_directory": _workflows_module._check_setups_directory,
    "_copy_to_clipboard": _workflows_module._copy_to_clipboard,
    "_generate_out_reports": _workflows_module._generate_out_reports,
    "_sense_index_map": _workflows_module._sense_index_map,
    "_prepare_pack_context": _workflows_module._prepare_pack_context,
    "compute_insights": _workflows_module.compute_insights,
    "_compute_insights": _workflows_module._compute_insights,
    "_phase_deviation_messages": _workflows_module._phase_deviation_messages,
    "orchestrate_delta_metrics": _workflows_module.orchestrate_delta_metrics,
    "compute_setup_plan": _workflows_module.compute_setup_plan,
    "build_setup_plan_payload": _workflows_module.build_setup_plan_payload,
    "assemble_session_payload": _workflows_module.assemble_session_payload,
    "_assemble_session_payload": _workflows_module.assemble_session_payload,
    "_resolve_pack_root": _common_module.resolve_pack_root,
    "_resolve_track_argument": _common_module.resolve_track_argument,
    "_load_pack_cars": _common_module.load_pack_cars,
    "_load_pack_profiles": _common_module.load_pack_profiles,
    "_load_pack_track_profiles": _common_module.load_pack_track_profiles,
    "_load_pack_modifiers": _common_module.load_pack_modifiers,
    "_group_records_by_lap": _common_module.group_records_by_lap,
    "_render_payload": _common_module.render_payload,
    "_resolve_exports": _common_module.resolve_exports,
    "_validated_export": _common_module.validated_export,
    "_add_export_argument": _common_module.add_export_argument,
    "_load_records": _cli_io_module._load_records,
    "_load_records_from_namespace": _common_module._load_records_from_namespace,
    "raf_to_telemetry_records": _cli_io_module.raf_to_telemetry_records,
    "_load_pack_lfs_class_overrides": _workflows_module._load_pack_lfs_class_overrides,
}

globals().update(_HELPER_EXPORTS)

for _name in dir(_workflows_module):
    if not _name.startswith("_") or _name in _HELPER_EXPORTS:
        continue
    globals().setdefault(_name, getattr(_workflows_module, _name))


def run_cli(args: Optional[Sequence[str]] = None) -> str:
    """Execute the TNFR × LFS command line interface."""

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=None,
        help="Path to the TOML configuration file to load.",
    )
    config_parser.add_argument(
        "--pack-root",
        dest="pack_root",
        type=Path,
        default=None,
        help="Root directory of a TNFR × LFS pack overriding resources.pack_root.",
    )
    config_parser.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        help="Logging level (default: info).",
    )
    config_parser.add_argument(
        "--log-output",
        dest="log_output",
        default=None,
        help="Logging destination (stdout, stderr or a file path).",
    )
    config_parser.add_argument(
        "--log-format",
        dest="log_format",
        choices=("json", "text"),
        default=None,
        help="Logging formatter (json or text).",
    )
    preliminary, remaining = config_parser.parse_known_args(args)
    remaining = list(remaining)
    if preliminary.pack_root is not None:
        remaining = ["--pack-root", str(preliminary.pack_root)] + remaining

    config = load_cli_config(preliminary.config_path)
    cache_options = parse_cache_options(config, pack_root=preliminary.pack_root)
    cache_helpers.configure_cache_from_options(cache_options)
    config["_cache_options"] = cache_options
    logging_config = dict(config.get("logging", {}))
    if preliminary.log_level is not None:
        logging_config["level"] = preliminary.log_level
    if preliminary.log_output is not None:
        logging_config["output"] = preliminary.log_output
    if preliminary.log_format is not None:
        logging_config["format"] = preliminary.log_format
    logging_config.setdefault("level", "info")
    logging_config.setdefault("output", "stderr")
    logging_config.setdefault("format", "json")
    config["logging"] = logging_config
    setup_logging(config)

    parser = build_parser(config)
    parser.set_defaults(config_path=preliminary.config_path)
    parser.set_defaults(log_level=logging_config.get("level"))
    parser.set_defaults(log_output=logging_config.get("output"))
    parser.set_defaults(log_format=logging_config.get("format"))
    namespace = parser.parse_args(remaining, namespace=preliminary)
    namespace.config = config
    namespace.cache_options = cache_options
    namespace.config_path = (
        getattr(namespace, "config_path", None)
        or preliminary.config_path
        or config.get("_config_path")
    )

    handler = getattr(namespace, "handler", None)
    if handler is None:
        raise CliError(
            f"Unknown command '{getattr(namespace, 'command', None)}'.",
            category="usage",
            context={"command": getattr(namespace, "command", None)},
        )

    try:
        result = handler(namespace, config=config)
    except CliError as exc:
        if not exc.logged:
            log_cli_error(exc.payload, exc_info=exc)
            exc.logged = True
        message = exc.payload.message if exc.payload else str(exc)
        if message:
            sys.stdout.write(message)
            if not message.endswith("\n"):
                sys.stdout.write("\n")
        raise SystemExit(exc.status_code) from exc
    if result:
        sys.stdout.write(result)
        if not result.endswith("\n"):
            sys.stdout.write("\n")
    return result


def main() -> None:  # pragma: no cover - thin wrapper
    run_cli()


if __name__ == "__main__":  # pragma: no cover - CLI invocation guard
    main()
