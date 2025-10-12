"""Utilities to derive the canonical plugin configuration templates."""

from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tnfr_lfs.configuration import load_project_plugins_config


def canonical_plugins_mapping(
    project_root: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Return the canonical plugin configuration mapping and its source path."""

    pyproject_path = _locate_pyproject(project_root)
    loaded = load_project_plugins_config(pyproject_path)
    if loaded is None:
        raise RuntimeError(
            "Unable to locate '[tool.tnfr_lfs.plugins]' in the project configuration"
        )

    mapping, source = loaded
    return copy.deepcopy(mapping), source


def canonical_plugins_template(project_root: Path | None = None) -> str:
    """Render the canonical plugin template using ``pyproject.toml`` as source."""

    mapping, _ = canonical_plugins_mapping(project_root)
    return render_plugins_template(mapping)


def render_plugins_template(mapping: Mapping[str, Any]) -> str:
    """Serialise ``mapping`` into the ``plugins.toml`` template format."""

    plugins_table = mapping.get("plugins")
    if not isinstance(plugins_table, Mapping):
        raise ValueError("'plugins' table is required to render the template")

    lines: list[str] = []
    _emit_toml_table(lines, "plugins", plugins_table)

    profiles_table = mapping.get("profiles")
    if isinstance(profiles_table, Mapping) and profiles_table:
        if any(not isinstance(value, Mapping) for value in profiles_table.values()):
            if lines and lines[-1] != "":
                lines.append("")
            _emit_toml_table(lines, "profiles", profiles_table)
        else:
            for key, value in profiles_table.items():
                if lines and lines[-1] != "":
                    lines.append("")
                _emit_toml_table(lines, f"profiles.{key}", value)

    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point to render the canonical ``plugins.toml`` template."""

    parser = argparse.ArgumentParser(
        description=(
            "Render the canonical plugins configuration derived from pyproject.toml"
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Directory containing pyproject.toml (defaults to repository root)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination file. Writes to stdout when omitted.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    template_text = canonical_plugins_template(args.project_root)

    if args.output is None:
        sys.stdout.write(template_text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(template_text)

    return 0


def _locate_pyproject(project_root: Path | None) -> Path:
    if project_root is not None:
        project_root = project_root.expanduser()
        candidate = project_root / "pyproject.toml" if project_root.is_dir() else project_root
        if not candidate.exists():
            raise FileNotFoundError(f"Unable to find pyproject.toml at '{candidate}'")
        return candidate

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            return candidate

    raise RuntimeError("Unable to locate pyproject.toml relative to the current module")


def _emit_toml_table(lines: list[str], table_name: str, values: Mapping[str, Any]) -> None:
    lines.append(f"[{table_name}]")
    for key, value in values.items():
        if isinstance(value, Mapping):
            if lines and lines[-1] != "":
                lines.append("")
            _emit_toml_table(lines, f"{table_name}.{key}", value)
            continue
        lines.append(f"{key} = {_format_toml_value(value)}")


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, Path):
        return f'"{value.as_posix()}"'
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"

    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    raise SystemExit(main())


__all__ = [
    "canonical_plugins_mapping",
    "canonical_plugins_template",
    "main",
    "render_plugins_template",
]
