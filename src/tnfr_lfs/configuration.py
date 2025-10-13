"""Helpers to load project-level configuration files."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping as ABCMapping
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore


_PROJECT_FILENAME = "pyproject.toml"


def _as_dict(payload: ABCMapping[str, Any]) -> dict[str, Any]:
    """Recursively coerce TOML mappings into regular dictionaries."""

    result: dict[str, Any] = {}
    for key, value in payload.items():
        key_str = str(key)
        if isinstance(value, ABCMapping):
            result[key_str] = _as_dict(value)
        elif isinstance(value, list):
            result[key_str] = [
                _as_dict(item) if isinstance(item, ABCMapping) else item for item in value
            ]
        else:
            result[key_str] = value
    return result


def _resolve_pyproject_path(candidate: Path) -> Path | None:
    """Return the concrete ``pyproject.toml`` path for ``candidate`` if possible."""

    candidate = candidate.expanduser()
    if candidate.name == _PROJECT_FILENAME:
        return candidate
    if candidate.suffix:
        return None
    return candidate / _PROJECT_FILENAME


def _iter_unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: dict[Path, None] = {}
    ordered: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve(strict=False)
        if resolved in seen:
            continue
        seen[resolved] = None
        ordered.append(resolved)
    return ordered


def _load_toml_mapping(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if isinstance(data, ABCMapping):
        return _as_dict(data)
    return None


def load_project_config(path: Path) -> tuple[dict[str, Any], Path] | None:
    """Load the `[tool.tnfr_lfs]` section from ``pyproject.toml``."""

    pyproject_path = _resolve_pyproject_path(path)
    if pyproject_path is None:
        return None

    pyproject_path = pyproject_path.expanduser().resolve(strict=False)
    pyproject_payload = _load_toml_mapping(pyproject_path)
    if not pyproject_payload:
        return None

    tool_section = pyproject_payload.get("tool")
    if not isinstance(tool_section, ABCMapping):
        return None

    cli_section = tool_section.get("tnfr_lfs")
    if not isinstance(cli_section, ABCMapping):
        return None

    config = _as_dict(cli_section)

    core_section = tool_section.get("tnfr_core")
    if isinstance(core_section, ABCMapping):
        config.setdefault("core", _as_dict(core_section))

    return config, pyproject_path


def load_project_plugins_config(
    path: Path,
) -> tuple[dict[str, Any], Path] | None:
    """Expose the ``[tool.tnfr_lfs.plugins]`` block from ``pyproject.toml``."""

    loaded = load_project_config(path)
    if loaded is None:
        return None

    config, source_path = loaded

    plugins_table = config.get("plugins")
    if not isinstance(plugins_table, ABCMapping):
        return None

    payload: dict[str, Any] = {"plugins": copy.deepcopy(dict(plugins_table))}

    profiles_table = config.get("profiles")
    if isinstance(profiles_table, ABCMapping):
        payload["profiles"] = copy.deepcopy(dict(profiles_table))

    return payload, source_path

__all__ = [
    "load_project_config",
    "load_project_plugins_config",
]
