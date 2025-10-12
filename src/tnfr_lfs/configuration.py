"""Helpers to load project-level configuration files."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping as ABCMapping, Sequence
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore


_DEFAULT_LEGACY_FILENAME = "tnfr_lfs.toml"
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


def _deep_merge(base: ABCMapping[str, Any], overlay: ABCMapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` returning a new mapping."""

    merged = {str(key): value for key, value in base.items()}
    for key, overlay_value in overlay.items():
        key_str = str(key)
        base_value = merged.get(key_str)
        if isinstance(base_value, ABCMapping) and isinstance(overlay_value, ABCMapping):
            merged[key_str] = _deep_merge(base_value, overlay_value)
        else:
            merged[key_str] = overlay_value
    return merged


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


def load_project_config(
    path: Path,
    *,
    legacy_candidates: Sequence[Path] | None = None,
) -> tuple[dict[str, Any], Path] | None:
    """Load `[tool.tnfr_lfs]` from ``pyproject.toml`` combining legacy overrides."""

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

    legacy_paths: list[Path] = []
    if legacy_candidates is not None:
        legacy_paths.extend(legacy_candidates)
    legacy_paths.append(pyproject_path.with_name(_DEFAULT_LEGACY_FILENAME))

    merged: dict[str, Any] = {}
    for legacy_path in _iter_unique_paths(legacy_paths):
        if legacy_path == pyproject_path:
            continue
        legacy_payload = _load_toml_mapping(legacy_path)
        if not legacy_payload:
            continue
        merged = _deep_merge(merged, legacy_payload)

    merged = _deep_merge(merged, config)
    return merged, pyproject_path


def load_project_plugins_config(
    path: Path,
    *,
    legacy_candidates: Sequence[Path] | None = None,
) -> tuple[dict[str, Any], Path] | None:
    """Expose the ``[tool.tnfr_lfs.plugins]`` block from ``pyproject.toml``."""

    loaded = load_project_config(path, legacy_candidates=legacy_candidates)
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


def canonical_cli_config_block(pyproject_path: Path) -> str:
    """Return the raw ``[tool.tnfr_lfs]`` snippet from ``pyproject.toml``.

    The helper keeps documentation snippets and the legacy ``tnfr_lfs.toml``
    template in sync with the canonical configuration.
    """

    text = pyproject_path.read_text(encoding="utf8")
    lines = text.splitlines()
    capture_prefix = "[tool.tnfr_lfs"
    captured: list[str] = []
    capturing = False
    for line in lines:
        if line.startswith(capture_prefix):
            capturing = True
        elif capturing and line.startswith("[tool.") and not line.startswith(
            capture_prefix
        ):
            break
        if capturing:
            captured.append(line.rstrip())

    while captured and captured[0].strip() == "":
        captured.pop(0)
    while captured and captured[-1].strip() == "":
        captured.pop()

    if not captured:
        raise ValueError(f"No [tool.tnfr_lfs] section found in {pyproject_path}")

    return "\n".join(captured)


__all__ = [
    "canonical_cli_config_block",
    "load_project_config",
    "load_project_plugins_config",
]
