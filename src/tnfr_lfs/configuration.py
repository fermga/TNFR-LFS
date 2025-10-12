"""Helpers to load project-level configuration files."""

from __future__ import annotations

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


__all__ = ["load_project_config"]
