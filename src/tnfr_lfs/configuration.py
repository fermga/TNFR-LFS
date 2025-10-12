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


def write_legacy_cli_config(
    pyproject_path: Path, *, output_path: Path | None = None
) -> Path:
    """Persist the canonical CLI configuration into ``tnfr_lfs.toml``.

    Parameters
    ----------
    pyproject_path:
        Path to a ``pyproject.toml`` file or its parent directory.
    output_path:
        Optional location for the generated legacy file. When omitted, the
        helper writes alongside ``pyproject.toml`` using the default
        ``tnfr_lfs.toml`` filename.

    Returns
    -------
    pathlib.Path
        The resolved path of the generated legacy configuration file.
    """

    resolved_pyproject = _resolve_pyproject_path(pyproject_path)
    if resolved_pyproject is None:
        raise ValueError(f"Unable to resolve pyproject.toml from {pyproject_path!s}")

    resolved_pyproject = resolved_pyproject.expanduser().resolve(strict=True)

    canonical_block = canonical_cli_config_block(resolved_pyproject)
    legacy_lines: list[str] = []
    prefix = "tool.tnfr_lfs"

    for line in canonical_block.splitlines():
        stripped = line.strip()
        if stripped.startswith("[[tool.tnfr_lfs"):
            raise ValueError(
                "Legacy configuration generation does not support array tables "
                "within [tool.tnfr_lfs]."
            )
        if stripped.startswith("[tool.tnfr_lfs") and stripped.endswith("]"):
            table_name = stripped[1:-1]
            if table_name == prefix:
                # ``[tool.tnfr_lfs]`` would introduce an empty legacy table; skip it.
                line = ""
            elif table_name.startswith(f"{prefix}."):
                suffix = table_name[len(prefix) + 1 :]
                indentation = line[: len(line) - len(line.lstrip())]
                line = f"{indentation}[{suffix}]"
            else:  # pragma: no cover - unexpected format guard
                raise ValueError(
                    "Encountered an unexpected table while generating legacy configuration: "
                    f"{table_name}"
                )
        if line:
            legacy_lines.append(line)
        else:
            legacy_lines.append("")

    legacy_block = "\n".join(legacy_lines).rstrip() + "\n"

    if output_path is not None:
        target_path = Path(output_path).expanduser().resolve(strict=False)
    else:
        target_path = resolved_pyproject.with_name(_DEFAULT_LEGACY_FILENAME)

    header_lines = [
        "# Auto-generated from the canonical [tool.tnfr_lfs] block in pyproject.toml.",
        "# Regenerate via tnfr_lfs.configuration.write_legacy_cli_config(pyproject_path).",
        "# Legacy consumers may continue reading tnfr_lfs.toml; new tooling reads pyproject.toml.",
    ]
    header_text = "\n".join(header_lines).rstrip() + "\n\n"

    target_path.write_text(
        header_text + legacy_block,
        encoding="utf8",
    )

    return target_path


__all__ = [
    "canonical_cli_config_block",
    "load_project_config",
    "load_project_plugins_config",
    "write_legacy_cli_config",
]
