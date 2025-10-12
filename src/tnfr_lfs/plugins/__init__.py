"""Plugin infrastructure for TNFR × LFS extensions."""

from collections.abc import Sequence
from pathlib import Path

from .base import TNFRPlugin
from .interfaces import PluginContract, PluginMetadata
from .config import PluginConfig, PluginConfigError
from .registry import (
    available_operator_identifiers,
    get_plugin_operator_requirements,
    iter_plugin_operator_requirements,
    PluginMetadataError,
    plugin_metadata,
    register_plugin_metadata,
)

def plugin_config_from_project(
    pyproject_path: Path | None = None,
    *,
    default_profile: str | None = None,
    legacy_candidates: Sequence[Path] | None = None,
) -> PluginConfig:
    """Return a :class:`PluginConfig` built from ``pyproject.toml`` metadata."""

    return PluginConfig.from_project(
        pyproject_path,
        default_profile=default_profile,
        legacy_candidates=legacy_candidates,
    )


__all__ = [
    "TNFRPlugin",
    "PluginContract",
    "PluginMetadata",
    "PluginConfig",
    "PluginConfigError",
    "available_operator_identifiers",
    "get_plugin_operator_requirements",
    "iter_plugin_operator_requirements",
    "PluginMetadataError",
    "plugin_config_from_project",
    "plugin_metadata",
    "register_plugin_metadata",
]
