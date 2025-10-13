"""Plugin infrastructure for TNFR × LFS extensions."""

from pathlib import Path

from tnfr_lfs.plugins.base import TNFRPlugin
from tnfr_lfs.plugins.interfaces import PluginContract, PluginMetadata
from tnfr_lfs.plugins.config import PluginConfig, PluginConfigError
from tnfr_lfs.plugins.registry import (
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
) -> PluginConfig:
    """Return a :class:`PluginConfig` built from ``pyproject.toml`` metadata."""

    return PluginConfig.from_project(
        pyproject_path,
        default_profile=default_profile,
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
