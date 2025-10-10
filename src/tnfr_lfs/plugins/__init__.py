"""Plugin infrastructure for TNFR × LFS extensions."""

from .base import TNFRPlugin
from .interfaces import PluginContract, PluginMetadata
from .registry import (
    available_operator_identifiers,
    get_plugin_operator_requirements,
    iter_plugin_operator_requirements,
    PluginMetadataError,
    plugin_metadata,
    register_plugin_metadata,
)

__all__ = [
    "TNFRPlugin",
    "PluginContract",
    "PluginMetadata",
    "available_operator_identifiers",
    "get_plugin_operator_requirements",
    "iter_plugin_operator_requirements",
    "PluginMetadataError",
    "plugin_metadata",
    "register_plugin_metadata",
]
