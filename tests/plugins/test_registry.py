from __future__ import annotations

import pytest

from tnfr_lfs.plugins import (
    PluginMetadataError,
    TNFRPlugin,
    available_operator_identifiers,
    get_plugin_operator_requirements,
    iter_plugin_operator_requirements,
    plugin_metadata,
    register_plugin_metadata,
)


class _BasePlugin(TNFRPlugin):
    """Minimal plugin used to validate metadata handling."""


def test_register_plugin_metadata_records_requirements() -> None:
    register_plugin_metadata(
        _BasePlugin,
        operators=["emission_operator", "coherence_operator"],
    )

    requirements = get_plugin_operator_requirements(_BasePlugin)

    assert requirements == ("emission_operator", "coherence_operator")
    registry_map = dict(iter_plugin_operator_requirements())
    assert registry_map[_BasePlugin] == requirements


def test_plugin_metadata_decorator_registers_class() -> None:
    @plugin_metadata(operators=["reception_operator", "coherence_operator"])
    class DecoratedPlugin(TNFRPlugin):
        """Plugin declared through the decorator helper."""

    requirements = get_plugin_operator_requirements(DecoratedPlugin)

    assert requirements == ("reception_operator", "coherence_operator")


def test_re_registration_with_different_metadata_fails() -> None:
    register_plugin_metadata(_BasePlugin, operators=["emission_operator"])

    with pytest.raises(PluginMetadataError):
        register_plugin_metadata(_BasePlugin, operators=["coherence_operator"])


def test_invalid_operator_identifier_raises() -> None:
    available = available_operator_identifiers()
    assert "emission_operator" in available

    with pytest.raises(PluginMetadataError):
        register_plugin_metadata(_BasePlugin, operators=["not_an_operator"])


def test_lookup_for_unregistered_plugin_raises() -> None:
    class UnregisteredPlugin(TNFRPlugin):
        """Plugin intentionally not registered with the metadata registry."""

    with pytest.raises(LookupError):
        get_plugin_operator_requirements(UnregisteredPlugin)


def test_duplicate_operator_entries_are_collapsed() -> None:
    register_plugin_metadata(
        _BasePlugin,
        operators=[
            "coherence_operator",
            "coherence_operator",
            "reception_operator",
        ],
    )

    requirements = get_plugin_operator_requirements(_BasePlugin)

    assert requirements == ("coherence_operator", "reception_operator")
