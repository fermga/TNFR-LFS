from __future__ import annotations

import pytest

from tnfr_lfs.plugins import (
    PluginContract,
    PluginMetadata,
    TNFRPlugin,
    register_plugin_metadata,
)
from tnfr_lfs.plugins.registry import _clear_registry


@pytest.fixture(autouse=True)
def clear_registry() -> None:
    _clear_registry()
    register_plugin_metadata(
        _RegisteredPlugin,
        operators=["coherence_operator", "emission_operator"],
    )
    yield
    _clear_registry()


class _RegisteredPlugin(TNFRPlugin):
    """Minimal plugin used to exercise metadata and contract helpers."""


def test_plugin_metadata_normalises_sequences() -> None:
    metadata = PluginMetadata(
        identifier="telemetry",
        version="1.0.0",
        description="Telemetry enrichment",
        required_operators=["coherence_operator", "coherence_operator", "emission_operator"],
        optional_dependencies=["redis", "numpy", "redis"],
    )

    assert metadata.identifier == "telemetry"
    assert metadata.version == "1.0.0"
    assert metadata.required_operators == (
        "coherence_operator",
        "emission_operator",
    )
    assert metadata.optional_dependencies == ("redis", "numpy")


def test_plugin_metadata_from_plugin_uses_registry_requirements() -> None:
    metadata = PluginMetadata.from_plugin(
        _RegisteredPlugin,
        identifier="registered-plugin",
        version="2.3.1",
        description="Provides registered functionality",
    )

    assert metadata.required_operators == (
        "coherence_operator",
        "emission_operator",
    )
    assert metadata.identifier == "registered-plugin"
    assert metadata.version == "2.3.1"


def test_plugin_contract_validates_against_registry() -> None:
    metadata = PluginMetadata.from_plugin(
        _RegisteredPlugin,
        identifier="registered-plugin",
        version="2.3.1",
    )

    contract = PluginContract(plugin_cls=_RegisteredPlugin, metadata=metadata)

    assert contract.plugin_cls is _RegisteredPlugin
    assert contract.metadata == metadata


def test_plugin_contract_factory_reuses_metadata_from_plugin() -> None:
    contract = PluginContract.from_plugin(
        _RegisteredPlugin,
        identifier="registered-plugin",
        version="2.3.1",
        optional_dependencies=["redis"],
    )

    assert contract.metadata.optional_dependencies == ("redis",)
    assert contract.metadata.required_operators == (
        "coherence_operator",
        "emission_operator",
    )


def test_mismatched_metadata_raises_value_error() -> None:
    invalid_metadata = PluginMetadata(
        identifier="registered-plugin",
        version="2.3.1",
        required_operators=["coherence_operator"],
    )

    with pytest.raises(ValueError):
        PluginContract(plugin_cls=_RegisteredPlugin, metadata=invalid_metadata)
