"""Tests for :mod:`tnfr_lfs.plugins.manager`."""

from __future__ import annotations

import textwrap
from pathlib import Path

from tnfr_lfs.plugins.manager import PluginManager
from tnfr_lfs.plugins.config import PluginConfig
from tnfr_lfs.plugins import registry
from tnfr_lfs.plugins.base import TNFRPlugin

from tests.helpers import write_plugin_manager_config, write_plugin_module


def test_discover_plugins_registers_valid_plugins(tmp_path: Path) -> None:
    write_plugin_module(
        tmp_path,
        module_name="sample_plugin",
        class_name="SamplePlugin",
        body="""
        def analyze(self, payload):
            return {"status": "ok"}
        """,
    )

    manager = PluginManager()
    discovered = manager.discover_plugins(tmp_path)

    assert discovered
    key, registration = next(iter(discovered.items()))
    assert key in manager.plugin_registry
    assert issubclass(registration.cls, TNFRPlugin)
    assert registration.operators == ("emission_operator",)


def test_load_and_unload_plugin(tmp_path: Path) -> None:
    write_plugin_module(
        tmp_path,
        module_name="loader_plugin",
        class_name="LoaderPlugin",
        body="""
        def analyze(self, payload):
            return payload
        """,
    )

    manager = PluginManager()
    manager.discover_plugins(tmp_path)
    plugin_name = next(iter(manager.plugin_registry))

    instance = manager.load_plugin(plugin_name)
    assert instance.identifier == "loaderplugin"

    manager.unload_plugin(plugin_name)
    assert plugin_name not in manager.plugins


def test_execute_analysis_collects_results_and_errors(tmp_path: Path) -> None:
    write_plugin_module(
        tmp_path,
        module_name="success_plugin",
        class_name="SuccessPlugin",
        body="""
        def analyze(self, payload):
            return {"value": payload["number"]}
        """,
    )

    write_plugin_module(
        tmp_path,
        module_name="failing_plugin",
        class_name="FailingPlugin",
        body="""
        def analyze(self, payload):
            raise RuntimeError("boom")
        """,
    )

    manager = PluginManager()
    manager.discover_plugins(tmp_path)

    for plugin_name in list(manager.plugin_registry):
        manager.load_plugin(plugin_name)

    payload = {"number": 42}
    outcome = manager.execute_analysis(payload)

    assert "results" in outcome and "errors" in outcome
    assert len(outcome["results"]) == 1
    assert len(outcome["errors"]) == 1
    success_result = next(iter(outcome["results"].values()))
    assert success_result == {"value": 42}


def test_get_plugin_health_reports_loaded_state(tmp_path: Path) -> None:
    write_plugin_module(
        tmp_path,
        module_name="health_plugin",
        class_name="HealthPlugin",
        body="""
        def analyze(self, payload):
            return {}
        """,
    )

    manager = PluginManager()
    manager.discover_plugins(tmp_path)
    plugin_name = next(iter(manager.plugin_registry))
    manager.load_plugin(plugin_name)

    health = manager.get_plugin_health()
    assert plugin_name in health
    plugin_health = health[plugin_name]
    assert plugin_health["loaded"] is True
    assert plugin_health["required_operators"] == ("emission_operator",)
    assert plugin_health["missing_operators"] == ()


def test_discovery_skips_plugins_without_metadata(tmp_path: Path) -> None:
    module_path = tmp_path / "invalid_plugin.py"
    module_path.write_text(
        textwrap.dedent(
            """
            from tnfr_lfs.plugins.base import TNFRPlugin

            class InvalidPlugin(TNFRPlugin):
                def __init__(self):
                    super().__init__(identifier='invalid', display_name='Invalid', version='1.0')

                def analyze(self, payload):
                    return {}
            """
        )
    )

    manager = PluginManager()
    manager.discover_plugins(tmp_path)

    assert not manager.plugin_registry


def test_execute_analysis_respects_max_concurrent_from_config(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    write_plugin_module(
        plugin_dir,
        module_name="first_plugin",
        class_name="FirstPlugin",
        body="""
        def analyze(self, payload):
            return {"name": "first"}
        """,
    )

    write_plugin_module(
        plugin_dir,
        module_name="second_plugin",
        class_name="SecondPlugin",
        body="""
        def analyze(self, payload):
            return {"name": "second"}
        """,
    )

    config_path = write_plugin_manager_config(
        tmp_path,
        plugin_dir=plugin_dir,
        plugins={
            "first": {"enabled": True},
            "second": {"enabled": True},
        },
        profiles={
            "single": {
                "plugins": ["first", "second"],
                "max_concurrent": 1,
            }
        },
    )
    config = PluginConfig(config_path)
    config.set_profile("single")

    manager = PluginManager(config=config)

    assert manager.plugin_registry

    for plugin_name in list(manager.plugin_registry):
        manager.load_plugin(plugin_name)

    result = manager.execute_analysis({"value": 42})

    assert len(manager.plugins) == 2
    assert len(result["results"]) == 1
    assert result["errors"] == {}
