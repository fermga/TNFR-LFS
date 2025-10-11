"""Tests for :mod:`tnfr_lfs.plugins.config`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tnfr_lfs.plugins.config import PluginConfig, PluginConfigError
from tests.helpers import write_plugin_config_text


def test_plugin_config_loads_and_merges_global_settings(tmp_path: Path) -> None:
    config_path = write_plugin_config_text(
        tmp_path,
        """
        [plugins]
        auto_discover = true
        plugin_dir = "plugins"
        max_concurrent = 4
        enabled = ["telemetry", "exporter"]

        [plugins.telemetry]
        entry_point = "pkg.telemetry:Plugin"
        enabled = true
        buffer_seconds = 10

        [plugins.telemetry.transport]
        protocol = "udp"
        port = 3000

        [plugins.exporter]
        entry_point = "pkg.exporter:Plugin"

        [plugins.exporter.json]
        path = "out/export.json"

        [plugins.relay]
        enabled = false
        entry_point = "pkg.relay:Plugin"
        channel = "zeromq"

        [profiles.practice]
        plugins = ["telemetry", "exporter"]
        max_concurrent = 2

        [profiles.practice.telemetry]
        enabled = true
        buffer_seconds = 30

        [profiles.practice.exporter]
        enabled = true
        flush_on_shutdown = false
        """,
    )

    config = PluginConfig(config_path)

    assert config.auto_discover is True
    assert config.plugin_dir == (config_path.parent / "plugins").resolve()
    assert config.max_concurrent == 4

    telemetry_config = config.get_plugin_config("telemetry")
    exporter_config = config.get_plugin_config("exporter")
    relay_config = config.get_plugin_config("relay")

    assert telemetry_config["enabled"] is True
    assert telemetry_config["transport"]["protocol"] == "udp"
    assert exporter_config["enabled"] is True
    assert exporter_config["json"]["path"] == "out/export.json"
    assert relay_config["enabled"] is False

    enabled_plugins = config.enabled_plugins()
    assert set(enabled_plugins) == {"telemetry", "exporter"}


def test_plugin_config_profile_overrides(tmp_path: Path) -> None:
    config_path = write_plugin_config_text(
        tmp_path,
        """
        [plugins]
        auto_discover = false
        plugin_dir = "plugins"
        max_concurrent = 3

        [plugins.telemetry]
        enabled = true
        buffer_seconds = 10

        [plugins.exporter]
        enabled = true
        flush_on_shutdown = true

        [plugins.relay]
        enabled = false

        [profiles.racing]
        plugins = ["telemetry", "exporter", "relay"]
        max_concurrent = 1

        [profiles.racing.telemetry]
        buffer_seconds = 5

        [profiles.racing.relay]
        enabled = true
        channel = "zeromq"
        """,
    )

    config = PluginConfig(config_path)

    config.set_profile("racing")
    assert config.active_profile == "racing"
    assert config.max_concurrent == 1

    telemetry = config.get_plugin_config("telemetry")
    relay = config.get_plugin_config("relay")

    assert telemetry["buffer_seconds"] == 5
    assert relay["enabled"] is True
    assert relay["channel"] == "zeromq"
    assert set(config.enabled_plugins()) == {"telemetry", "exporter", "relay"}


def test_reload_config_preserves_active_profile(tmp_path: Path) -> None:
    config_path = write_plugin_config_text(
        tmp_path,
        """
        [plugins]
        auto_discover = true
        plugin_dir = "plugins"
        max_concurrent = 2

        [plugins.exporter]
        enabled = true
        flush_on_shutdown = true

        [profiles.practice]
        plugins = ["exporter"]
        max_concurrent = 1

        [profiles.practice.exporter]
        flush_on_shutdown = false
        """,
    )

    config = PluginConfig(config_path)
    config.set_profile("practice")

    exporter_before = config.get_plugin_config("exporter")

    config_path.write_text(
        textwrap.dedent(
            """
            [plugins]
            auto_discover = true
            plugin_dir = "plugins"
            max_concurrent = 2

            [plugins.exporter]
            enabled = true
            flush_on_shutdown = true

            [profiles.practice]
            plugins = ["exporter"]
            max_concurrent = 1

            [profiles.practice.exporter]
            flush_on_shutdown = true
            """
        )
    )

    config.reload_config()

    assert config.active_profile == "practice"
    exporter_after = config.get_plugin_config("exporter")
    assert exporter_before["flush_on_shutdown"] is False
    assert exporter_after["flush_on_shutdown"] is True


def test_reload_config_failure_retains_previous_state(tmp_path: Path) -> None:
    config_path = write_plugin_config_text(
        tmp_path,
        """
        [plugins]
        auto_discover = false

        [plugins.telemetry]
        enabled = true
        """,
    )

    config = PluginConfig(config_path)
    config.set_profile(None)
    baseline = config.get_plugin_config("telemetry")

    config_path.write_text("this is not valid toml")

    config.reload_config()

    assert config.get_plugin_config("telemetry") == baseline

    config_path.write_text("[plugins]\n")
    with pytest.raises(PluginConfigError):
        PluginConfig(config_path)
