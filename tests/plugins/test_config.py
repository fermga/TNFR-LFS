"""Tests for :mod:`tnfr_lfs.plugins.config`."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from tnfr_lfs.plugins.config import PluginConfig, PluginConfigError
from tnfr_lfs.plugins.template import canonical_plugins_mapping, render_plugins_template
from tests.helpers import build_plugin_config_mapping, write_plugin_config_text


def test_plugin_config_from_pyproject_block(
    canonical_plugins_payload: tuple[dict[str, Any], Path]
) -> None:
    mapping, source_path = canonical_plugins_payload
    config = PluginConfig.from_mapping(mapping, source=source_path)

    assert config.auto_discover is True
    assert config.plugin_dir == (source_path.parent / "plugins").resolve()

    config.set_profile("practice")
    assert config.active_profile == "practice"
    assert config.max_concurrent == 2

    exporter = config.get_plugin_config("exporter")
    assert exporter["targets"] == ["html"]
    assert exporter["path"] == "out/practice-dashboard.html"
    assert exporter["template"] == "docs/templates/practice.html"

    config.set_profile("racing")
    assert set(config.enabled_plugins()) == {"telemetry", "exporter", "relay"}

    mapping["plugins"]["telemetry"]["flush_interval"] = "3s"
    config.reload_config()
    config.set_profile(None)
    telemetry = config.get_plugin_config("telemetry")
    assert telemetry["flush_interval"] == "3s"


def test_from_project_matches_mapping(tmp_path: Path) -> None:
    mapping = build_plugin_config_mapping(
        auto_discover=True,
        plugin_dir="plugins",
        max_concurrent=4,
        enabled=["telemetry"],
        plugins={
            "telemetry": {
                "entry_point": "pkg.telemetry:Plugin",
                "buffer_seconds": 10,
            },
            "exporter": {
                "enabled": False,
                "entry_point": "pkg.exporter:Plugin",
            },
        },
        profiles={
            "practice": {
                "plugins": ["telemetry"],
                "telemetry": {"enabled": True, "buffer_seconds": 5},
            }
        },
    )

    template_lines = render_plugins_template(mapping).strip().splitlines()
    pyproject_lines = [
        f"[tool.tnfr_lfs.{line[1:-1]}]" if line.startswith("[") else line
        for line in template_lines
    ]
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text("\n".join(pyproject_lines) + "\n")

    config_from_project = PluginConfig.from_project(
        pyproject_path, default_profile="practice"
    )
    config_from_mapping = PluginConfig.from_mapping(
        mapping, default_profile="practice", source=pyproject_path
    )

    assert config_from_project.active_profile == config_from_mapping.active_profile
    assert config_from_project.auto_discover == config_from_mapping.auto_discover
    assert config_from_project.max_concurrent == config_from_mapping.max_concurrent
    assert config_from_project.plugin_dir == config_from_mapping.plugin_dir
    assert set(config_from_project.enabled_plugins()) == set(
        config_from_mapping.enabled_plugins()
    )

    for plugin_name in ("telemetry", "exporter"):
        assert config_from_project.get_plugin_config(
            plugin_name
        ) == config_from_mapping.get_plugin_config(plugin_name)


def test_canonical_template_matches_pyproject_configuration(
    canonical_plugins_template_text: str,
) -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    from_pyproject = render_plugins_template(
        canonical_plugins_mapping(pyproject_path)[0]
    )
    assert from_pyproject == canonical_plugins_template_text


def test_reload_config_from_mapping_applies_updates(tmp_path: Path) -> None:
    mapping = build_plugin_config_mapping(
        auto_discover=False,
        plugin_dir="plugins",
        plugins={"telemetry": {"enabled": True, "buffer_seconds": 10}},
    )

    source = tmp_path / "pyproject.toml"
    config = PluginConfig.from_mapping(mapping, source=source)

    assert config.auto_discover is False
    assert config.plugin_dir == (source.parent / "plugins").resolve()

    telemetry = config.get_plugin_config("telemetry")
    assert telemetry["buffer_seconds"] == 10

    mapping["plugins"]["telemetry"]["buffer_seconds"] = 5
    config.reload_config()
    updated = config.get_plugin_config("telemetry")
    assert updated["buffer_seconds"] == 5


def test_from_mapping_requires_plugins_table(tmp_path: Path) -> None:
    with pytest.raises(PluginConfigError):
        PluginConfig.from_mapping({}, source=tmp_path / "pyproject.toml")


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
