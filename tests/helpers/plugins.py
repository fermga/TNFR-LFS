"""Test helpers for managing the plugin metadata registry and scaffolding."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Tuple
import textwrap

from tnfr_lfs.plugins.base import TNFRPlugin
from tnfr_lfs.plugins import register_plugin_metadata
from tnfr_lfs.plugins.registry import _clear_registry
from tnfr_lfs.plugins.template import render_plugins_template

PluginRegistration = Tuple[type[TNFRPlugin], Sequence[str]]
RegisterPlugin = Callable[[type[TNFRPlugin], Sequence[str]], type[TNFRPlugin]]


@contextmanager
def plugin_registry_state(
    initial: Iterable[PluginRegistration] | None = None,
) -> Iterator[RegisterPlugin]:
    """Manage the plugin registry state for tests.

    The registry is cleared on entry and exit. Optionally, ``initial`` registrations
    can be supplied as an iterable of ``(plugin_cls, operators)`` tuples. The yielded
    callable allows tests to register additional plugin metadata during the managed
    context.
    """

    if initial is None:
        initial = ()

    def register(plugin_cls: type[TNFRPlugin], operators: Sequence[str]) -> type[TNFRPlugin]:
        return register_plugin_metadata(plugin_cls, operators=operators)

    try:
        _clear_registry()
        for plugin_cls, operators in initial:
            register(plugin_cls, operators)
        yield register
    finally:
        _clear_registry()


def write_plugin_config_text(base_directory: Path, content: str) -> Path:
    """Write a ``plugins.toml`` file with the provided ``content``."""

    base_directory.mkdir(parents=True, exist_ok=True)
    config_path = base_directory / "plugins.toml"
    config_path.write_text(textwrap.dedent(content))
    return config_path


def build_plugin_config_mapping(
    *,
    auto_discover: bool = True,
    plugin_dir: str | Path | None = None,
    max_concurrent: int = 0,
    enabled: Sequence[str] | None = None,
    plugins: Mapping[str, Mapping[str, Any]] | None = None,
    profiles: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construct an in-memory representation of ``plugins.toml`` data."""

    plugin_settings: dict[str, Any] = {
        "auto_discover": auto_discover,
        "max_concurrent": max_concurrent,
    }

    if plugin_dir is not None:
        plugin_settings["plugin_dir"] = (
            plugin_dir.as_posix() if isinstance(plugin_dir, Path) else plugin_dir
        )

    if enabled is not None:
        plugin_settings["enabled"] = list(enabled)

    if plugins:
        for name, settings in plugins.items():
            plugin_settings[name] = copy.deepcopy(dict(settings))

    mapping: dict[str, Any] = {"plugins": plugin_settings}

    if profiles:
        mapping["profiles"] = {
            name: copy.deepcopy(dict(settings)) for name, settings in profiles.items()
        }

    return mapping


def write_plugin_module(
    directory: Path,
    *,
    module_name: str,
    class_name: str,
    body: str,
    operators: Sequence[str] = ("emission_operator",),
    identifier: str | None = None,
    display_name: str | None = None,
    version: str = "1.0",
) -> Path:
    """Create a plugin module on disk for discovery focused tests.

    Parameters
    ----------
    directory:
        Target directory where the module should be written.
    module_name:
        Name of the module (without the ``.py`` suffix).
    class_name:
        Name of the plugin implementation class.
    body:
        Body of the plugin class. This should contain the method definitions the
        test needs (e.g. ``def analyze(...): ...``). The content is automatically
        de-dented and indented to live within the class scope.
    operators:
        Iterable of operator identifiers to register with the plugin metadata
        decorator. Defaults to ``("emission_operator",)`` which is the minimal
        set required by most manager tests.
    identifier, display_name, version:
        Optional overrides for the values passed to ``TNFRPlugin`` during
        construction. When omitted they default to values derived from the class
        name (``class_name.lower()`` and ``class_name`` respectively) and a
        version of ``"1.0"``.

    Returns
    -------
    pathlib.Path
        The path to the generated module. This allows tests to perform further
        assertions or include the module in discovery directories directly.
    """

    directory.mkdir(parents=True, exist_ok=True)

    identifier = identifier or class_name.lower()
    display_name = display_name or class_name
    operator_list = ", ".join(f'"{op}"' for op in operators)

    lines = [
        "from tnfr_lfs.plugins.base import TNFRPlugin",
        "from tnfr_lfs.plugins import registry",
        "",
        f"@registry.plugin_metadata(operators=[{operator_list}])",
        f"class {class_name}(TNFRPlugin):",
        "    def __init__(self):",
        "        super().__init__(",
        f'            identifier="{identifier}",',
        f'            display_name="{display_name}",',
        f'            version="{version}",',
        "        )",
        "",
    ]

    body_lines = textwrap.dedent(body).strip("\n").splitlines()
    if not body_lines:
        body_lines = ["pass"]
    lines.extend(f"    {line}" for line in body_lines)

    module_path = directory / f"{module_name}.py"
    module_path.write_text("\n".join(lines) + "\n")
    return module_path


def write_plugin_manager_config(
    directory: Path,
    *,
    config_name: str = "plugins.toml",
    plugin_dir: Path | None = None,
    auto_discover: bool = True,
    max_concurrent: int = 0,
    plugins: Mapping[str, Mapping[str, Any]] | None = None,
    profiles: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Author a TOML configuration file tailored for plugin manager tests.

    The helper mirrors :class:`tnfr_lfs.plugins.config.PluginConfig` defaults
    whilst allowing callers to override the pieces they care about. A minimal
    example that writes a configuration enabling two plugins and a single
    profile looks like::

        write_plugin_manager_config(
            tmp_path,
            plugin_dir=plugin_dir,
            plugins={"first": {"enabled": True}, "second": {"enabled": True}},
            profiles={
                "single": {
                    "plugins": ["first", "second"],
                    "max_concurrent": 1,
                }
            },
        )

    Parameters
    ----------
    directory:
        Base path where the configuration file should be written.
    config_name:
        Name of the TOML file. Defaults to ``"plugins.toml"``.
    plugin_dir:
        Optional path to the plugin directory that should be encoded in the
        configuration. When provided it is serialised using POSIX semantics to
        match the expectations of :class:`PluginConfig`.
    auto_discover, max_concurrent:
        Values for the top-level ``[plugins]`` section.
    plugins:
        Mapping of plugin identifiers to the key/value pairs that should be
        written under ``[plugins.<name>]`` sections.
    profiles:
        Mapping of profile identifiers to the corresponding configuration to be
        written under ``[profiles.<name>]`` sections.

    Returns
    -------
    pathlib.Path
        The path to the generated TOML configuration.
    """

    mapping = build_plugin_config_mapping(
        auto_discover=auto_discover,
        plugin_dir=plugin_dir,
        max_concurrent=max_concurrent,
        plugins=plugins,
        profiles=profiles,
    )

    config_path = directory / config_name
    config_path.write_text(render_plugins_template(mapping))
    return config_path
