"""Configuration utilities for the plugin subsystem."""

from __future__ import annotations

import copy
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping

try:  # pragma: no cover - Python < 3.11 fallback
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


from tnfr_lfs.configuration import load_project_plugins_config


logger = logging.getLogger(__name__)


class PluginConfigError(RuntimeError):
    """Raised when the plugin configuration is invalid."""


@dataclass(frozen=True)
class _ProfileData:
    """Internal representation of a profile configuration."""

    settings: Dict[str, Any]
    plugins: tuple[str, ...] | None
    overrides: Dict[str, Dict[str, Any]]
    extends: tuple[str, ...] = ()


class PluginConfig:
    """Load, validate and expose plugin configuration information."""

    def __init__(
        self,
        path: str | Path,
        *,
        default_profile: str | None = None,
        _injected_data: Mapping[str, Any] | None = None,
    ) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._raw: Dict[str, Any] = {}
        self._settings: Dict[str, Any] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}
        self._profiles: Dict[str, _ProfileData] = {}
        self._globally_enabled: set[str] = set()
        self._active_profile: str | None = None
        self._injected_data: Mapping[str, Any] | None = _injected_data
        self._data_loader: Callable[[], Dict[str, Any]] = self._read_file
        self._uses_injected_data = _injected_data is not None

        if self._uses_injected_data:
            self._data_loader = self._load_injected_data

        if self._uses_injected_data:
            if self._path == Path("<memory>"):
                self._source_description = "in-memory mapping"
            else:
                self._source_description = f"in-memory mapping ({self._path})"
        else:
            self._source_description = f"'{self._path}'"

        self.reload_config(initial=True)

        if default_profile is not None:
            self.set_profile(default_profile)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        default_profile: str | None = None,
        source: str | Path | None = None,
    ) -> "PluginConfig":
        """Construct a configuration instance from ``data``.

        Parameters
        ----------
        data:
            Mapping containing the configuration structure as if it were parsed
            from ``plugins.toml``.
        default_profile:
            Optional profile name to activate after validation.
        source:
            Optional path describing the origin of ``data``. When provided it is
            used for relative path resolution (``plugin_dir``) and logging. If
            omitted, a placeholder ``"<memory>"`` path is used.
        """

        source_path = Path(source) if source is not None else Path("<memory>")
        return cls(source_path, default_profile=default_profile, _injected_data=data)

    @classmethod
    def from_project(
        cls,
        pyproject_path: Path | None = None,
        *,
        default_profile: str | None = None,
    ) -> "PluginConfig":
        """Construct a configuration from ``pyproject.toml`` metadata."""

        if pyproject_path is None:
            pyproject_path = Path.cwd()

        loaded = load_project_plugins_config(pyproject_path)
        if loaded is None:
            raise PluginConfigError(
                "Unable to locate '[tool.tnfr_lfs.plugins]' in project configuration"
            )

        mapping, source_path = loaded
        return cls.from_mapping(
            mapping,
            default_profile=default_profile,
            source=source_path,
        )

    @property
    def path(self) -> Path:
        """Return the resolved configuration file path."""

        return self._path

    @property
    def active_profile(self) -> str | None:
        """Name of the currently active profile if any."""

        with self._lock:
            return self._active_profile

    @property
    def available_profiles(self) -> tuple[str, ...]:
        """Return the names of profiles defined in the configuration."""

        with self._lock:
            return tuple(self._profiles.keys())

    @property
    def auto_discover(self) -> bool:
        """Flag controlling whether plugin auto-discovery is enabled."""

        with self._lock:
            value = self._resolve_setting("auto_discover", False)
            assert isinstance(value, bool)
            return value

    @property
    def plugin_dir(self) -> Path:
        """Return the directory to scan for plugins."""

        with self._lock:
            raw = self._resolve_setting("plugin_dir", None)
            if raw is None:
                candidate = self._path.parent / "plugins"
            else:
                if not isinstance(raw, str):
                    raise PluginConfigError("'plugin_dir' must be a string path")
                candidate = Path(raw)
                if not candidate.is_absolute():
                    candidate = (self._path.parent / candidate).resolve()

            return candidate

    @property
    def max_concurrent(self) -> int:
        """Return the configured concurrency limit (``0`` disables the limit)."""

        with self._lock:
            value = self._resolve_setting("max_concurrent", 0)
            if not isinstance(value, int) or value < 0:
                raise PluginConfigError("'max_concurrent' must be a non-negative integer")
            return value

    def set_profile(self, profile_name: str | None) -> None:
        """Activate ``profile_name`` or clear the active profile when ``None``."""

        with self._lock:
            if profile_name is None:
                if self._active_profile is not None:
                    logger.info("Deactivated plugin profile '%s'", self._active_profile)
                self._active_profile = None
                return

            if profile_name not in self._profiles:
                logger.error("Attempted to activate unknown plugin profile '%s'", profile_name)
                raise KeyError(f"Unknown plugin profile '{profile_name}'")

            self._active_profile = profile_name
            logger.info("Activated plugin profile '%s'", profile_name)

    def reload_config(self, *, initial: bool = False) -> None:
        """Reload the configuration from disk applying validation atomically."""

        with self._lock:
            try:
                new_raw = self._data_loader()
                new_settings, new_plugins, new_profiles, new_enabled = self._validate(new_raw)
            except PluginConfigError:
                logger.exception(
                    "Failed to parse plugin configuration from %s", self._source_description
                )
                if initial:
                    raise
                return

            self._raw = new_raw
            self._settings = new_settings
            self._plugin_configs = new_plugins
            self._profiles = new_profiles
            self._globally_enabled = new_enabled

            if self._active_profile and self._active_profile not in self._profiles:
                logger.warning(
                    "Active profile '%s' no longer present after reload; clearing.",
                    self._active_profile,
                )
                self._active_profile = None

            logger.info("Reloaded plugin configuration from %s", self._source_description)

    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Return the merged configuration for ``plugin_name`` respecting profiles."""

        with self._lock:
            if plugin_name not in self._plugin_configs:
                raise KeyError(f"Plugin '{plugin_name}' is not defined in configuration")

            base = copy.deepcopy(self._plugin_configs[plugin_name])
            enabled = bool(base.get("enabled", plugin_name in self._globally_enabled))

            profile = self._current_profile()
            if profile is not None:
                plugin_list = profile.plugins
                if plugin_list is not None and plugin_name not in plugin_list:
                    enabled = False

                overrides = profile.overrides.get(plugin_name)
                if overrides:
                    base = self._merge_dicts(base, overrides)
                    if "enabled" in overrides:
                        enabled = bool(overrides["enabled"])

            base["enabled"] = enabled
            return base

    def enabled_plugins(self) -> tuple[str, ...]:
        """Return the plugin identifiers that are enabled for the active profile."""

        with self._lock:
            result: list[str] = []
            for name in self._plugin_configs:
                try:
                    config = self.get_plugin_config(name)
                except KeyError:
                    continue
                if config.get("enabled") is True:
                    result.append(name)
            return tuple(result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _read_file(self) -> Dict[str, Any]:
        if not self._path.exists():
            raise PluginConfigError(f"Configuration file '{self._path}' does not exist")

        try:
            with self._path.open("rb") as stream:
                return tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:  # type: ignore[attr-defined]
            raise PluginConfigError("Unable to load plugin configuration") from exc

    def _load_injected_data(self) -> Dict[str, Any]:
        if self._injected_data is None:
            raise PluginConfigError("Injected configuration is not available")

        if not isinstance(self._injected_data, Mapping):
            raise PluginConfigError("Injected configuration must be a mapping")

        return copy.deepcopy(dict(self._injected_data))

    def _validate(
        self, data: Mapping[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, _ProfileData], set[str]]:
        plugins_section = data.get("plugins")
        if not isinstance(plugins_section, MutableMapping):
            raise PluginConfigError("Configuration is missing a '[plugins]' table")

        settings: Dict[str, Any] = {}
        plugin_configs: Dict[str, Dict[str, Any]] = {}
        globally_enabled: set[str] = set()

        for key, value in plugins_section.items():
            if key == "auto_discover":
                if not isinstance(value, bool):
                    raise PluginConfigError("'[plugins].auto_discover' must be a boolean")
                settings[key] = value
                continue

            if key == "plugin_dir":
                if not isinstance(value, str):
                    raise PluginConfigError("'[plugins].plugin_dir' must be a string path")
                settings[key] = value
                continue

            if key == "max_concurrent":
                if not isinstance(value, int) or value < 0:
                    raise PluginConfigError("'[plugins].max_concurrent' must be a non-negative integer")
                settings[key] = value
                continue

            if key == "enabled":
                if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                    raise PluginConfigError("'[plugins].enabled' must be an array of plugin names")
                globally_enabled = set(value)
                continue

            if not isinstance(value, MutableMapping):
                raise PluginConfigError(
                    f"Plugin '{key}' must be represented as a table in '[plugins]'"
                )

            plugin_config = copy.deepcopy(value)
            enabled_value = plugin_config.get("enabled")
            if enabled_value is None:
                plugin_config["enabled"] = key in globally_enabled
            elif not isinstance(enabled_value, bool):
                raise PluginConfigError(
                    f"'[plugins.{key}].enabled' must be a boolean value"
                )

            plugin_configs[key] = plugin_config

        if not plugin_configs:
            raise PluginConfigError("At least one plugin must be defined under '[plugins]'")

        # Normalise defaults when omitted
        settings.setdefault("auto_discover", False)
        settings.setdefault("plugin_dir", None)
        settings.setdefault("max_concurrent", 0)

        profiles_data = self._parse_profiles(data.get("profiles"), plugin_configs)
        profiles_data = self._resolve_profile_inheritance(profiles_data)

        # Ensure the global enabled list only references known plugins
        unknown_globally_enabled = globally_enabled - set(plugin_configs.keys())
        if unknown_globally_enabled:
            raise PluginConfigError(
                "'[plugins].enabled' references unknown plugins: "
                + ", ".join(sorted(unknown_globally_enabled))
            )

        return settings, plugin_configs, profiles_data, globally_enabled

    def _parse_profiles(
        self,
        profiles_section: Any,
        plugin_configs: Mapping[str, Dict[str, Any]],
    ) -> Dict[str, _ProfileData]:
        if profiles_section is None:
            return {}

        if not isinstance(profiles_section, MutableMapping):
            raise PluginConfigError("'[profiles]' must be a table of profile definitions")

        profiles: Dict[str, _ProfileData] = {}

        for profile_name, profile_data in profiles_section.items():
            if not isinstance(profile_data, MutableMapping):
                raise PluginConfigError(
                    f"Profile '{profile_name}' must be represented as a table"
                )

            extends_field = profile_data.get("extends")
            if extends_field is None:
                extends: tuple[str, ...] = ()
            elif isinstance(extends_field, str):
                extends = (extends_field,)
            elif isinstance(extends_field, list) and all(
                isinstance(item, str) for item in extends_field
            ):
                extends = tuple(extends_field)
            else:
                raise PluginConfigError(
                    f"'[profiles.{profile_name}].extends' must be a string or an array of profile names"
                )

            plugins_field = profile_data.get("plugins")
            plugins: tuple[str, ...] | None
            if plugins_field is None:
                plugins = None
            elif isinstance(plugins_field, list) and all(isinstance(item, str) for item in plugins_field):
                plugins = tuple(plugins_field)
            else:
                raise PluginConfigError(
                    f"'[profiles.{profile_name}].plugins' must be an array of plugin names"
                )

            settings: Dict[str, Any] = {}
            overrides: Dict[str, Dict[str, Any]] = {}

            for key, value in profile_data.items():
                if key in {"plugins", "extends"}:
                    continue

                if isinstance(value, MutableMapping):
                    overrides[key] = copy.deepcopy(value)
                    continue

                settings[key] = value

            if "auto_discover" in settings and not isinstance(settings["auto_discover"], bool):
                raise PluginConfigError(
                    f"'[profiles.{profile_name}].auto_discover' must be a boolean"
                )

            if "plugin_dir" in settings and not isinstance(settings["plugin_dir"], str):
                raise PluginConfigError(
                    f"'[profiles.{profile_name}].plugin_dir' must be a string"
                )

            if "max_concurrent" in settings:
                value = settings["max_concurrent"]
                if not isinstance(value, int) or value < 0:
                    raise PluginConfigError(
                        f"'[profiles.{profile_name}].max_concurrent' must be a non-negative integer"
                    )

            unknown_plugins = set(overrides) - set(plugin_configs)
            if unknown_plugins:
                raise PluginConfigError(
                    f"Profile '{profile_name}' references unknown plugins: "
                    + ", ".join(sorted(unknown_plugins))
                )

            for plugin, override in overrides.items():
                if "enabled" in override and not isinstance(override["enabled"], bool):
                    raise PluginConfigError(
                        f"'[profiles.{profile_name}.{plugin}].enabled' must be a boolean"
                    )

            profiles[profile_name] = _ProfileData(
                settings=settings,
                plugins=plugins,
                overrides=overrides,
                extends=extends,
            )

        return profiles

    def _resolve_profile_inheritance(
        self, profiles: Dict[str, _ProfileData]
    ) -> Dict[str, _ProfileData]:
        """Resolve ``extends`` relationships among profiles."""

        resolved: Dict[str, _ProfileData] = {}
        resolving: set[str] = set()

        def resolve(profile_name: str) -> _ProfileData:
            if profile_name in resolved:
                return resolved[profile_name]

            if profile_name in resolving:
                raise PluginConfigError(
                    f"Circular inheritance detected while resolving profile '{profile_name}'"
                )

            resolving.add(profile_name)
            profile = profiles[profile_name]

            merged_settings: Dict[str, Any] = {}
            merged_plugins: tuple[str, ...] | None = None
            merged_overrides: Dict[str, Dict[str, Any]] = {}

            for parent_name in profile.extends:
                if parent_name not in profiles:
                    raise PluginConfigError(
                        f"Profile '{profile_name}' extends unknown profile '{parent_name}'"
                    )
                parent = resolve(parent_name)
                merged_settings = self._merge_dicts(merged_settings, parent.settings)
                if parent.plugins is not None:
                    merged_plugins = parent.plugins
                for plugin_name, override in parent.overrides.items():
                    merged_overrides[plugin_name] = copy.deepcopy(override)

            merged_settings = self._merge_dicts(merged_settings, profile.settings)
            if profile.plugins is not None:
                merged_plugins = profile.plugins

            for plugin_name, override in profile.overrides.items():
                base_override = merged_overrides.get(plugin_name, {})
                merged_overrides[plugin_name] = self._merge_dicts(base_override, override)

            resolving.remove(profile_name)

            resolved_profile = _ProfileData(
                settings=merged_settings,
                plugins=merged_plugins,
                overrides=merged_overrides,
                extends=(),
            )
            resolved[profile_name] = resolved_profile
            return resolved_profile

        for name in profiles:
            resolve(name)

        return resolved

    def _resolve_setting(self, key: str, default: Any) -> Any:
        value = self._settings.get(key, default)

        profile = self._current_profile()
        if profile is not None and key in profile.settings:
            value = profile.settings[key]

        return value

    def _current_profile(self) -> _ProfileData | None:
        if self._active_profile is None:
            return None

        return self._profiles.get(self._active_profile)

    def _merge_dicts(self, base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(base)
        for key, value in overrides.items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = self._merge_dicts(result[key], value)  # type: ignore[index]
            else:
                result[key] = copy.deepcopy(value)
        return result


__all__ = ["PluginConfig", "PluginConfigError"]

