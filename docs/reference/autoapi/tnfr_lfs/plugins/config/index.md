# `tnfr_lfs.plugins.config` module
Configuration utilities for the plugin subsystem.

## Classes
### `PluginConfigError` (RuntimeError)
Raised when the plugin configuration is invalid.

### `PluginConfig`
Load, validate and expose plugin configuration information.

#### Methods
- `from_mapping(cls, data: Mapping[str, Any], *, default_profile: str | None = None, source: str | Path | None = None) -> 'PluginConfig'`
  - Construct a configuration instance from ``data``.

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
- `from_project(cls, pyproject_path: Path | None = None, *, default_profile: str | None = None) -> 'PluginConfig'`
  - Construct a configuration from ``pyproject.toml`` metadata.
- `path(self) -> Path`
  - Return the resolved configuration file path.
- `active_profile(self) -> str | None`
  - Name of the currently active profile if any.
- `available_profiles(self) -> tuple[str, ...]`
  - Return the names of profiles defined in the configuration.
- `auto_discover(self) -> bool`
  - Flag controlling whether plugin auto-discovery is enabled.
- `plugin_dir(self) -> Path`
  - Return the directory to scan for plugins.
- `max_concurrent(self) -> int`
  - Return the configured concurrency limit (``0`` disables the limit).
- `set_profile(self, profile_name: str | None) -> None`
  - Activate ``profile_name`` or clear the active profile when ``None``.
- `reload_config(self, *, initial: bool = False) -> None`
  - Reload the configuration from disk applying validation atomically.
- `get_plugin_config(self, plugin_name: str) -> Dict[str, Any]`
  - Return the merged configuration for ``plugin_name`` respecting profiles.
- `enabled_plugins(self) -> tuple[str, ...]`
  - Return the plugin identifiers that are enabled for the active profile.

## Attributes
- `logger = logging.getLogger(__name__)`

