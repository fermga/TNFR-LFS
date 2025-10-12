# `tnfr_lfs.plugins.manager` module
Plugin manager responsible for discovery and lifecycle operations.

## Classes
### `RegisteredPlugin`
Metadata describing a discovered plugin class.

### `PluginManager`
Manage discovery, loading and execution of TNFR × LFS plugins.

#### Methods
- `discover_plugins(self, plugin_dir: str | Path) -> Dict[str, RegisteredPlugin]`
  - Discover plugins located in ``plugin_dir``.

Parameters
----------
plugin_dir:
    Directory containing plugin modules or packages.

Returns
-------
Dict[str, RegisteredPlugin]
    Mapping of qualified plugin names to registration metadata.
- `load_plugin(self, plugin_name: str, *args: Any, **kwargs: Any) -> TNFRPlugin`
  - Load and instantiate the plugin identified by ``plugin_name``.
- `unload_plugin(self, plugin_name: str) -> None`
  - Unload the plugin instance identified by ``plugin_name``.
- `execute_analysis(self, payload: Mapping[str, Any] | None = None) -> Dict[str, Any]`
  - Execute ``analyze`` on all loaded plugins returning results and errors.
- `get_plugin_health(self) -> Dict[str, Dict[str, Any]]`
  - Return health information for registered plugins.

## Attributes
- `logger = logging.getLogger(__name__)`

