# `tnfr_lfs.plugins` package
Plugin infrastructure for TNFR × LFS extensions.

## Submodules
- [`tnfr_lfs.plugins.base`](base/index.md)
- [`tnfr_lfs.plugins.config`](config/index.md)
- [`tnfr_lfs.plugins.interfaces`](interfaces/index.md)
- [`tnfr_lfs.plugins.manager`](manager/index.md)
- [`tnfr_lfs.plugins.registry`](registry/index.md)
- [`tnfr_lfs.plugins.template`](template/index.md)

## Functions
- `plugin_config_from_project(pyproject_path: Path | None = None, *, default_profile: str | None = None) -> PluginConfig`
  - Return a :class:`PluginConfig` built from ``pyproject.toml`` metadata.

