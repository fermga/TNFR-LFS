# `tnfr_lfs.configuration` module
Helpers to load project-level configuration files.

## Functions
- `load_project_config(path: Path) -> tuple[dict[str, Any], Path] | None`
  - Load `[tool.tnfr_lfs]` from ``pyproject.toml``.
- `load_project_plugins_config(path: Path) -> tuple[dict[str, Any], Path] | None`
  - Expose the ``[tool.tnfr_lfs.plugins]`` block from ``pyproject.toml``.

