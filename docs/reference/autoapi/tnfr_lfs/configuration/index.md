# `tnfr_lfs.configuration` module
Helpers to load project-level configuration files.

## Functions
- `load_project_config(path: Path, *, legacy_candidates: Sequence[Path] | None = None) -> tuple[dict[str, Any], Path] | None`
  - Load `[tool.tnfr_lfs]` from ``pyproject.toml`` combining legacy overrides.
- `load_project_plugins_config(path: Path, *, legacy_candidates: Sequence[Path] | None = None) -> tuple[dict[str, Any], Path] | None`
  - Expose the ``[tool.tnfr_lfs.plugins]`` block from ``pyproject.toml``.
- `canonical_cli_config_block(pyproject_path: Path) -> str`
  - Return the raw ``[tool.tnfr_lfs]`` snippet from ``pyproject.toml``.

The helper keeps documentation snippets and the legacy ``tnfr_lfs.toml``
template in sync with the canonical configuration.
- `canonical_cli_config_legacy_text(pyproject_path: Path) -> str`
  - Return the canonical legacy ``tnfr_lfs.toml`` contents.
- `write_legacy_cli_config(pyproject_path: Path, *, output_path: Path | None = None) -> Path`
  - Persist the canonical CLI configuration into ``tnfr_lfs.toml``.

Parameters
----------
pyproject_path:
    Path to a ``pyproject.toml`` file or its parent directory.
output_path:
    Optional location for the generated legacy file. When omitted, the
    helper writes alongside ``pyproject.toml`` using the default
    ``tnfr_lfs.toml`` filename.

Returns
-------
pathlib.Path
    The resolved path of the generated legacy configuration file.

