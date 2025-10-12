# `tnfr_lfs.plugins.template` module
Utilities to derive the canonical plugin configuration templates.

## Functions
- `canonical_plugins_mapping(project_root: Path | None = None) -> tuple[dict[str, Any], Path]`
  - Return the canonical plugin configuration mapping and its source path.
- `canonical_plugins_template(project_root: Path | None = None) -> str`
  - Render the canonical plugin template using ``pyproject.toml`` as source.
- `render_plugins_template(mapping: Mapping[str, Any]) -> str`
  - Serialise ``mapping`` into the ``plugins.toml`` template format.
- `main(argv: Sequence[str] | None = None) -> int`
  - CLI entry point to render the canonical ``plugins.toml`` template.

