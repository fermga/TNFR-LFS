# `tnfr_lfs.resources.paths` module
Helpers to locate bundled TNFR × LFS resources.

## Functions
- `set_pack_root_override(path: Path | None) -> None`
  - Force :func:`pack_root` to return ``path`` (used in tests).
- `pack_root() -> Path`
  - Return the directory that contains the installed pack resources.
- `data_root() -> Path`
  - Return the directory containing the pack's ``data`` tree.
- `modifiers_root() -> Path`
  - Return the directory containing packaged combo modifiers.

