# `tnfr_lfs.cli.io` module
Telemetry and configuration helpers for the TNFR × LFS CLI.

## Functions
- `load_cli_config(path: Optional[Path] = None) -> Dict[str, Any]`
  - Load CLI defaults from ``pyproject.toml`` files.

## Attributes
- `CONFIG_ENV_VAR = 'TNFR_LFS_CONFIG'`
- `PROJECT_CONFIG_FILENAME = 'pyproject.toml'`
- `Records = List[TelemetryRecord]`

