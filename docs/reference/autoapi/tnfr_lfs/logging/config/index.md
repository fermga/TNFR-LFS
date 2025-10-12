# `tnfr_lfs.logging.config` module
Logging helpers for TNFR × LFS.

## Classes
### `JsonFormatter` (logging.Formatter)
Serialise log records as JSON payloads.

#### Methods
- `format(self, record: logging.LogRecord) -> str`

## Functions
- `setup_logging(config: Mapping[str, Any]) -> None`
  - Configure the root logger using a mapping of options.

