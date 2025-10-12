# `tnfr_lfs.cli.errors` module
Error helpers for the TNFR × LFS command line tools.

## Classes
### `ErrorPayload`
Structured representation of an error emitted by the CLI.

#### Methods
- `as_dict(self) -> Mapping[str, Any]`

### `CliError` (RuntimeError)
Consistent error type raised by CLI helpers.

#### Methods
- `payload(self) -> ErrorPayload`
- `from_context(cls, message: str, *, category: str = _DEFAULT_CATEGORY, status_code: Optional[int] = None, context: Optional[Mapping[str, Any]] = None, logger: Optional[logging.Logger] = None, cause: Optional[BaseException] = None) -> 'CliError'`

## Functions
- `build_error_payload(message: str, *, category: str = _DEFAULT_CATEGORY, status_code: Optional[int] = None, context: Optional[Mapping[str, Any]] = None) -> ErrorPayload`
  - Create a :class:`ErrorPayload` describing a CLI failure.
- `log_cli_error(payload: ErrorPayload, *, logger: Optional[logging.Logger] = None, exc_info: Optional[BaseException] = None) -> None`
  - Emit ``payload`` using ``logger.error`` with structured context.

