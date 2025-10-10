"""Error helpers for the TNFR × LFS command line tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional

__all__ = [
    "CliError",
    "ErrorPayload",
    "build_error_payload",
    "log_cli_error",
]


_CATEGORY_STATUS_CODES: Mapping[str, int] = {
    "runtime": 1,
    "usage": 2,
    "io": 3,
    "not_found": 4,
}

_DEFAULT_CATEGORY = "runtime"
_DEFAULT_LOGGER_NAME = "tnfr_lfs.cli"


@dataclass(frozen=True, slots=True)
class ErrorPayload:
    """Structured representation of an error emitted by the CLI."""

    status_code: int
    category: str
    message: str
    context: Mapping[str, Any]

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "status_code": self.status_code,
            "category": self.category,
            "message": self.message,
            "context": dict(self.context),
        }


def _normalise_context(context: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not context:
        return {}
    payload: MutableMapping[str, Any] = {}
    for key, value in context.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
        else:
            payload[key] = str(value)
    return dict(payload)


def build_error_payload(
    message: str,
    *,
    category: str = _DEFAULT_CATEGORY,
    status_code: Optional[int] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> ErrorPayload:
    """Create a :class:`ErrorPayload` describing a CLI failure."""

    resolved_category = category or _DEFAULT_CATEGORY
    resolved_status = (
        status_code
        if status_code is not None
        else _CATEGORY_STATUS_CODES.get(resolved_category, _CATEGORY_STATUS_CODES[_DEFAULT_CATEGORY])
    )
    safe_context = _normalise_context(context)
    return ErrorPayload(
        status_code=resolved_status,
        category=resolved_category,
        message=message,
        context=safe_context,
    )


def log_cli_error(
    payload: ErrorPayload,
    *,
    logger: Optional[logging.Logger] = None,
    exc_info: Optional[BaseException] = None,
) -> None:
    """Emit ``payload`` using ``logger.error`` with structured context."""

    target = logger or logging.getLogger(_DEFAULT_LOGGER_NAME)
    target.error(
        payload.message,
        extra={
            "event": "cli.error",
            "category": payload.category,
            "status_code": payload.status_code,
            "context": dict(payload.context),
        },
        exc_info=exc_info,
    )


class CliError(RuntimeError):
    """Consistent error type raised by CLI helpers."""

    __slots__ = ("category", "status_code", "context", "_payload", "logged")

    def __init__(
        self,
        message: str,
        *,
        category: str = _DEFAULT_CATEGORY,
        status_code: Optional[int] = None,
        context: Optional[Mapping[str, Any]] = None,
        payload: Optional[ErrorPayload] = None,
        logged: bool = False,
    ) -> None:
        super().__init__(message)
        self.category = category or _DEFAULT_CATEGORY
        resolved_payload = payload or build_error_payload(
            message,
            category=self.category,
            status_code=status_code,
            context=context,
        )
        self.status_code = resolved_payload.status_code
        self.context = dict(resolved_payload.context)
        self._payload = resolved_payload
        self.logged = logged

    @property
    def payload(self) -> ErrorPayload:
        return self._payload

    @classmethod
    def from_context(
        cls,
        message: str,
        *,
        category: str = _DEFAULT_CATEGORY,
        status_code: Optional[int] = None,
        context: Optional[Mapping[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        cause: Optional[BaseException] = None,
    ) -> "CliError":
        payload = build_error_payload(
            message,
            category=category,
            status_code=status_code,
            context=context,
        )
        log_cli_error(payload, logger=logger, exc_info=cause)
        return cls(
            message,
            category=category,
            status_code=payload.status_code,
            context=payload.context,
            payload=payload,
            logged=True,
        )
