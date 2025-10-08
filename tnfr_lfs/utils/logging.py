"""Logging helpers for TNFR × LFS."""

from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping

_RESERVED_RECORD_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Serialise log records as JSON payloads."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - thin wrapper
        payload: MutableMapping[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_FIELDS
        }
        if extras:
            payload["extra"] = _serialise_extra(extras)
        return json.dumps(payload, default=_json_default, ensure_ascii=False)


def _serialise_extra(extras: Mapping[str, Any]) -> Mapping[str, Any]:
    serialised: dict[str, Any] = {}
    for key, value in extras.items():
        if isinstance(value, Mapping):
            serialised[key] = _serialise_extra(value)
            continue
        serialised[key] = _json_default(value)
    return serialised


def _json_default(value: Any) -> Any:  # pragma: no cover - basic coercions
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return repr(value)


def _coerce_level(value: Any, default: int = logging.INFO) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        candidate = logging.getLevelName(value.upper())
        if isinstance(candidate, int):
            return candidate
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed
    return default


def _resolve_logging_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "logging" in config and isinstance(config.get("logging"), Mapping):
        return config["logging"]  # type: ignore[return-value]
    return config


def setup_logging(config: Mapping[str, Any]) -> None:
    """Configure the root logger using a mapping of options."""

    options = dict(_resolve_logging_config(config))
    level = _coerce_level(options.get("level"), logging.INFO)
    output_value = options.get("output", "stderr")
    normalised_output = (
        output_value.strip().lower() if isinstance(output_value, str) else None
    )
    formatter = str(options.get("format", "json")).lower()

    if normalised_output == "stderr" or output_value is None:
        handler_config: dict[str, Any] = {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        }
    elif normalised_output == "stdout":
        handler_config = {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        }
    else:
        handler_config = {
            "class": "logging.FileHandler",
            "filename": str(output_value),
            "encoding": "utf8",
        }

    if formatter == "text":
        formatter_config = {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        }
        formatter_name = "text"
    else:
        formatter_config = {
            "()": "tnfr_lfs.utils.logging.JsonFormatter",
        }
        formatter_name = "json"

    handler_config["formatter"] = formatter_name

    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            formatter_name: formatter_config,
        },
        "handlers": {
            "default": handler_config,
        },
        "root": {
            "handlers": ["default"],
            "level": level,
        },
    }

    logging.captureWarnings(True)
    logging.config.dictConfig(config_dict)


__all__ = ["JsonFormatter", "setup_logging"]

