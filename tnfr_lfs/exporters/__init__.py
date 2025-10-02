"""Exporter registry for TNFR-LFS outputs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Protocol

from ..core.epi import EPIResult


class Exporter(Protocol):
    """Exporter callable protocol."""

    def __call__(self, results: Dict[str, Any]) -> str:  # pragma: no cover - interface only
        ...


def _normalise(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalise(item) for key, item in value.items()}
    return value


def json_exporter(results: Dict[str, Any]) -> str:
    import json

    payload = _normalise(results)
    return json.dumps(payload, indent=2, sort_keys=True)


def csv_exporter(results: Dict[str, Any]) -> str:
    from io import StringIO

    buffer = StringIO()
    buffer.write("timestamp,epi,delta_nfr,delta_si\n")
    for result in results.get("series", []):
        if not isinstance(result, EPIResult):
            raise TypeError("CSV exporter expects EPIResult instances")
        buffer.write(
            f"{result.timestamp:.3f},{result.epi:.4f},{result.delta_nfr:.3f},{result.delta_si:.3f}\n"
        )
    return buffer.getvalue()


exporters_registry = {
    "json": json_exporter,
    "csv": csv_exporter,
}

__all__ = ["Exporter", "json_exporter", "csv_exporter", "exporters_registry"]
