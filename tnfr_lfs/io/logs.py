"""Persistence helpers for telemetry run logs."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, Tuple

from ..core.epi import TelemetryRecord


def write_run(
    records: Sequence[TelemetryRecord],
    path: str | Path,
    *,
    compress: bool = True,
) -> None:
    """Persist ``records`` to ``path`` using a newline-delimited JSON format.

    Parameters
    ----------
    records:
        Sequence of :class:`~tnfr_lfs.core.epi.TelemetryRecord` objects to serialise.
    path:
        Destination file.  Parent directories are created automatically.
    compress:
        When ``True`` the payload is compressed using gzip.  ``iter_run`` is able
        to transparently read both compressed and uncompressed payloads.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compress else open

    with opener(destination, "wt", encoding="utf8") as handle:
        for record in records:
            json.dump(asdict(record), handle, sort_keys=True)
            handle.write("\n")


def iter_run(path: str | Path) -> Iterator[TelemetryRecord]:
    """Yield telemetry samples previously persisted with :func:`write_run`."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Telemetry run {source} does not exist")

    try:
        with gzip.open(source, "rt", encoding="utf8") as handle:
            yield from _iter_records(handle)
        return
    except OSError:
        pass

    with source.open("r", encoding="utf8") as handle:
        yield from _iter_records(handle)


def _iter_records(handle: Iterable[str]) -> Iterator[TelemetryRecord]:
    for line in handle:
        if not line.strip():
            continue
        yield _decode_record(json.loads(line))


def _decode_record(payload: dict[str, Any]) -> TelemetryRecord:
    values = dict(payload)
    reference = values.get("reference")
    if isinstance(reference, dict):
        values["reference"] = _decode_record(reference)
    elif reference is None or isinstance(reference, TelemetryRecord):
        values["reference"] = reference
    else:
        raise TypeError("Invalid reference payload for TelemetryRecord")
    return TelemetryRecord(**values)


class DeterministicReplayer:
    """Simple container that yields telemetry samples deterministically."""

    def __init__(self, records: Iterable[TelemetryRecord]):
        self._records: Tuple[TelemetryRecord, ...] = tuple(records)

    def __iter__(self) -> Iterator[TelemetryRecord]:
        return iter(self._records)

    def iter(self) -> Iterator[TelemetryRecord]:
        """Return a new iterator over the stored telemetry sequence."""

        return iter(self._records)
