"""Persistence helpers for telemetry run logs."""

from __future__ import annotations

import gzip
import json
import math
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, Tuple

from ..core.epi import TelemetryRecord, _MISSING_FLOAT

RUNS_DIR = Path("runs")


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


def prepare_run_destination(
    *,
    car_model: str,
    track_name: str,
    output_dir: str | Path | None = None,
    suffix: str = ".jsonl",
    force: bool = False,
) -> Path:
    """Return a timestamped run path ensuring the parent directory exists.

    Parameters
    ----------
    car_model:
        Name of the car model used for slug generation.
    track_name:
        Name of the track used for slug generation.
    output_dir:
        Optional base directory for the run.  Defaults to :data:`RUNS_DIR`.
    suffix:
        File suffix to append to the generated name.  ``.jsonl`` by default.
    force:
        When ``True`` existing files are reused; otherwise ``FileExistsError``
        is raised on collisions.
    """

    directory = Path(output_dir).expanduser() if output_dir else RUNS_DIR
    timestamp = datetime.now(timezone.utc)
    car_slug = _slugify_token(car_model)
    track_slug = _slugify_token(track_name)
    destination = directory / f"{car_slug}_{track_slug}_{timestamp:%Y%m%d_%H%M%S_%f}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        raise FileExistsError(
            f"Telemetry run {destination} already exists. Use --force to overwrite."
        )
    return destination


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
    for key, value in list(values.items()):
        if isinstance(value, float) and math.isnan(value):
            values[key] = _MISSING_FLOAT
    return TelemetryRecord(**values)


def _slugify_token(token: str) -> str:
    normalised = token.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", normalised, flags=re.ASCII)
    slug = slug.strip("_")
    return slug or "run"


class DeterministicReplayer:
    """Simple container that yields telemetry samples deterministically."""

    def __init__(self, records: Iterable[TelemetryRecord]):
        self._records: Tuple[TelemetryRecord, ...] = tuple(records)

    def __iter__(self) -> Iterator[TelemetryRecord]:
        return iter(self._records)

    def iter(self) -> Iterator[TelemetryRecord]:
        """Return a new iterator over the stored telemetry sequence."""

        return iter(self._records)
