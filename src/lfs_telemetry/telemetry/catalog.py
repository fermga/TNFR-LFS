"""Fast capture catalog for the workspace browser.

Scans a directory of CSV captures and returns lightweight metadata
records *without* loading the bodies. The MoTeC-style app uses this to
populate its left-pane file list.

The scan reads:

* the schema preamble (1st line, optional),
* the header row (column order),
* the first and last data rows (for car / track / lap-time / distance),

so cost is O(1) per file regardless of size.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .replay import detect_schema_version


@dataclass(frozen=True, slots=True)
class CaptureInfo:
    """One capture file's quick metadata."""

    path: Path
    schema_version: str | None
    car: str | None
    track: str | None
    samples: int
    lap_time_s: float | None
    distance_m: float | None
    file_size_bytes: int
    mtime: float           # POSIX timestamp (file modification time)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


def discover_captures(
    directory: str | Path,
    pattern: str = "*.csv",
    *,
    recursive: bool = False,
) -> list[CaptureInfo]:
    """Scan ``directory`` and return a list of :class:`CaptureInfo`.

    Files that fail to parse as LFS-Telemetry captures are skipped silently
    (they may be foreign CSVs that happened to match ``pattern``).
    """
    directory = Path(directory)
    glob = directory.rglob if recursive else directory.glob
    out: list[CaptureInfo] = []
    for path in sorted(glob(pattern)):
        if not path.is_file():
            continue
        info = inspect_capture(path)
        if info is not None:
            out.append(info)
    return out


def inspect_capture(path: str | Path) -> CaptureInfo | None:
    """Read enough of ``path`` to fill a :class:`CaptureInfo`.

    Returns ``None`` if the file is unreadable or lacks the minimum
    columns to qualify as a lfs-telemetry capture (``time_ms`` is required).

    Truly O(1) per file: reads the preamble + header + first data row,
    counts newlines via a chunked binary scan (I/O bound, no per-row
    Python overhead), and reverse-scans the tail for the last data row.
    Results are memoized by ``(path, mtime, size)`` so repeated workspace
    refreshes are essentially free for unchanged files.
    """
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return None

    cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _INSPECT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    schema = detect_schema_version(path)

    try:
        with path.open("r", encoding="utf-8", newline="") as fp:
            # Skip preamble (``#``-prefixed lines).
            pos = fp.tell()
            line = fp.readline()
            while line.startswith("#"):
                pos = fp.tell()
                line = fp.readline()
            fp.seek(pos)
            reader = csv.DictReader(fp)
            fieldnames = reader.fieldnames
            if fieldnames is None or "time_ms" not in fieldnames:
                return None
            first = next(reader, None)
            if first is None:
                info = CaptureInfo(
                    path=path, schema_version=schema, car=None, track=None,
                    samples=0, lap_time_s=None, distance_m=None,
                    file_size_bytes=stat.st_size, mtime=stat.st_mtime,
                )
                _INSPECT_CACHE[cache_key] = info
                return info

        # Sample count = newlines in the file minus header (and any
        # preamble lines). Fast chunked binary read.
        newline_total = 0
        with path.open("rb") as fb:
            while True:
                chunk = fb.read(1 << 16)
                if not chunk:
                    break
                newline_total += chunk.count(b"\n")
        # Subtract header + preamble lines we skipped above.
        # ``pos`` points at the start of the header line, so the number
        # of skipped (preamble) lines equals the count of '\n' in the
        # bytes [0, pos). The header itself adds one more '\n' to skip.
        with path.open("rb") as fb:
            preamble_bytes = fb.read(pos)
        preamble_newlines = preamble_bytes.count(b"\n")
        # If the very last byte isn't a newline the final row still
        # contributes one sample, so clamp to >=1 when we have ``first``.
        count = max(1, newline_total - preamble_newlines - 1)

        # Last data row: read the trailing ~8 KB and parse the final
        # non-empty line through the same DictReader fieldnames.
        last = first
        try:
            with path.open("rb") as fb:
                tail_size = min(stat.st_size, 8192)
                fb.seek(stat.st_size - tail_size)
                tail = fb.read(tail_size).decode("utf-8", errors="ignore")
            # Drop trailing newlines, take the final line.
            tail_lines = [ln for ln in tail.splitlines() if ln.strip()]
            if tail_lines and len(tail_lines) >= 1:
                last_row_text = tail_lines[-1]
                # Parse it with csv against the header fieldnames.
                last_reader = csv.DictReader(
                    [",".join(fieldnames), last_row_text],
                    fieldnames=fieldnames,
                )
                next(last_reader, None)  # header
                parsed = next(last_reader, None)
                if parsed is not None:
                    last = parsed
        except (OSError, csv.Error, UnicodeDecodeError):
            pass
    except (OSError, csv.Error, UnicodeDecodeError):
        return None

    car = (first.get("car") or "").strip() or None
    track = (first.get("ctx_track") or "").strip() or None

    lap_time_s: float | None = None
    try:
        t0 = float(first["time_ms"])
        t1 = float(last["time_ms"])
        lap_time_s = (t1 - t0) / 1000.0
    except (TypeError, ValueError, KeyError):
        pass

    # ``current_lap_dist_m`` is monotonically non-decreasing along a lap
    # so first/last bound the distance covered.
    d0 = _try_float(first.get("current_lap_dist_m"))
    d1 = _try_float(last.get("current_lap_dist_m"))
    distance_m: float | None = None
    if d0 is not None and d1 is not None:
        distance_m = max(0.0, d1 - d0)

    info = CaptureInfo(
        path=path,
        schema_version=schema,
        car=car,
        track=track,
        samples=count,
        lap_time_s=lap_time_s,
        distance_m=distance_m,
        file_size_bytes=stat.st_size,
        mtime=stat.st_mtime,
    )
    _INSPECT_CACHE[cache_key] = info
    return info


# Memoize by (path, mtime_ns, size) so unchanged files skip all I/O on
# subsequent ``Reload`` clicks.
_INSPECT_CACHE: dict[tuple[str, int, int], CaptureInfo] = {}


def captures_to_dataframe(items: Iterable[CaptureInfo]):
    """Render a list of :class:`CaptureInfo` as a ``pandas.DataFrame``.

    Imported lazily so :mod:`pandas` is not required just for the scan.
    """
    import pandas as pd  # local import keeps catalog dependency-light
    return pd.DataFrame([i.as_dict() for i in items])


def _try_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
