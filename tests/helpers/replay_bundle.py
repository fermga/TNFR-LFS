"""Utilities for interacting with replay bundle readers in tests."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import pandas as pd

from tnfr_core.epi import TelemetryRecord
from tnfr_lfs.telemetry.offline import ReplayCSVBundleReader

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only.
    import pytest


def read_reference_rows(
    path: Path | str,
    *,
    distance_column: str = "d",
    value_column: str = "test1",
    limit: int | None = 5,
) -> list[tuple[float, float]]:
    """Read reference telemetry rows from ``path``.

    Parameters
    ----------
    path:
        Path to the CSV file containing reference telemetry data.
    distance_column, value_column:
        Column names containing the distance and value measurements.
    limit:
        Maximum number of rows to read from the file. ``None`` reads all rows.
    """

    csv_path = Path(path)
    rows: list[tuple[float, float]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((float(row[distance_column]), float(row[value_column])))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def is_numeric_series(series: pd.Series) -> bool:
    """Return ``True`` if ``series`` has a numeric dtype."""

    return pd.api.types.is_numeric_dtype(series)


def is_numeric_value(value: object) -> bool:
    """Return ``True`` if ``value`` is a numeric scalar (excluding booleans)."""

    if isinstance(value, bool):  # Guard against bool being subclass of int.
        return False
    if isinstance(value, (int, float)):
        numeric = float(value)
        return math.isfinite(numeric) or math.isnan(numeric)
    return False


@dataclass
class RowToRecordCounter:
    """Shared state tracking calls to :meth:`ReplayCSVBundleReader._row_to_record`."""

    count: int = 0


@contextmanager
def monkeypatch_row_to_record_counter(
    monkeypatch: "pytest.MonkeyPatch",
) -> Iterator[RowToRecordCounter]:
    """Patch ``ReplayCSVBundleReader._row_to_record`` and track call count.

    The provided ``monkeypatch`` fixture is used to replace the implementation with a
    counting wrapper. The original implementation is restored when the context exits.
    """

    original = ReplayCSVBundleReader._row_to_record
    counter = RowToRecordCounter()

    def _counting_row_to_record(
        self: ReplayCSVBundleReader, row: tuple[object, ...], column_index: Mapping[str, int]
    ) -> TelemetryRecord:
        counter.count += 1
        return original(self, row, column_index)

    monkeypatch.setattr(ReplayCSVBundleReader, "_row_to_record", _counting_row_to_record)
    try:
        yield counter
    finally:
        setattr(ReplayCSVBundleReader, "_row_to_record", original)
