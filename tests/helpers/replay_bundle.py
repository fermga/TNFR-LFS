"""Utilities for interacting with replay bundle readers in tests."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from tnfr_lfs.core.epi import TelemetryRecord
from tnfr_lfs.ingestion.offline import ReplayCSVBundleReader

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only.
    import pytest


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
