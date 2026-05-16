"""``QAbstractTableModel`` over :class:`WorkspaceState` for the captures dock.

Columns mirror the Dash captures table so users keep the same mental
model. Sorting is delegated to ``QSortFilterProxyModel`` in the view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from ...telemetry import CaptureInfo


@dataclass(frozen=True)
class _Col:
    key: str
    header: str
    align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter


_COLUMNS: tuple[_Col, ...] = (
    _Col("file", "File"),
    _Col("car", "Car"),
    _Col("track", "Track"),
    _Col("samples", "Samples",
         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
    _Col("lap_time_s", "Lap (s)",
         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
    _Col("distance_m", "Dist (m)",
         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
    _Col("size_kb", "Size (KB)",
         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
)


class CapturesTableModel(QAbstractTableModel):
    """Read-only view of a list of :class:`CaptureInfo`."""

    PathRole = int(Qt.ItemDataRole.UserRole) + 1

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: list[CaptureInfo] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_captures(self, captures: Sequence[CaptureInfo]) -> None:
        """Replace the entire row set in one ``modelReset`` cycle."""
        self.beginResetModel()
        self._rows = list(captures)
        self.endResetModel()

    def capture_at(self, row: int) -> CaptureInfo | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def path_at(self, row: int) -> Path | None:
        cap = self.capture_at(row)
        return Path(cap.path) if cap else None

    # ------------------------------------------------------------------
    # QAbstractTableModel
    # ------------------------------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(_COLUMNS)

    def headerData(  # type: ignore[override]
        self, section: int, orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return _COLUMNS[section].header
        return section + 1

    def data(  # type: ignore[override]
        self, index: QModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None
        cap = self._rows[index.row()]
        col = _COLUMNS[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return self._render(cap, col.key)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(col.align)
        if role == self.PathRole:
            return str(cap.path)
        if role == Qt.ItemDataRole.ToolTipRole:
            return str(cap.path)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render(cap: CaptureInfo, key: str) -> str:
        if key == "file":
            return Path(cap.path).name
        if key == "car":
            return cap.car or ""
        if key == "track":
            return cap.track or ""
        if key == "samples":
            return f"{cap.samples:,}" if cap.samples else ""
        if key == "lap_time_s":
            return "" if cap.lap_time_s is None else f"{cap.lap_time_s:.3f}"
        if key == "distance_m":
            return "" if cap.distance_m is None else f"{cap.distance_m:.1f}"
        if key == "size_kb":
            return f"{cap.file_size_bytes / 1024.0:.1f}"
        return ""


__all__ = ["CapturesTableModel"]
