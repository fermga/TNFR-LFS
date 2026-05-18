"""Captures dock: searchable, sortable, multi-selectable list of stints."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSortFilterProxyModel, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPushButton,
    QTableView,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from ..models import CapturesTableModel
from ..signals import SignalBus
from ..workspace_state import WorkspaceState


class CapturesDock(QWidget):
    """Workspace browser. Multi-select → ``signals.laps_selected``."""

    def __init__(
        self,
        workspace: WorkspaceState,
        signals: SignalBus,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._workspace = workspace
        self._signals = signals

        self._model = CapturesTableModel(self)
        self._proxy = QSortFilterProxyModel(self)
        self._proxy.setSourceModel(self._model)
        self._proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._proxy.setFilterKeyColumn(-1)  # any column

        self._view = QTableView(self)
        self._view.setModel(self._proxy)
        self._view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._view.setSortingEnabled(True)
        self._view.setAlternatingRowColors(True)
        self._view.setShowGrid(False)
        self._view.verticalHeader().setVisible(False)
        self._view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._view.horizontalHeader().setStretchLastSection(False)
        self._view.horizontalHeader().setDefaultAlignment(
            Qt.AlignLeft | Qt.AlignVCenter,
        )
        self._view.selectionModel().selectionChanged.connect(self._emit_selection)

        # Search box.
        self._search = QLineEdit(self)
        self._search.setPlaceholderText(tr("Filter (file, car, track)\u2026"))
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._proxy.setFilterFixedString)

        # Toolbar (refresh + counter).
        self._toolbar = QToolBar(self)
        refresh_action = QAction(tr("Refresh"), self)
        refresh_action.triggered.connect(self.refresh)
        refresh_action.setShortcut("F5")
        self._toolbar.addAction(refresh_action)
        self._toolbar.addSeparator()
        self._summary_button = QPushButton(tr("0 captures"), self)
        self._summary_button.setFlat(True)
        self._summary_button.setEnabled(False)
        self._toolbar.addWidget(self._summary_button)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(self._toolbar, 0)
        bar.addWidget(self._search, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addLayout(bar)
        layout.addWidget(self._view, 1)

        self.refresh()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        captures = self._workspace.refresh()
        self._model.set_captures(captures)
        self._summary_button.setText(
            tr("{n} captures").format(n=len(captures)),
        )
        self._signals.captures_refreshed.emit()
        # Resize file column nicely; leave the rest interactive.
        self._view.resizeColumnsToContents()

    def selected_paths(self) -> list[Path]:
        rows = self._view.selectionModel().selectedRows()
        out: list[Path] = []
        for proxy_idx in rows:
            src_row = self._proxy.mapToSource(proxy_idx).row()
            p = self._model.path_at(src_row)
            if p is not None:
                out.append(p)
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _emit_selection(self, *_args) -> None:
        self._signals.laps_selected.emit(self.selected_paths())


__all__ = ["CapturesDock"]
