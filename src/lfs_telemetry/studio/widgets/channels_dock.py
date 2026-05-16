"""Channel browser dock: tree of groups → channels with checkboxes."""

from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem
from PySide6.QtWidgets import (
    QHeaderView,
    QPushButton,
    QToolBar,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ..models import ChannelTreeModel
from ..signals import SignalBus


# Suggested defaults — same set the Dash viewer started with.
DEFAULT_CHANNELS: tuple[str, ...] = (
    "speed_ms", "throttle", "brake", "input_steer", "rpm", "gear_lfs",
)

# Pre-canned analytical channel groups for one-click engineering views.
FRICTION_CIRCLE_CHANNELS: tuple[str, ...] = (
    "accel_x", "accel_y",
    "friction_use_FL", "friction_use_FR",
    "friction_use_RL", "friction_use_RR",
)
LOAD_TRANSFER_CHANNELS: tuple[str, ...] = (
    "wheel_FL_vertical_load_n", "wheel_FR_vertical_load_n",
    "wheel_RL_vertical_load_n", "wheel_RR_vertical_load_n",
    "transfer_long_n_real", "transfer_lat_n_real",
    "load_front_frac", "load_left_frac",
    "pitch", "roll",
)


class ChannelsDock(QWidget):
    """Tree view of channels grouped by category, with check-state toggles."""

    def __init__(self, signals: SignalBus, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._signals = signals
        self._model = ChannelTreeModel(self)
        self._view = QTreeView(self)
        self._view.setModel(self._model)
        self._view.setAlternatingRowColors(True)
        self._view.setSelectionMode(QTreeView.NoSelection)
        self._view.setUniformRowHeights(True)
        self._view.setHeaderHidden(False)
        self._view.expandAll()
        header = self._view.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        # Suppress the indicator column (the tree branch) so the layout
        # is more compact — channel names and check marks are enough.

        self._toolbar = QToolBar(self)
        clear = QPushButton("Clear all", self)
        clear.clicked.connect(self.clear_all)
        defaults = QPushButton("Defaults", self)
        defaults.clicked.connect(self.apply_defaults)
        friction = QPushButton("Friction circle", self)
        friction.setToolTip(
            "Show the channels needed to read a friction-circle / "
            "g-g diagram: long+lat acceleration and per-wheel μ-use."
        )
        friction.clicked.connect(self.apply_friction_circle)
        load = QPushButton("Load transfer", self)
        load.setToolTip(
            "Show vertical-load per wheel plus longitudinal / lateral "
            "transfer and chassis pitch/roll."
        )
        load.clicked.connect(self.apply_load_transfer)
        expand = QPushButton("Expand", self)
        expand.clicked.connect(self._view.expandAll)
        collapse = QPushButton("Collapse", self)
        collapse.clicked.connect(self._view.collapseAll)
        for btn in (defaults, friction, load, clear, expand, collapse):
            self._toolbar.addWidget(btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._view, 1)

        self._model.itemChanged.connect(self._on_item_changed)
        self._suppress_signal = False
        self._defaults_applied = False
        signals.available_columns_changed.connect(self.set_available_columns)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_available_columns(self, columns: Iterable[str]) -> None:
        cols = list(columns)
        self._suppress_signal = True
        try:
            self._model.set_available_columns(cols)
        finally:
            self._suppress_signal = False
        # First time we know the schema: auto-tick the default set so
        # the user immediately sees traces. Subsequent capture changes
        # respect whatever the user has on screen.
        if cols and not self._defaults_applied:
            self._defaults_applied = True
            self.apply_defaults()

    def apply_defaults(self) -> None:
        self._apply_preset(DEFAULT_CHANNELS)

    def apply_friction_circle(self) -> None:
        """Tick the channels needed for friction-circle inspection."""
        self._apply_preset(FRICTION_CIRCLE_CHANNELS)

    def apply_load_transfer(self) -> None:
        """Tick the per-wheel load + chassis transfer channels."""
        self._apply_preset(LOAD_TRANSFER_CHANNELS)

    def _apply_preset(self, columns: Iterable[str]) -> None:
        self._suppress_signal = True
        try:
            # Clear first so unchecked channels actually go off.
            self.clear_all(emit=False)
            self._model.set_checked_columns(columns)
        finally:
            self._suppress_signal = False
        self._emit()

    def clear_all(self, *, emit: bool = True) -> None:
        for group_row in range(self._model.rowCount()):
            group_item = self._model.item(group_row, 0)
            if group_item is None:
                continue
            for ch_row in range(group_item.rowCount()):
                name_item = group_item.child(ch_row, 0)
                if name_item.checkState() != Qt.CheckState.Unchecked:
                    name_item.setCheckState(Qt.CheckState.Unchecked)
        if emit:
            self._emit()

    def checked_columns(self) -> list[str]:
        return self._model.checked_columns()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_item_changed(self, _item: QStandardItem) -> None:
        if self._suppress_signal:
            return
        self._emit()

    def _emit(self) -> None:
        self._signals.channels_changed.emit(self.checked_columns())


__all__ = [
    "ChannelsDock",
    "DEFAULT_CHANNELS",
    "FRICTION_CIRCLE_CHANNELS",
    "LOAD_TRANSFER_CHANNELS",
]
