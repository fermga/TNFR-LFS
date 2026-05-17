"""Channel browser dock: tree of groups → channels with checkboxes."""

from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QStandardItem
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QToolBar,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ..models import ChannelTreeModel
from ..signals import SignalBus
from ..i18n import tr
from ...lfs_paths import QSETTINGS_APP as APP, QSETTINGS_ORG as ORG


# Hard cap on simultaneously plotted channels: above this the chart
# stack and the track-map overlay degrade. Enforced at tick time.
MAX_SELECTED_CHANNELS = 8

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
)

_PRESETS_KEY = "telemetry/presets"


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
        clear = QPushButton(tr("Clear all"), self)
        clear.clicked.connect(self.clear_all)
        defaults = QPushButton(tr("Defaults"), self)
        defaults.clicked.connect(self.apply_defaults)
        friction = QPushButton(tr("Friction circle"), self)
        friction.setToolTip(
            tr(
                "Show the channels needed to read a friction-circle / "
                "g-g diagram: long+lat acceleration and per-wheel "
                "\u03bc-use.",
            ),
        )
        friction.clicked.connect(self.apply_friction_circle)
        load = QPushButton(tr("Load transfer"), self)
        load.setToolTip(
            tr(
                "Show vertical-load per wheel plus longitudinal / "
                "lateral transfer.",
            ),
        )
        load.clicked.connect(self.apply_load_transfer)
        expand = QPushButton(tr("Expand"), self)
        expand.clicked.connect(self._view.expandAll)
        collapse = QPushButton(tr("Collapse"), self)
        collapse.clicked.connect(self._view.collapseAll)
        for btn in (defaults, friction, load, clear, expand, collapse):
            self._toolbar.addWidget(btn)

        # ----- Named presets -------------------------------------------
        self._preset_combo = QComboBox(self)
        self._preset_combo.setMinimumWidth(140)
        self._preset_combo.setToolTip(
            tr("Saved channel selections. Pick one to apply it."),
        )
        self._preset_save = QPushButton(tr("Save\u2026"), self)
        self._preset_save.setToolTip(
            tr("Save the current channel selection as a named preset."),
        )
        self._preset_save.clicked.connect(self._action_save_preset)
        self._preset_delete = QPushButton(tr("Delete"), self)
        self._preset_delete.setToolTip(tr("Delete the selected preset."))
        self._preset_delete.clicked.connect(self._action_delete_preset)
        self._preset_combo.activated.connect(self._on_preset_activated)

        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.addWidget(QLabel(tr("Preset:"), self))
        preset_row.addWidget(self._preset_combo, 1)
        preset_row.addWidget(self._preset_save)
        preset_row.addWidget(self._preset_delete)

        # ----- Selection-count status ----------------------------------
        self._status = QLabel(self)
        self._status.setStyleSheet("color: #888; padding: 2px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self._toolbar)
        layout.addLayout(preset_row)
        layout.addWidget(self._view, 1)
        layout.addWidget(self._status)

        self._model.itemChanged.connect(self._on_item_changed)
        self._suppress_signal = False
        self._defaults_applied = False
        signals.available_columns_changed.connect(self.set_available_columns)
        self._reload_presets()
        self._update_status()

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
            # Enforce the same cap when applying presets so they don't
            # silently exceed the limit either.
            cols = list(columns)[:MAX_SELECTED_CHANNELS]
            self._model.set_checked_columns(cols)
        finally:
            self._suppress_signal = False
        self._update_status()
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

    def _on_item_changed(self, item: QStandardItem) -> None:
        if self._suppress_signal:
            return
        # Enforce the hard cap. If the user just ticked a 9th channel,
        # revert it and warn via the status label.
        if (
            item.isCheckable()
            and item.checkState() == Qt.CheckState.Checked
            and len(self._model.checked_columns()) > MAX_SELECTED_CHANNELS
        ):
            self._suppress_signal = True
            try:
                item.setCheckState(Qt.CheckState.Unchecked)
            finally:
                self._suppress_signal = False
            self._status.setText(
                tr(
                    "Maximum {n} channels at once \u2014 untick one "
                    "before adding another.",
                ).format(n=MAX_SELECTED_CHANNELS),
            )
            self._status.setStyleSheet(
                "color: #c0392b; padding: 2px; font-weight: bold;"
            )
            return
        self._update_status()
        self._emit()

    def _update_status(self) -> None:
        n = len(self._model.checked_columns())
        self._status.setText(
            tr("{n} / {max} channels selected.").format(
                n=n, max=MAX_SELECTED_CHANNELS,
            ),
        )
        self._status.setStyleSheet("color: #888; padding: 2px;")

    # ------------------------------------------------------------------
    # Presets (QSettings-backed)
    # ------------------------------------------------------------------

    def _settings(self) -> QSettings:
        return QSettings(ORG, APP)

    def _load_presets(self) -> dict[str, list[str]]:
        raw = self._settings().value(_PRESETS_KEY, {}) or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, list[str]] = {}
        for name, cols in raw.items():
            if isinstance(cols, (list, tuple)):
                out[str(name)] = [str(c) for c in cols]
        return out

    def _save_presets(self, presets: dict[str, list[str]]) -> None:
        self._settings().setValue(_PRESETS_KEY, presets)

    def _reload_presets(self) -> None:
        self._preset_combo.blockSignals(True)
        try:
            self._preset_combo.clear()
            self._preset_combo.addItem(tr("(no preset)"), "")
            for name in sorted(self._load_presets()):
                self._preset_combo.addItem(name, name)
        finally:
            self._preset_combo.blockSignals(False)
        self._preset_delete.setEnabled(
            self._preset_combo.count() > 1
        )

    def _on_preset_activated(self, _idx: int) -> None:
        name = self._preset_combo.currentData()
        if not name:
            return
        presets = self._load_presets()
        cols = presets.get(name)
        if cols is None:
            return
        self._apply_preset(cols)

    def _action_save_preset(self) -> None:
        cols = self.checked_columns()
        if not cols:
            QMessageBox.information(
                self, tr("Save preset"),
                tr("Tick at least one channel before saving a preset."),
            )
            return
        current = self._preset_combo.currentData() or ""
        name, ok = QInputDialog.getText(
            self, tr("Save channel preset"),
            tr(
                "Preset name (e.g. Qualifying, Race start, Brake "
                "balance):",
            ),
            text=current if isinstance(current, str) else "",
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        presets = self._load_presets()
        if name in presets:
            resp = QMessageBox.question(
                self, tr("Overwrite preset?"),
                tr(
                    "A preset named \u2018{name}\u2019 already exists. "
                    "Overwrite it?",
                ).format(name=name),
            )
            if resp != QMessageBox.StandardButton.Yes:
                return
        presets[name] = list(cols)
        self._save_presets(presets)
        self._reload_presets()
        idx = self._preset_combo.findData(name)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)

    def _action_delete_preset(self) -> None:
        name = self._preset_combo.currentData()
        if not name:
            return
        resp = QMessageBox.question(
            self, tr("Delete preset?"),
            tr("Delete preset \u2018{name}\u2019?").format(name=name),
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        presets = self._load_presets()
        presets.pop(name, None)
        self._save_presets(presets)
        self._reload_presets()

    def _emit(self) -> None:
        self._signals.channels_changed.emit(self.checked_columns())


__all__ = [
    "ChannelsDock",
    "DEFAULT_CHANNELS",
    "FRICTION_CIRCLE_CHANNELS",
    "LOAD_TRANSFER_CHANNELS",
    "MAX_SELECTED_CHANNELS",
]
