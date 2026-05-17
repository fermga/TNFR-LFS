"""Channel tree (group → channel) backed by the registry in ``telemetry.channels``.

A ``QStandardItemModel`` of three levels:

* invisibleRootItem
  * Group (e.g. "Driver", "Engine", "Tyre") — non-checkable
    * Channel (column name, label, units) — checkable

We keep the model storage simple (no database) because the registry
fits in a few hundred items; the dock view renders it via ``QTreeView``.
The currently-checked column names are exposed via :meth:`checked_columns`
so the chart dock can subscribe directly to ``itemChanged``.
"""

from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel

from ...telemetry import ChannelInfo, channels_by_group
from ..i18n import current_language, tr


# Display order matches the Dash viewer for muscle memory continuity.
_GROUP_ORDER: tuple[str, ...] = (
    "Driver", "Engine", "Vehicle", "Chassis",
    "Suspension", "Tyre", "Derived", "Aids", "Other",
)


class ChannelTreeModel(QStandardItemModel):
    """Two-level tree (group → channel). Channels are checkable."""

    ColumnRole = int(Qt.ItemDataRole.UserRole) + 1

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setHorizontalHeaderLabels([tr("Channel"), tr("Units")])
        self._available_columns: set[str] = set()
        self._populate(channels_by_group())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_available_columns(self, columns: Iterable[str]) -> None:
        """Enable channels present in ``columns``, disable the rest.

        Disabled rows stay visible (greyed out) so the user understands
        which signals would appear with a richer capture; this matches
        MoTeC's behaviour and avoids a panel that mutates wildly across
        captures.
        """
        cols = set(columns)
        if cols == self._available_columns:
            return
        self._available_columns = cols
        for group_row in range(self.rowCount()):
            group_item = self.item(group_row, 0)
            if group_item is None:
                continue
            for ch_row in range(group_item.rowCount()):
                name_item = group_item.child(ch_row, 0)
                units_item = group_item.child(ch_row, 1)
                column = name_item.data(self.ColumnRole)
                enabled = column in cols
                flags = (
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                ) if enabled else Qt.ItemFlag.NoItemFlags
                name_item.setFlags(flags)
                if units_item is not None:
                    units_item.setFlags(flags)
                if not enabled and name_item.checkState() != Qt.CheckState.Unchecked:
                    name_item.setCheckState(Qt.CheckState.Unchecked)

    def set_checked_columns(self, columns: Iterable[str]) -> None:
        """Programmatically tick a set of channels (used for defaults)."""
        wanted = set(columns)
        for group_row in range(self.rowCount()):
            group_item = self.item(group_row, 0)
            if group_item is None:
                continue
            for ch_row in range(group_item.rowCount()):
                name_item = group_item.child(ch_row, 0)
                if name_item is None:
                    continue
                column = name_item.data(self.ColumnRole)
                if column in wanted and column in self._available_columns:
                    name_item.setCheckState(Qt.CheckState.Checked)

    def checked_columns(self) -> list[str]:
        """Return checked channel column names in tree order (stable)."""
        out: list[str] = []
        for group_row in range(self.rowCount()):
            group_item = self.item(group_row, 0)
            if group_item is None:
                continue
            for ch_row in range(group_item.rowCount()):
                name_item = group_item.child(ch_row, 0)
                if name_item is None:
                    continue
                if name_item.checkState() == Qt.CheckState.Checked:
                    out.append(name_item.data(self.ColumnRole))
        return out

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def _populate(self, groups: dict[str, list[ChannelInfo]]) -> None:
        lang = current_language()
        ordered = [g for g in _GROUP_ORDER if g in groups] + [
            g for g in sorted(groups) if g not in _GROUP_ORDER
        ]
        for group in ordered:
            group_item = QStandardItem(tr(group))
            group_item.setEditable(False)
            group_item.setSelectable(False)
            group_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            )
            blank = QStandardItem("")
            blank.setEditable(False)
            blank.setFlags(Qt.ItemFlag.ItemIsEnabled)
            for info in groups[group]:
                name = QStandardItem(tr(info.label))
                name.setEditable(False)
                name.setData(info.column, self.ColumnRole)
                name.setData(info.tooltip_html(translate=tr, language=lang),
                             Qt.ItemDataRole.ToolTipRole)
                name.setCheckable(True)
                name.setCheckState(Qt.CheckState.Unchecked)
                # Disabled by default until we know which columns the
                # selected lap actually has; ``set_available_columns``
                # flips the right ones on.
                name.setFlags(Qt.ItemFlag.NoItemFlags)
                units = QStandardItem(tr(info.units))
                units.setEditable(False)
                units.setData(info.tooltip_html(translate=tr, language=lang),
                              Qt.ItemDataRole.ToolTipRole)
                units.setFlags(Qt.ItemFlag.NoItemFlags)
                group_item.appendRow([name, units])
            self.appendRow([group_item, blank])


__all__ = ["ChannelTreeModel"]
