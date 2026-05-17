"""Charts dock: x-axis selector + the multi-channel chart.

Subscribes to :class:`SignalBus` and forwards the (laps, channels,
axis_kind) tuple to :class:`MultiChannelChart`. Lap loads are async via
:class:`LapLoader` so the GUI thread stays responsive while CSV parse +
enrichment run in the background.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ...telemetry import LapTelemetry
from ..charts import MultiChannelChart
from ..i18n import tr
from ..models import LapLoader
from ..signals import SignalBus
from ..theme import MUTED_COLOR, PANEL_COLOR, TEXT_COLOR, trace_color


def _color_chip(color_hex: str, size: int = 10) -> QPixmap:
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setBrush(QColor(color_hex))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(0, 0, size - 1, size - 1)
    painter.end()
    return pix


class _LapLegend(QFrame):
    """Strip of color-chip + lap-name entries shown above the chart."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            f"background-color: {PANEL_COLOR}; border-radius: 3px;"
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Maximum)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(14)
        self._placeholder = QLabel(tr("No laps loaded"), self)
        self._placeholder.setStyleSheet(f"color: {MUTED_COLOR};")
        self._layout.addWidget(self._placeholder)
        self._layout.addStretch(1)

    def set_laps(self, laps: list[LapTelemetry]) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if not laps:
            self._placeholder = QLabel(tr("No laps loaded"), self)
            self._placeholder.setStyleSheet(f"color: {MUTED_COLOR};")
            self._layout.addWidget(self._placeholder)
            self._layout.addStretch(1)
            return
        for idx, lap in enumerate(laps):
            color = trace_color(idx)
            chip = QLabel(self)
            chip.setPixmap(_color_chip(color, 10))
            name = lap.source_path.name if lap.source_path else f"lap{idx}"
            suffix = tr(" (ref)") if idx == 0 and len(laps) > 1 else ""
            label = QLabel(f"{name}{suffix}", self)
            label.setStyleSheet(f"color: {TEXT_COLOR};")
            entry = QWidget(self)
            row = QHBoxLayout(entry)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            row.addWidget(chip)
            row.addWidget(label)
            self._layout.addWidget(entry)
        self._layout.addStretch(1)


class ChartsDock(QWidget):
    """The right-hand chart area: axis controls + stacked plots."""

    def __init__(
        self,
        loader: LapLoader,
        signals: SignalBus,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._loader = loader
        self._signals = signals

        self._axis_group = QButtonGroup(self)
        self._axis_distance = QRadioButton(tr("Distance"), self)
        self._axis_time = QRadioButton(tr("Time"), self)
        self._axis_distance.setChecked(True)
        self._axis_group.addButton(self._axis_distance, 0)
        self._axis_group.addButton(self._axis_time, 1)
        self._axis_group.buttonToggled.connect(self._on_axis_toggled)

        toolbar = QToolBar(self)
        toolbar.addWidget(QLabel(tr("X-axis: ")))
        toolbar.addWidget(self._axis_distance)
        toolbar.addWidget(self._axis_time)
        toolbar.addSeparator()

        export_png = QPushButton(tr("Export PNG…"), self)
        export_png.clicked.connect(self._export_all_png)
        toolbar.addWidget(export_png)

        export_csv = QPushButton(tr("Export CSV…"), self)
        export_csv.clicked.connect(self._export_all_csv)
        toolbar.addWidget(export_csv)

        toolbar.addSeparator()
        self._caption = QLabel(tr("No laps selected"), self)
        self._caption.setStyleSheet("color: #8a939e;")
        toolbar.addWidget(self._caption)

        self._legend = _LapLegend(self)
        self._chart = MultiChannelChart(self)

        self._scroll = QScrollArea(self)
        self._scroll.setWidget(self._chart)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(toolbar)
        layout.addWidget(self._legend)
        layout.addWidget(self._scroll, 1)

        self._requested_paths: List[Path] = []
        self._loaded_laps: Dict[Path, LapTelemetry] = {}
        self._channels: List[str] = []

        signals.laps_selected.connect(self._on_laps_selected)
        signals.channels_changed.connect(self._on_channels_changed)
        loader.lap_loaded.connect(self._on_lap_loaded)
        loader.lap_failed.connect(self._on_lap_failed)
        self._chart.cursor_moved.connect(signals.cursor_moved)
        self._chart.cursor_left.connect(signals.cursor_left)

    def _on_axis_toggled(self, *_args) -> None:
        kind = "distance" if self._axis_distance.isChecked() else "time"
        self._chart.set_axis_kind(kind)
        self._signals.x_axis_changed.emit(kind)

    def _on_laps_selected(self, paths: List[Path]) -> None:
        self._requested_paths = [Path(p) for p in paths]
        wanted = set(self._requested_paths)
        self._loaded_laps = {
            p: lap for p, lap in self._loaded_laps.items() if p in wanted
        }
        if not self._requested_paths:
            self._caption.setText(tr("No laps selected"))
            self._chart.set_laps([])
            return
        missing = [
            p for p in self._requested_paths
            if p not in self._loaded_laps
        ]
        if missing:
            self._caption.setText(
                tr("Loading {n} of {total} lap(s)…").format(
                    n=len(missing), total=len(self._requested_paths),
                )
            )
            for path in missing:
                self._loader.request(path)
        self._refresh_chart()

    def _on_channels_changed(self, channels: List[str]) -> None:
        self._channels = list(channels)
        self._chart.set_channels(self._channels)

    def _on_lap_loaded(self, path: Path, lap: LapTelemetry) -> None:
        path = Path(path)
        if path not in self._requested_paths:
            return
        self._loaded_laps[path] = lap
        self._refresh_chart()

    def _on_lap_failed(self, path: Path, message: str) -> None:
        self._signals.status_message.emit(
            tr("Failed to load {name}: {error}").format(
                name=Path(path).name, error=message,
            ),
            8000,
        )

    def _refresh_chart(self) -> None:
        laps = [
            self._loaded_laps[p]
            for p in self._requested_paths
            if p in self._loaded_laps
        ]
        if laps:
            cols = list(laps[0].enriched.columns)
            self._signals.available_columns_changed.emit(cols)
        self._chart.set_laps(laps)
        self._legend.set_laps(laps)
        if not laps:
            return
        if len(laps) == len(self._requested_paths):
            self._caption.setText(
                tr("{n} lap(s)").format(n=len(laps)),
            )
        else:
            self._caption.setText(
                tr("{n} of {total} lap(s) loaded").format(
                    n=len(laps), total=len(self._requested_paths),
                )
            )

    def _export_all_png(self) -> None:
        rows = self._chart.exportable_rows()
        if not rows:
            QMessageBox.information(
                self,
                tr("Export charts"),
                tr("No telemetry charts available to export yet."),
            )
            return
        folder = QFileDialog.getExistingDirectory(
            self,
            tr("Choose folder for PNG export"),
            str(self._default_export_dir()),
        )
        if not folder:
            return
        out_dir = Path(folder)
        exported = 0
        for channel, widget in rows:
            file_name = self._safe_file_name(f"telemetry_{channel}.png")
            out_path = out_dir / file_name
            if widget.grab().save(str(out_path), "PNG"):
                exported += 1
        self._signals.status_message.emit(
            tr("Exported {n} PNG chart(s) to {path}.").format(
                n=exported,
                path=out_dir,
            ),
            5000,
        )

    def _export_all_csv(self) -> None:
        laps = [
            self._loaded_laps[p]
            for p in self._requested_paths
            if p in self._loaded_laps
        ]
        if not laps or not self._channels:
            QMessageBox.information(
                self,
                tr("Export charts"),
                tr("Load laps and select channels before exporting CSV."),
            )
            return
        folder = QFileDialog.getExistingDirectory(
            self,
            tr("Choose folder for CSV export"),
            str(self._default_export_dir()),
        )
        if not folder:
            return
        out_dir = Path(folder)
        x_col = "distance_m" if self._axis_distance.isChecked() else "t_s"
        exported = 0
        for channel in self._channels:
            series_map: dict[str, pd.Series] = {}
            for idx, lap in enumerate(laps):
                if x_col not in lap.enriched.columns:
                    continue
                if channel not in lap.enriched.columns:
                    continue
                lap_name = self._lap_name(lap.source_path, idx)
                series_map[f"{lap_name}__{x_col}"] = pd.Series(
                    lap.enriched[x_col].to_numpy()
                )
                series_map[f"{lap_name}__{channel}"] = pd.Series(
                    lap.enriched[channel].to_numpy()
                )
            if not series_map:
                continue
            file_name = self._safe_file_name(f"telemetry_{channel}.csv")
            out_path = out_dir / file_name
            pd.DataFrame(series_map).to_csv(out_path, index=False)
            exported += 1
        self._signals.status_message.emit(
            tr("Exported {n} CSV chart(s) to {path}.").format(
                n=exported,
                path=out_dir,
            ),
            5000,
        )

    def _default_export_dir(self) -> Path:
        if self._requested_paths:
            return self._requested_paths[0].parent
        return Path.cwd()

    def _lap_name(self, path: Path | None, idx: int) -> str:
        if path is None:
            return f"lap{idx + 1}"
        return path.stem

    def _safe_file_name(self, name: str) -> str:
        forbidden = '<>:"/\\|?*'
        clean = "".join("_" if ch in forbidden else ch for ch in name)
        return clean.strip().replace(" ", "_")


__all__ = ["ChartsDock"]
