"""Charts dock: x-axis selector + the multi-channel chart.

Subscribes to :class:`SignalBus` and forwards the (laps, channels,
axis_kind) tuple to :class:`MultiChannelChart`. Lap loads are async via
:class:`LapLoader` so the GUI thread stays responsive while CSV parse +
enrichment run in the background.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...lfs_paths import QSETTINGS_APP as APP
from ...lfs_paths import QSETTINGS_ORG as ORG
from ...telemetry import LapTelemetry
from ..charts import MultiChannelChart
from ..i18n import tr
from ..models import LapLoader
from ..signals import SignalBus
from ..theme import MUTED_COLOR, PANEL_COLOR, TEXT_COLOR, trace_color


# QSettings keys for the overlay/normalize toggles.
_SETTINGS_OVERLAY = "chartsdock/overlay"
_SETTINGS_NORMALIZE = "chartsdock/normalize"


# Canonical overlay presets. Each entry: (label, channels, overlay,
# normalize). The channels list is applied to the channels dock via the
# ``channels_requested`` signal; missing columns are silently skipped
# by the dock, so presets that reference advanced channels degrade
# gracefully on stripped-down captures.
_CANONICAL_PAIRS: tuple[tuple[str, tuple[str, ...], bool, bool], ...] = (
    ("Throttle + Brake",
     ("throttle", "brake"), True, False),
    ("Throttle + Brake + Clutch",
     ("throttle", "brake", "clutch"), True, False),
    ("Speed + Throttle + Brake (norm.)",
     ("speed_ms", "throttle", "brake"), True, True),
    ("Steer + Lat. accel (norm.)",
     ("input_steer", "accel_y"), True, True),
    ("Slip ratio × 4 wheels",
     ("wheel_FL_slip_ratio", "wheel_FR_slip_ratio",
      "wheel_RL_slip_ratio", "wheel_RR_slip_ratio"), True, False),
    ("Vert. load × 4 wheels",
     ("wheel_FL_vertical_load_n", "wheel_FR_vertical_load_n",
      "wheel_RL_vertical_load_n", "wheel_RR_vertical_load_n"), True, False),
    ("Tyre temp × 4 wheels",
     ("wheel_FL_air_temp_c", "wheel_FR_air_temp_c",
      "wheel_RL_air_temp_c", "wheel_RR_air_temp_c"), True, False),
    ("Susp. travel × 4 wheels",
     ("wheel_FL_susp_deflect_m", "wheel_FR_susp_deflect_m",
      "wheel_RL_susp_deflect_m", "wheel_RR_susp_deflect_m"), True, False),
    ("Friction use × 4 wheels",
     ("friction_use_FL", "friction_use_FR",
      "friction_use_RL", "friction_use_RR"), True, False),
    ("Brake bias + Brake pedal (norm.)",
     ("brake_bias_front_real", "brake"), True, True),
)


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

        self._chk_overlay = QCheckBox(tr("Overlay"), self)
        self._chk_overlay.setToolTip(
            tr("Group channels with the same units into one chart row")
        )
        toolbar.addWidget(self._chk_overlay)

        self._chk_normalize = QCheckBox(tr("Normalize"), self)
        self._chk_normalize.setToolTip(
            tr("Rescale every trace to its own 0–1 range so channels "
               "with very different magnitudes stay comparable")
        )
        toolbar.addWidget(self._chk_normalize)

        # Canonical overlay presets (drop-down).
        self._pairs_btn = QToolButton(self)
        self._pairs_btn.setText(tr("Canonical pairs…"))
        self._pairs_btn.setToolTip(
            tr("Apply a recommended channel overlay (e.g. Throttle + "
               "Brake, Slip ratio × 4 wheels)")
        )
        self._pairs_btn.setPopupMode(QToolButton.InstantPopup)
        self._pairs_menu = QMenu(self._pairs_btn)
        for label, channels, overlay, normalize in _CANONICAL_PAIRS:
            action = QAction(tr(label), self._pairs_menu)
            action.setData((tuple(channels), overlay, normalize))
            action.triggered.connect(self._on_canonical_pair_triggered)
            self._pairs_menu.addAction(action)
        self._pairs_btn.setMenu(self._pairs_menu)
        toolbar.addWidget(self._pairs_btn)
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

        self._requested_paths: list[Path] = []
        self._loaded_laps: dict[Path, LapTelemetry] = {}
        self._channels: list[str] = []

        signals.laps_selected.connect(self._on_laps_selected)
        signals.channels_changed.connect(self._on_channels_changed)
        loader.lap_loaded.connect(self._on_lap_loaded)
        loader.lap_failed.connect(self._on_lap_failed)
        self._chart.cursor_moved.connect(signals.cursor_moved)
        self._chart.cursor_left.connect(signals.cursor_left)

        # Restore persisted overlay/normalize toggle state before wiring
        # the signals, so the initial load doesn't write the defaults
        # back to disk and we avoid an unnecessary rebuild.
        s = self._settings()
        ov = self._coerce_bool(s.value(_SETTINGS_OVERLAY, False))
        nm = self._coerce_bool(s.value(_SETTINGS_NORMALIZE, False))
        self._chk_overlay.setChecked(ov)
        self._chk_normalize.setChecked(nm)
        self._chart.set_overlay_mode(ov)
        self._chart.set_normalize(nm)
        # Wire overlay/normalize after restore so persistence kicks in
        # only on subsequent user toggles.
        self._chk_overlay.toggled.connect(self._on_overlay_toggled)
        self._chk_normalize.toggled.connect(self._on_normalize_toggled)

    def _on_axis_toggled(self, *_args) -> None:
        kind = "distance" if self._axis_distance.isChecked() else "time"
        self._chart.set_axis_kind(kind)
        self._signals.x_axis_changed.emit(kind)

    # ------------------------------------------------------------------
    # Overlay / normalize / canonical pairs
    # ------------------------------------------------------------------

    def _settings(self) -> QSettings:
        return QSettings(ORG, APP)

    @staticmethod
    def _coerce_bool(value) -> bool:
        # QSettings returns strings on some platforms ("true"/"false").
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        try:
            return bool(int(value))
        except (TypeError, ValueError):
            return bool(value)

    def _on_overlay_toggled(self, checked: bool) -> None:
        self._chart.set_overlay_mode(checked)
        self._settings().setValue(_SETTINGS_OVERLAY, bool(checked))

    def _on_normalize_toggled(self, checked: bool) -> None:
        self._chart.set_normalize(checked)
        self._settings().setValue(_SETTINGS_NORMALIZE, bool(checked))

    def _on_canonical_pair_triggered(self) -> None:
        action = self.sender()
        if not isinstance(action, QAction):
            return
        payload = action.data()
        if not isinstance(payload, tuple) or len(payload) != 3:
            return
        channels, overlay, normalize = payload
        # Toggle overlay/normalize first so when the channels arrive the
        # chart already lays them out grouped.
        if bool(overlay) != self._chk_overlay.isChecked():
            self._chk_overlay.setChecked(bool(overlay))
        else:
            # Same value: ensure the chart honours it (defensive).
            self._chart.set_overlay_mode(bool(overlay))
        if bool(normalize) != self._chk_normalize.isChecked():
            self._chk_normalize.setChecked(bool(normalize))
        else:
            self._chart.set_normalize(bool(normalize))
        # Ask the channels dock to tick the preset's channels.
        self._signals.channels_requested.emit(list(channels))
        self._signals.status_message.emit(
            tr("Applied canonical pair: {name}").format(name=action.text()),
            3500,
        )

    def _on_laps_selected(self, paths: list[Path]) -> None:
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

    def _on_channels_changed(self, channels: list[str]) -> None:
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
