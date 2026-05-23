"""MoTeC-inspired Telemetry tab built on the workbook data model.

Replaces the legacy single-chart :class:`ChartsDock` central widget.
A *workbook* contains one or more *worksheets*; each worksheet is a
vertical stack of *component cards*. Each card renders one slice of
telemetry (graph, bar, …) and owns its own channel list, so the user
can lay out multiple synchronized views per worksheet à la MoTeC i2.

Phase 2 scope (this file):

* `graph` components are rendered with the existing
  :class:`MultiChannelChart` so we inherit overlay/normalize, cursor
  sync, delta-vs-reference and PNG export.
* `bar` components render lap-mean values per channel as grouped bars
  (one group per lap, one bar per channel) — useful for per-wheel
  comparisons like tyre temps or vertical-load means.
* The legacy :class:`ChannelsDock` keeps working as a global channel
  picker: its emissions are routed to the *active* card (the one whose
  header was last clicked); becoming active pushes the card's current
  channels back to the dock so the two stay in sync.
* Cards can be reordered within a worksheet via up/down arrows on the
  header. Splitter sizes persist per (workbook, worksheet) so the
  user's manual height tweaks survive across launches.
* The active workbook + worksheet index persist across launches.

A proper channel browser, per-component editor dialog and the
remaining component types (histogram, xy, gauge, trackmap, report)
land in subsequent commits.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...lfs_paths import QSETTINGS_APP as APP
from ...lfs_paths import QSETTINGS_ORG as ORG
from ...telemetry import LapTelemetry, channel_info
from ..charts import MultiChannelChart
from ..i18n import tr
from ..models import LapLoader
from ..signals import SignalBus
from ..theme import MUTED_COLOR, PANEL_COLOR, TEXT_COLOR, trace_color
from ..workbooks import (
    Component,
    Workbook,
    Worksheet,
    builtin_template,
    builtin_template_names,
    default_workbook,
    list_user_workbooks,
    load_workbook,
    save_user_workbook,
)

LOG = logging.getLogger(__name__)


# QSettings keys --------------------------------------------------------
_SETTINGS_WORKBOOK = "workbooktab/workbook"        # last workbook file name
_SETTINGS_WS_INDEX = "workbooktab/worksheet_index"  # last worksheet idx
_SETTINGS_AXIS = "workbooktab/axis_kind"            # "distance" | "time"
# Per-(workbook, worksheet) splitter sizes live under
# ``workbooktab/sizes/<workbook>/<sheet_idx>`` as a comma-joined list
# of integers so QSettings doesn't have to round-trip lists.
_SETTINGS_SIZES_PREFIX = "workbooktab/sizes"


# ---------------------------------------------------------------------------
# Bar panel (lap-mean per channel, grouped by lap)
# ---------------------------------------------------------------------------


class _BarPanel(QWidget):
    """Grouped bars showing each channel's lap-mean.

    One group per channel along x, one bar per lap inside each group.
    Useful for per-wheel comparisons (tyre temps, vertical load means)
    where stacked line traces are hard to read at a glance.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._laps: list[LapTelemetry] = []
        self._channels: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._plot = pg.PlotWidget(self)
        self._plot.setBackground(PANEL_COLOR)
        self._plot.showGrid(x=False, y=True, alpha=0.25)
        self._plot.getPlotItem().getAxis("left").setTextPen(TEXT_COLOR)
        self._plot.getPlotItem().getAxis("bottom").setTextPen(TEXT_COLOR)
        layout.addWidget(self._plot)

    def set_laps(self, laps: list[LapTelemetry]) -> None:
        self._laps = list(laps)
        self._rebuild()

    def set_channels(self, channels: list[str]) -> None:
        self._channels = list(channels)
        self._rebuild()

    # ------------------------------------------------------------------

    def _short_label(self, ch: str) -> str:
        # ``wheel_FL_air_temp_c`` -> ``FL`` when 4 channels share the
        # same suffix; otherwise use the channel's own short label.
        info = channel_info(ch)
        return info.label or ch

    def _rebuild(self) -> None:
        plot = self._plot
        plot.clear()
        if not self._laps or not self._channels:
            return

        n_chans = len(self._channels)
        n_laps = len(self._laps)
        # Bar width within each group: shrink with more laps.
        group_width = 0.8
        bar_w = group_width / max(n_laps, 1)

        any_finite = False
        units_seen: set[str] = set()
        for lap_idx, lap in enumerate(self._laps):
            heights: list[float] = []
            xs: list[float] = []
            df = lap.enriched
            offset = (lap_idx - (n_laps - 1) / 2.0) * bar_w
            for ch_idx, ch in enumerate(self._channels):
                if ch not in df.columns:
                    continue
                arr = df[ch].to_numpy(dtype=float)
                if not np.isfinite(arr).any():
                    continue
                mean = float(np.nanmean(arr))
                if math.isnan(mean):
                    continue
                heights.append(mean)
                xs.append(ch_idx + offset)
                any_finite = True
                units_seen.add(channel_info(ch).units or "")
            if not xs:
                continue
            color = trace_color(lap_idx)
            name = (
                lap.source_path.name if lap.source_path
                else f"lap{lap_idx}"
            )
            bars = pg.BarGraphItem(
                x=np.asarray(xs), height=np.asarray(heights),
                width=bar_w * 0.9,
                brush=pg.mkBrush(color),
                pen=pg.mkPen("#202830", width=1),
                name=name,
            )
            plot.addItem(bars)

        # X-axis ticks: one per channel, labelled with a short name.
        ticks = [
            (i, self._short_label(self._channels[i]))
            for i in range(n_chans)
        ]
        plot.getPlotItem().getAxis("bottom").setTicks([ticks])

        # Y-axis label: if every channel shares a unit, use it; else
        # fall back to a generic label.
        if len(units_seen) == 1 and next(iter(units_seen)):
            unit = next(iter(units_seen))
            plot.setLabel("left", f"mean [{unit}]", color=TEXT_COLOR)
        else:
            plot.setLabel("left", tr("lap mean"), color=TEXT_COLOR)

        if any_finite:
            plot.getViewBox().autoRange()


# ---------------------------------------------------------------------------
# Component card
# ---------------------------------------------------------------------------


class _ComponentCard(QFrame):
    """One renderable telemetry component (graph, bar, …).

    A card is the unit of MoTeC-style composition: it has its own
    title bar (drag handle placeholder, title, options, close button)
    and a body widget chosen by ``component.type``.
    """

    # Emitted when the user clicks anywhere in the header so the parent
    # worksheet can flag this card as active and route the global
    # channel picker to it.
    activated = Signal(object)  # self
    # User requested to remove this component from the worksheet.
    remove_requested = Signal(object)  # self
    # User toggled overlay/normalize via the per-card header.
    options_changed = Signal(object)  # self
    # User pressed the up/down arrow on the header. Payload: (self, delta)
    # where delta is -1 (move up) or +1 (move down).
    move_requested = Signal(object, int)

    def __init__(
        self,
        component: Component,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.component = component
        self._active = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("componentCard")
        self._apply_active_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._header = self._build_header()
        layout.addWidget(self._header)

        self._body = self._build_body()
        layout.addWidget(self._body, 1)

    # ---- header -------------------------------------------------------

    def _build_header(self) -> QWidget:
        hdr = QFrame(self)
        hdr.setObjectName("componentHeader")
        hdr.setStyleSheet(
            f"#componentHeader {{ background-color: {PANEL_COLOR}; "
            f"border-radius: 2px; }}"
        )
        h = QHBoxLayout(hdr)
        h.setContentsMargins(6, 2, 4, 2)
        h.setSpacing(6)

        self._title_lbl = QLabel(self.component.title, hdr)
        self._title_lbl.setStyleSheet(
            f"color: {TEXT_COLOR}; font-weight: 600;"
        )
        h.addWidget(self._title_lbl)

        self._channels_lbl = QLabel("", hdr)
        self._channels_lbl.setStyleSheet(f"color: {MUTED_COLOR};")
        self._refresh_channels_label()
        h.addWidget(self._channels_lbl, 1)

        # Per-component overlay / normalize options live on the header
        # (graph components only — other types ignore them).
        if self.component.type == "graph":
            self._chk_overlay = QCheckBox(tr("Overlay"), hdr)
            self._chk_overlay.setChecked(
                bool(self.component.options.get("overlay", True))
            )
            self._chk_overlay.toggled.connect(self._on_overlay_toggled)
            h.addWidget(self._chk_overlay)

            self._chk_normalize = QCheckBox(tr("Norm."), hdr)
            self._chk_normalize.setChecked(
                bool(self.component.options.get("normalize", False))
            )
            self._chk_normalize.toggled.connect(self._on_normalize_toggled)
            h.addWidget(self._chk_normalize)
        else:
            self._chk_overlay = None
            self._chk_normalize = None

        close_btn = QToolButton(hdr)
        close_btn.setText("×")
        close_btn.setToolTip(tr("Remove component"))
        close_btn.clicked.connect(lambda: self.remove_requested.emit(self))

        up_btn = QToolButton(hdr)
        up_btn.setText("▲")
        up_btn.setToolTip(tr("Move component up"))
        up_btn.clicked.connect(
            lambda: self.move_requested.emit(self, -1)
        )
        h.addWidget(up_btn)

        down_btn = QToolButton(hdr)
        down_btn.setText("▼")
        down_btn.setToolTip(tr("Move component down"))
        down_btn.clicked.connect(
            lambda: self.move_requested.emit(self, +1)
        )
        h.addWidget(down_btn)

        h.addWidget(close_btn)

        # Clicking anywhere in the header (except controls) activates.
        hdr.mousePressEvent = self._header_clicked  # type: ignore[assignment]
        return hdr

    def _header_clicked(self, _event) -> None:
        self.activated.emit(self)

    # ---- body ---------------------------------------------------------

    def _build_body(self) -> QWidget:
        if self.component.type == "graph":
            chart = MultiChannelChart(self)
            chart.set_channels(list(self.component.channels))
            chart.set_overlay_mode(
                bool(self.component.options.get("overlay", True))
            )
            chart.set_normalize(
                bool(self.component.options.get("normalize", False))
            )
            self._chart: MultiChannelChart | None = chart
            self._bar: _BarPanel | None = None
            return chart
        if self.component.type == "bar":
            bar = _BarPanel(self)
            bar.set_channels(list(self.component.channels))
            self._chart = None
            self._bar = bar
            return bar
        # Placeholder for not-yet-implemented component types
        # (gauge, histogram, xy, trackmap, report).
        ph = QLabel(
            tr("{kind!r} components arrive in a later commit.").format(
                kind=self.component.type
            ),
            self,
        )
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setStyleSheet(
            f"color: {MUTED_COLOR}; padding: 18px; "
            f"background-color: {PANEL_COLOR}; border-radius: 3px;"
        )
        self._chart = None
        self._bar = None
        return ph

    # ---- public api ---------------------------------------------------

    def set_active(self, active: bool) -> None:
        if active == self._active:
            return
        self._active = active
        self._apply_active_style()

    def is_active(self) -> bool:
        return self._active

    def set_laps(self, laps: list[LapTelemetry]) -> None:
        if self._chart is not None:
            self._chart.set_laps(laps)
        if self._bar is not None:
            self._bar.set_laps(laps)

    def set_axis_kind(self, kind: str) -> None:
        if self._chart is not None:
            self._chart.set_axis_kind(kind)

    def set_cursor_x(self, x: float) -> None:
        if self._chart is not None:
            self._chart.set_cursor_x(x)

    def hide_cursor(self) -> None:
        if self._chart is not None:
            self._chart.hide_cursor()

    def set_channels(self, channels: list[str]) -> None:
        # Update the model first so it persists.
        self.component.channels = list(channels)
        if self._chart is not None:
            self._chart.set_channels(list(channels))
        if self._bar is not None:
            self._bar.set_channels(list(channels))
        self._refresh_channels_label()

    def chart(self) -> MultiChannelChart | None:
        return self._chart

    # ---- helpers ------------------------------------------------------

    def _apply_active_style(self) -> None:
        if self._active:
            self.setStyleSheet(
                "#componentCard { border: 1px solid #4ea1ff; "
                "border-radius: 3px; }"
            )
        else:
            self.setStyleSheet(
                "#componentCard { border: 1px solid #2a2f36; "
                "border-radius: 3px; }"
            )

    def _refresh_channels_label(self) -> None:
        chans = self.component.channels
        if not chans:
            text = tr("(no channels)")
        elif len(chans) <= 3:
            text = ", ".join(chans)
        else:
            text = tr("{first} (+{rest} more)").format(
                first=", ".join(chans[:3]),
                rest=len(chans) - 3,
            )
        self._channels_lbl.setText(text)

    def _on_overlay_toggled(self, checked: bool) -> None:
        self.component.options["overlay"] = bool(checked)
        if self._chart is not None:
            self._chart.set_overlay_mode(bool(checked))
        self.options_changed.emit(self)

    def _on_normalize_toggled(self, checked: bool) -> None:
        self.component.options["normalize"] = bool(checked)
        if self._chart is not None:
            self._chart.set_normalize(bool(checked))
        self.options_changed.emit(self)


# ---------------------------------------------------------------------------
# Workbook tab
# ---------------------------------------------------------------------------


class WorkbookTab(QWidget):
    """The new MoTeC-style Telemetry tab.

    Owns lap loading, the active :class:`Workbook` and the per-worksheet
    splitter of :class:`_ComponentCard` widgets. Forwards cursor sync
    across every card and routes the global :class:`ChannelsDock`
    selection to the *active* card so the dock keeps being useful
    while we work on a proper channel browser (Phase 3).
    """

    def __init__(
        self,
        loader: LapLoader,
        signals: SignalBus,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._loader = loader
        self._signals = signals

        # Active workbook state.
        self._workbook: Workbook = self._load_persisted_or_default()
        # Map worksheet index -> (splitter, [card,...]).
        self._worksheet_widgets: dict[
            int, tuple[QSplitter, list[_ComponentCard]]
        ] = {}
        self._active_card: _ComponentCard | None = None
        # Lap state.
        self._requested_paths: list[Path] = []
        self._loaded_laps: dict[Path, LapTelemetry] = {}
        self._axis_kind: str = self._restore_axis_kind()

        # ---- top toolbar ---------------------------------------------
        toolbar = QToolBar(self)

        toolbar.addWidget(QLabel(tr("Workbook:"), self))
        self._wb_combo = QComboBox(self)
        self._wb_combo.setMinimumWidth(180)
        self._reload_workbook_combo(select_name=self._workbook.name)
        self._wb_combo.currentIndexChanged.connect(self._on_workbook_changed)
        toolbar.addWidget(self._wb_combo)

        templates_btn = QToolButton(self)
        templates_btn.setText(tr("Templates"))
        templates_btn.setPopupMode(QToolButton.InstantPopup)
        templates_menu = QMenu(templates_btn)
        for name in builtin_template_names():
            act = QAction(name, templates_menu)
            act.triggered.connect(
                lambda _checked=False, n=name: self._load_template(n)
            )
            templates_menu.addAction(act)
        templates_btn.setMenu(templates_menu)
        toolbar.addWidget(templates_btn)

        save_btn = QPushButton(tr("Save as…"), self)
        save_btn.clicked.connect(self._save_workbook_as)
        toolbar.addWidget(save_btn)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel(tr("X-axis: "), self))
        self._axis_combo = QComboBox(self)
        self._axis_combo.addItem(tr("Distance"), "distance")
        self._axis_combo.addItem(tr("Time"), "time")
        self._axis_combo.setCurrentIndex(
            0 if self._axis_kind == "distance" else 1
        )
        self._axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        toolbar.addWidget(self._axis_combo)

        toolbar.addSeparator()
        add_ws_btn = QPushButton(tr("+ Worksheet"), self)
        add_ws_btn.clicked.connect(self._add_worksheet)
        toolbar.addWidget(add_ws_btn)

        add_graph_btn = QPushButton(tr("+ Graph"), self)
        add_graph_btn.clicked.connect(
            lambda: self._add_component(kind="graph")
        )
        toolbar.addWidget(add_graph_btn)

        add_bar_btn = QPushButton(tr("+ Bar"), self)
        add_bar_btn.clicked.connect(
            lambda: self._add_component(kind="bar")
        )
        toolbar.addWidget(add_bar_btn)

        toolbar.addSeparator()
        self._caption = QLabel(tr("No laps selected"), self)
        self._caption.setStyleSheet(f"color: {MUTED_COLOR};")
        toolbar.addWidget(self._caption)

        # ---- worksheet tabs ------------------------------------------
        # A QTabWidget keeps things simple; each tab hosts a
        # QScrollArea wrapping a QSplitter of component cards.
        from PySide6.QtWidgets import QTabWidget  # local import: keep
        # top-of-file imports tidy.
        self._ws_tabs = QTabWidget(self)
        self._ws_tabs.setDocumentMode(True)
        self._ws_tabs.setMovable(True)
        self._ws_tabs.setTabsClosable(True)
        self._ws_tabs.tabCloseRequested.connect(self._on_close_worksheet)
        self._ws_tabs.currentChanged.connect(self._on_worksheet_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)
        layout.addWidget(toolbar)
        layout.addWidget(self._ws_tabs, 1)

        self._rebuild_worksheet_tabs()
        # Restore last-used worksheet index.
        idx = int(self._settings().value(_SETTINGS_WS_INDEX, 0) or 0)
        if 0 <= idx < self._ws_tabs.count():
            self._ws_tabs.setCurrentIndex(idx)

        # ---- signal wiring -------------------------------------------
        signals.laps_selected.connect(self._on_laps_selected)
        signals.channels_changed.connect(self._on_channels_changed)
        loader.lap_loaded.connect(self._on_lap_loaded)
        loader.lap_failed.connect(self._on_lap_failed)
        signals.cursor_moved.connect(self._on_external_cursor)
        signals.cursor_left.connect(self._on_external_cursor_left)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _settings(self) -> QSettings:
        return QSettings(ORG, APP)

    def _restore_axis_kind(self) -> str:
        val = self._settings().value(_SETTINGS_AXIS, "distance")
        return "time" if str(val) == "time" else "distance"

    def _load_persisted_or_default(self) -> Workbook:
        name = self._settings().value(_SETTINGS_WORKBOOK, "")
        if name:
            for path in list_user_workbooks():
                if path.stem == name or path.name == name:
                    try:
                        return load_workbook(path)
                    except Exception:  # noqa: BLE001
                        LOG.warning(
                            "failed to load persisted workbook %s; "
                            "falling back to default", path,
                        )
                        break
            # Try built-in by name.
            if name in builtin_template_names():
                try:
                    return builtin_template(name)
                except Exception:  # noqa: BLE001
                    pass
        return default_workbook()

    def _persist_active_workbook_name(self) -> None:
        self._settings().setValue(_SETTINGS_WORKBOOK, self._workbook.name)

    # ------------------------------------------------------------------
    # Worksheet / component construction
    # ------------------------------------------------------------------

    def _rebuild_worksheet_tabs(self) -> None:
        # Clear existing tabs.
        self._ws_tabs.blockSignals(True)
        while self._ws_tabs.count():
            w = self._ws_tabs.widget(0)
            self._ws_tabs.removeTab(0)
            if w is not None:
                w.deleteLater()
        self._worksheet_widgets.clear()
        self._active_card = None

        if not self._workbook.worksheets:
            # Ensure at least one empty worksheet exists so users have
            # somewhere to drop a component.
            self._workbook.worksheets.append(Worksheet(title=tr("Sheet 1")))

        for idx, ws in enumerate(self._workbook.worksheets):
            page = self._build_worksheet_page(idx, ws)
            self._ws_tabs.addTab(page, ws.title)
        self._ws_tabs.blockSignals(False)

    def _build_worksheet_page(self, idx: int, ws: Worksheet) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        host = QWidget()
        vbox = QVBoxLayout(host)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Vertical, host)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(4)
        vbox.addWidget(splitter)

        cards: list[_ComponentCard] = []
        for comp in ws.components:
            card = self._make_card(comp)
            cards.append(card)
            splitter.addWidget(card)

        if cards:
            sizes = self._restore_splitter_sizes(idx, len(cards))
            splitter.setSizes(sizes)
            cards[0].set_active(True)
            self._active_card = cards[0]
        else:
            placeholder = QLabel(
                tr("This worksheet is empty. Add a component with "
                   "“+ Graph”."),
                host,
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(f"color: {MUTED_COLOR}; padding: 24px;")
            splitter.addWidget(placeholder)

        scroll.setWidget(host)
        self._worksheet_widgets[idx] = (splitter, cards)
        # Persist splitter sizes whenever the user drags a handle.
        splitter.splitterMoved.connect(
            lambda _pos, _idx, s=splitter, sheet_idx=idx:
            self._persist_splitter_sizes(sheet_idx, s.sizes())
        )
        return scroll

    def _make_card(self, comp: Component) -> _ComponentCard:
        card = _ComponentCard(comp, parent=self)
        card.activated.connect(self._on_card_activated)
        card.remove_requested.connect(self._on_card_remove)
        card.move_requested.connect(self._on_card_move)
        # Push current lap / axis state so newly-created cards render
        # immediately even mid-session.
        laps = self._ordered_loaded_laps()
        if laps:
            card.set_laps(laps)
        card.set_axis_kind(self._axis_kind)
        if card.chart() is not None:
            card.chart().cursor_moved.connect(self._signals.cursor_moved)
            card.chart().cursor_left.connect(self._signals.cursor_left)
        return card

    # ------------------------------------------------------------------
    # Toolbar handlers
    # ------------------------------------------------------------------

    def _reload_workbook_combo(self, select_name: str | None = None) -> None:
        self._wb_combo.blockSignals(True)
        self._wb_combo.clear()
        # Built-in templates first.
        for name in builtin_template_names():
            self._wb_combo.addItem(
                tr("[Template] {n}").format(n=name), ("builtin", name)
            )
        # Then user workbooks from disk.
        for path in list_user_workbooks():
            self._wb_combo.addItem(path.stem, ("user", str(path)))
        # Try to re-select the requested name (template or user file stem).
        target_idx = -1
        for i in range(self._wb_combo.count()):
            kind, payload = self._wb_combo.itemData(i)
            if kind == "builtin" and payload == select_name:
                target_idx = i
                break
            if kind == "user" and Path(payload).stem == select_name:
                target_idx = i
                break
        if target_idx < 0 and self._wb_combo.count():
            target_idx = 0
        if target_idx >= 0:
            self._wb_combo.setCurrentIndex(target_idx)
        self._wb_combo.blockSignals(False)

    def _on_workbook_changed(self, index: int) -> None:
        if index < 0:
            return
        data = self._wb_combo.itemData(index)
        if not data:
            return
        kind, payload = data
        try:
            if kind == "builtin":
                self._workbook = builtin_template(payload)
            else:
                self._workbook = load_workbook(Path(payload))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self, tr("Workbook"),
                tr("Failed to load workbook: {err}").format(err=exc),
            )
            return
        self._rebuild_worksheet_tabs()
        self._persist_active_workbook_name()

    def _load_template(self, name: str) -> None:
        try:
            self._workbook = builtin_template(name)
        except KeyError:
            return
        self._rebuild_worksheet_tabs()
        idx = self._wb_combo.findData(("builtin", name))
        if idx >= 0:
            self._wb_combo.blockSignals(True)
            self._wb_combo.setCurrentIndex(idx)
            self._wb_combo.blockSignals(False)
        self._persist_active_workbook_name()

    def _save_workbook_as(self) -> None:
        name, ok = QInputDialog.getText(
            self, tr("Save workbook"), tr("Workbook name:"),
            text=self._workbook.name,
        )
        if not ok or not name.strip():
            return
        self._workbook.name = name.strip()
        try:
            save_user_workbook(self._workbook)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self, tr("Save workbook"),
                tr("Failed to save: {err}").format(err=exc),
            )
            return
        self._signals.status_message.emit(
            tr("Saved workbook ‘{n}’.").format(n=self._workbook.name), 4000,
        )
        self._reload_workbook_combo(select_name=self._workbook.name)
        self._persist_active_workbook_name()

    def _on_axis_changed(self, index: int) -> None:
        kind = self._axis_combo.itemData(index) or "distance"
        if kind == self._axis_kind:
            return
        self._axis_kind = kind
        for _splitter, cards in self._worksheet_widgets.values():
            for card in cards:
                card.set_axis_kind(kind)
        self._signals.x_axis_changed.emit(kind)
        self._settings().setValue(_SETTINGS_AXIS, kind)

    def _add_worksheet(self) -> None:
        name, ok = QInputDialog.getText(
            self, tr("New worksheet"), tr("Worksheet name:"),
            text=tr("Sheet {n}").format(n=len(self._workbook.worksheets) + 1),
        )
        if not ok or not name.strip():
            return
        ws = Worksheet(title=name.strip())
        self._workbook.worksheets.append(ws)
        self._rebuild_worksheet_tabs()
        self._ws_tabs.setCurrentIndex(self._ws_tabs.count() - 1)

    def _on_close_worksheet(self, index: int) -> None:
        if not (0 <= index < len(self._workbook.worksheets)):
            return
        if len(self._workbook.worksheets) <= 1:
            QMessageBox.information(
                self, tr("Worksheet"),
                tr("A workbook needs at least one worksheet."),
            )
            return
        ws = self._workbook.worksheets[index]
        resp = QMessageBox.question(
            self, tr("Close worksheet"),
            tr("Remove worksheet ‘{n}’ from this workbook?").format(
                n=ws.title
            ),
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        self._workbook.worksheets.pop(index)
        self._rebuild_worksheet_tabs()

    def _on_worksheet_changed(self, index: int) -> None:
        self._settings().setValue(_SETTINGS_WS_INDEX, int(index))
        bundle = self._worksheet_widgets.get(index)
        if bundle:
            _splitter, cards = bundle
            self._active_card = cards[0] if cards else None
            for c in cards:
                c.set_active(c is self._active_card)
            self._push_active_card_channels_to_dock()

    def _add_component(self, kind: str = "graph") -> None:
        ws_idx = self._ws_tabs.currentIndex()
        if not (0 <= ws_idx < len(self._workbook.worksheets)):
            return
        ws = self._workbook.worksheets[ws_idx]
        if kind == "graph":
            title = tr("New graph")
        else:
            title = tr("New {k}").format(k=kind)
        ws.components.append(Component(type=kind, title=title, channels=[]))
        self._rebuild_worksheet_tabs()
        self._ws_tabs.setCurrentIndex(ws_idx)

    # ------------------------------------------------------------------
    # Card / SignalBus routing
    # ------------------------------------------------------------------

    def _on_card_activated(self, card: _ComponentCard) -> None:
        if card is self._active_card:
            return
        if self._active_card is not None:
            self._active_card.set_active(False)
        self._active_card = card
        card.set_active(True)
        self._push_active_card_channels_to_dock()

    def _on_card_remove(self, card: _ComponentCard) -> None:
        ws_idx = self._ws_tabs.currentIndex()
        if not (0 <= ws_idx < len(self._workbook.worksheets)):
            return
        ws = self._workbook.worksheets[ws_idx]
        # Identify the component by identity (Component is unhashable
        # because dataclass with mutable fields → compare by ``is``).
        try:
            comp_idx = next(
                i for i, c in enumerate(ws.components) if c is card.component
            )
        except StopIteration:
            return
        ws.components.pop(comp_idx)
        self._rebuild_worksheet_tabs()
        self._ws_tabs.setCurrentIndex(ws_idx)

    def _on_card_move(self, card: _ComponentCard, delta: int) -> None:
        ws_idx = self._ws_tabs.currentIndex()
        if not (0 <= ws_idx < len(self._workbook.worksheets)):
            return
        ws = self._workbook.worksheets[ws_idx]
        try:
            comp_idx = next(
                i for i, c in enumerate(ws.components) if c is card.component
            )
        except StopIteration:
            return
        new_idx = comp_idx + int(delta)
        if not (0 <= new_idx < len(ws.components)) or new_idx == comp_idx:
            return
        ws.components.insert(new_idx, ws.components.pop(comp_idx))
        # Drop persisted sizes for this sheet — the layout changed and
        # the old per-index heights no longer match the new card order.
        self._clear_splitter_sizes(ws_idx)
        self._rebuild_worksheet_tabs()
        self._ws_tabs.setCurrentIndex(ws_idx)

    # ------------------------------------------------------------------
    # Splitter-size persistence helpers
    # ------------------------------------------------------------------

    def _sizes_key(self, ws_idx: int) -> str:
        wb_name = self._workbook.name or "_"
        return f"{_SETTINGS_SIZES_PREFIX}/{wb_name}/{ws_idx}"

    def _persist_splitter_sizes(
        self, ws_idx: int, sizes: list[int]
    ) -> None:
        if not sizes:
            return
        self._settings().setValue(
            self._sizes_key(ws_idx), ",".join(str(int(s)) for s in sizes)
        )

    def _restore_splitter_sizes(
        self, ws_idx: int, n_cards: int
    ) -> list[int]:
        raw = self._settings().value(self._sizes_key(ws_idx), "")
        if not raw:
            return [100] * n_cards
        try:
            sizes = [int(s) for s in str(raw).split(",") if s.strip()]
        except ValueError:
            return [100] * n_cards
        if len(sizes) != n_cards or any(s < 0 for s in sizes):
            return [100] * n_cards
        return sizes

    def _clear_splitter_sizes(self, ws_idx: int) -> None:
        self._settings().remove(self._sizes_key(ws_idx))

    def _push_active_card_channels_to_dock(self) -> None:
        if self._active_card is None:
            return
        # Mirror the active card's channels back to the global picker
        # so the user sees what's currently plotted.
        self._signals.channels_requested.emit(
            list(self._active_card.component.channels)
        )

    def _on_channels_changed(self, channels: list[str]) -> None:
        # The global channels dock toggled — apply to the active card.
        if self._active_card is None:
            return
        self._active_card.set_channels(list(channels))

    # ------------------------------------------------------------------
    # Lap loading / cursor sync
    # ------------------------------------------------------------------

    def _ordered_loaded_laps(self) -> list[LapTelemetry]:
        return [
            self._loaded_laps[p]
            for p in self._requested_paths
            if p in self._loaded_laps
        ]

    def _on_laps_selected(self, paths: list[Path]) -> None:
        self._requested_paths = [Path(p) for p in paths]
        wanted = set(self._requested_paths)
        self._loaded_laps = {
            p: lap for p, lap in self._loaded_laps.items() if p in wanted
        }
        if not self._requested_paths:
            self._caption.setText(tr("No laps selected"))
            for _s, cards in self._worksheet_widgets.values():
                for c in cards:
                    c.set_laps([])
            return
        missing = [
            p for p in self._requested_paths if p not in self._loaded_laps
        ]
        if missing:
            self._caption.setText(
                tr("Loading {n} of {total} lap(s)…").format(
                    n=len(missing), total=len(self._requested_paths),
                )
            )
            for path in missing:
                self._loader.request(path)
        self._refresh_all_cards()

    def _on_lap_loaded(self, path: Path, lap: LapTelemetry) -> None:
        path = Path(path)
        if path not in self._requested_paths:
            return
        self._loaded_laps[path] = lap
        self._refresh_all_cards()

    def _on_lap_failed(self, path: Path, message: str) -> None:
        self._signals.status_message.emit(
            tr("Failed to load {name}: {error}").format(
                name=Path(path).name, error=message,
            ),
            8000,
        )

    def _refresh_all_cards(self) -> None:
        laps = self._ordered_loaded_laps()
        if laps:
            cols = list(laps[0].enriched.columns)
            self._signals.available_columns_changed.emit(cols)
        for _s, cards in self._worksheet_widgets.values():
            for c in cards:
                c.set_laps(laps)
        if not laps:
            return
        if len(laps) == len(self._requested_paths):
            self._caption.setText(tr("{n} lap(s)").format(n=len(laps)))
        else:
            self._caption.setText(
                tr("{n} of {total} lap(s) loaded").format(
                    n=len(laps), total=len(self._requested_paths),
                )
            )

    def _on_external_cursor(self, x: float) -> None:
        # A chart elsewhere moved its crosshair → mirror across every
        # card in every worksheet so the whole workbook stays in lockstep.
        for _s, cards in self._worksheet_widgets.values():
            for c in cards:
                c.set_cursor_x(x)

    def _on_external_cursor_left(self) -> None:
        for _s, cards in self._worksheet_widgets.values():
            for c in cards:
                c.hide_cursor()


__all__ = ["WorkbookTab"]
