"""Stacked multi-channel chart with synchronized cursor and x-axis.

Built on a vertical layout of :class:`pyqtgraph.PlotWidget` instances
sharing one :class:`pyqtgraph.ViewBox` x-range. A single crosshair (x
line) is drawn on each child plot; mouse moves on any chart broadcast
``cursor_moved`` so sibling docks (a future track-map / GG-circle) can
follow.

Performance properties (validated empirically):

* Adding/removing a channel does **not** rebuild the other plots — only
  the affected row's ``PlotDataItem`` is created or destroyed.
* X arrays are cached per ``(lap, axis_kind)`` and y arrays per
  ``(lap, column, axis_kind, n)`` via :class:`LapArrayCache`.
* For series above 20k samples, LTTB downsamples to 4k points before
  it ever crosses into the GUI thread.
* Cross-trace hover does *not* aggregate (no Plotly ``x unified`` cost):
  per-trace hover is opt-in via the chart context menu.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QPen
from PySide6.QtWidgets import QSizePolicy, QSplitter, QVBoxLayout, QWidget

from ...telemetry import LapTelemetry, channel_info
from ...telemetry.comparison import LapComparison
from ...telemetry.sectors import insim_split_distances_m, sector_times_s
from ..i18n import tr
from ..theme import CURSOR_COLOR, GRID_COLOR, MUTED_COLOR, TEXT_COLOR, trace_color
from .lap_arrays import LapArrayCache


class _Row(QObject):
    """One PlotWidget + the trace items currently drawn on it."""

    def __init__(self, column: str, units: str, parent: "MultiChannelChart") -> None:
        super().__init__(parent)
        self.column = column
        self.units = units
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(110)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.getAxis("left").setLabel(_axis_label(column, units),
                                           color=TEXT_COLOR)
        self.plot.getAxis("bottom").setStyle(showValues=False)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        for ax in ("left", "bottom"):
            axis = self.plot.getAxis(ax)
            axis.setPen(pg.mkPen(GRID_COLOR))
            axis.setTextPen(pg.mkPen(MUTED_COLOR))
        # Crosshair (one per row; visibility toggled by parent on hover).
        self.cursor = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(CURSOR_COLOR, width=1, style=Qt.DashLine),
        )
        self.cursor.setZValue(10)
        self.cursor.hide()
        self.plot.addItem(self.cursor, ignoreBounds=True)
        # column → PlotDataItem; one item per (lap, channel) but we
        # store them in MultiChannelChart, not per row.
        self.items: Dict[int, pg.PlotDataItem] = {}
        # InSim split markers (one InfiniteLine per split). Cleared and
        # rebuilt whenever the lap set or x-axis kind changes.
        self.sector_lines: List[pg.InfiniteLine] = []
        # Optional theoretical-best overlay (delta row only).
        self.theo_best_item: pg.PlotDataItem | None = None


class MultiChannelChart(QWidget):
    """A vertically-stacked, x-linked, cursor-synced multi-channel chart.

    Public API::

        chart = MultiChannelChart()
        chart.set_axis_kind("distance" | "time")
        chart.set_laps([lap1, lap2])
        chart.set_channels(["speed_ms", "throttle", "brake"])
    """

    # External cursor sync (consumed by the dock owner).
    cursor_moved = Signal(float)
    cursor_left = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cache = LapArrayCache()
        self._laps: list[LapTelemetry] = []
        self._channels: list[str] = []
        self._axis_kind: str = "distance"
        self._rows: Dict[str, _Row] = {}
        self._row_order: List[str] = []
        # Optional delta-vs-reference row, shown above the channels
        # when there are >=2 laps and the x-axis is distance.
        self._delta_row: _Row | None = None
        # Splitter so the user can drag row heights to taste.
        self._splitter = QSplitter(Qt.Vertical, self)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(2)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._splitter)

        # Defer linking + cursor wiring to a single shot post-show so
        # the splitter has finalized its child geometry first.
        self._link_timer = QTimer(self)
        self._link_timer.setSingleShot(True)
        self._link_timer.setInterval(0)
        self._link_timer.timeout.connect(self._sync_x_link)

        # Throttle external cursor broadcasts to the screen refresh rate.
        self._cursor_throttle = QTimer(self)
        self._cursor_throttle.setSingleShot(True)
        self._cursor_throttle.setInterval(16)  # ~60 fps
        self._pending_cursor: float | None = None
        self._cursor_throttle.timeout.connect(self._flush_cursor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_axis_kind(self, kind: str) -> None:
        if kind not in ("distance", "time"):
            return
        if kind == self._axis_kind:
            return
        self._axis_kind = kind
        self._sync_delta_row()
        self._rebuild_traces()

    def set_laps(self, laps: Sequence[LapTelemetry]) -> None:
        # Drop cache entries for laps we're about to forget.
        new_ids = {id(lap) for lap in laps}
        for lap in self._laps:
            if id(lap) not in new_ids:
                self._cache.drop_lap(lap)
        self._laps = list(laps)
        self._sync_delta_row()
        self._rebuild_traces()

    def set_channels(self, channels: Sequence[str]) -> None:
        new_set = set(channels)
        old_set = set(self._channels)
        if new_set == old_set and list(channels) == self._channels:
            return
        self._channels = list(channels)
        self._sync_rows()
        self._rebuild_traces()

    def set_cursor_x(self, x: float, source: "MultiChannelChart | None" = None) -> None:
        """Set the crosshair on every row to ``x``. Re-entrancy safe."""
        if source is self:
            return
        for row in self._rows.values():
            row.cursor.setPos(x)
            row.cursor.show()
        if self._delta_row is not None:
            self._delta_row.cursor.setPos(x)
            self._delta_row.cursor.show()

    def hide_cursor(self) -> None:
        for row in self._rows.values():
            row.cursor.hide()
        if self._delta_row is not None:
            self._delta_row.cursor.hide()

    def exportable_rows(self) -> list[tuple[str, QWidget]]:
        """Return visible telemetry rows as ``(channel, widget)``.

        Used by the charts dock to export each generated graph as PNG.
        """
        out: list[tuple[str, QWidget]] = []
        for col in self._row_order:
            row = self._rows.get(col)
            if row is None:
                continue
            out.append((col, row.plot))
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sync_rows(self) -> None:
        """Add/remove plot rows so they match ``self._channels``."""
        wanted = list(self._channels)
        # Remove rows no longer needed.
        for col in list(self._rows):
            if col not in wanted:
                row = self._rows.pop(col)
                row.plot.setParent(None)
                row.plot.deleteLater()
        # Add new rows in the requested order.
        existing_widgets = {
            self._splitter.widget(i): i for i in range(self._splitter.count())
        }
        for position, col in enumerate(wanted):
            if col in self._rows:
                # Reorder if needed.
                row = self._rows[col]
                cur_index = existing_widgets.get(row.plot)
                if cur_index is not None and cur_index != position:
                    self._splitter.insertWidget(position, row.plot)
                continue
            info = channel_info(col)
            row = _Row(col, info.units, self)
            row.plot.setToolTip(info.tooltip_html(translate=tr))
            self._rows[col] = row
            self._splitter.insertWidget(position, row.plot)
            self._wire_cursor(row)
        self._row_order = list(wanted)
        # Bottom row gets x tick labels; everyone else hides them.
        for i, col in enumerate(self._row_order):
            row = self._rows[col]
            is_bottom = i == len(self._row_order) - 1
            row.plot.getAxis("bottom").setStyle(showValues=is_bottom)
            row.plot.getAxis("bottom").setLabel(
                _x_axis_label(self._axis_kind) if is_bottom else None,
                color=TEXT_COLOR,
            )
        self._update_min_height()
        self._link_timer.start()

    def _sync_delta_row(self) -> None:
        """Insert/remove the delta-vs-reference row at index 0.

        Shown only when there are >=2 laps and the x-axis is distance.
        """
        want = len(self._laps) >= 2 and self._axis_kind == "distance"
        if want and self._delta_row is None:
            row = _Row("__delta__", "s", self)
            row.plot.getAxis("left").setLabel("Δt vs ref [s]",
                                              color=TEXT_COLOR)
            self._delta_row = row
            self._splitter.insertWidget(0, row.plot)
            self._wire_cursor(row)
            self._link_timer.start()
        elif not want and self._delta_row is not None:
            self._delta_row.plot.setParent(None)
            self._delta_row.plot.deleteLater()
            self._delta_row = None
        self._update_min_height()

    def _update_min_height(self) -> None:
        """Size hint = (rows * row-min) + handles, so a wrapping
        QScrollArea can show its vertical bar when channels overflow.
        """
        n = len(self._row_order) + (1 if self._delta_row is not None else 0)
        if n <= 0:
            self.setMinimumHeight(0)
            return
        per_row = 110
        handle = self._splitter.handleWidth()
        total = n * per_row + max(0, n - 1) * handle
        self.setMinimumHeight(total)

    def _rebuild_delta_traces(self) -> None:
        """Recompute delta-vs-reference traces for every non-ref lap."""
        row = self._delta_row
        if row is None:
            return
        for item in list(row.items.values()):
            row.plot.removeItem(item)
        row.items.clear()
        if row.theo_best_item is not None:
            row.plot.removeItem(row.theo_best_item)
            row.theo_best_item = None
        if len(self._laps) < 2 or self._axis_kind != "distance":
            return
        ref = self._laps[0]
        for lap_idx, lap in enumerate(self._laps[1:], start=1):
            try:
                cmp = LapComparison.from_laps(ref, lap)
                d = cmp.distance_grid_m
                dt = cmp.delta_time_s
            except Exception:
                continue
            if d.size < 2 or dt.size != d.size:
                continue
            color = trace_color(lap_idx)
            pen = QPen(pg.mkColor(color))
            pen.setWidthF(1.2)
            pen.setCosmetic(True)
            item = pg.PlotDataItem(
                d, dt, pen=pen, antialias=True,
                autoDownsample=False, clipToView=False,
                skipFiniteCheck=True,
                name=f"delta_lap{lap_idx}",
            )
            row.plot.addItem(item)
            row.items[lap_idx] = item
        # Zero baseline for reference.
        zero = pg.InfiniteLine(
            angle=0, pos=0.0, movable=False,
            pen=pg.mkPen(MUTED_COLOR, width=1, style=Qt.DotLine),
        )
        row.plot.addItem(zero, ignoreBounds=True)
        self._rebuild_theo_best_trace()
        vb = row.plot.getViewBox()
        vb.enableAutoRange(axis=vb.YAxis, enable=True)
        vb.autoRange()

    def _split_distances(self) -> List[float]:
        """Sorted split distances (m) from the reference lap, if any."""
        if not self._laps:
            return []
        try:
            return list(insim_split_distances_m(self._laps[0]))
        except Exception:
            return []

    def _ref_total_distance(self) -> float:
        if not self._laps:
            return 0.0
        ref = self._laps[0]
        try:
            d = ref.enriched.get("distance_m")
            if d is None:
                return 0.0
            arr = d.to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)] if arr.size else arr
            return float(arr[-1]) if arr.size else 0.0
        except Exception:
            return 0.0

    def _rebuild_theo_best_trace(self) -> None:
        """Stepped trace of theoretical-best cumulative gap vs ref lap.

        For each sector boundary we take the *best* sector time across
        all loaded laps (excluding race-start laps) and accumulate the
        gap against the reference lap's sector times. Drawn as a step
        curve so a sub-second improvement at a single sector is
        instantly visible.
        """
        row = self._delta_row
        if row is None or len(self._laps) < 2:
            return
        splits = self._split_distances()
        if not splits:
            return
        total_d = self._ref_total_distance()
        if total_d <= splits[-1]:
            return
        boundaries: List[float] = list(splits)
        # sector_times_s uses (n+1) sectors for n boundaries.
        try:
            ref_times = sector_times_s(
                self._laps[0], boundaries_m=boundaries,
            )
        except Exception:
            return
        if not ref_times:
            return
        per_sector_min = list(ref_times)
        for lap in self._laps:
            if getattr(lap, "is_race_start", False):
                continue
            try:
                times = sector_times_s(lap, boundaries_m=boundaries)
            except Exception:
                continue
            if len(times) != len(per_sector_min):
                continue
            for i, t in enumerate(times):
                if np.isfinite(t) and t < per_sector_min[i]:
                    per_sector_min[i] = float(t)
        # Cumulative gap vs reference, evaluated at each sector end.
        sector_ends = boundaries + [total_d]
        cum_gap = np.cumsum(
            np.array(per_sector_min, dtype=float)
            - np.array(ref_times, dtype=float)
        )
        # Build a step curve starting at (0, 0).
        xs = np.concatenate(([0.0], np.asarray(sector_ends, dtype=float)))
        ys = np.concatenate(([0.0], cum_gap))
        pen = QPen(pg.mkColor("#ffd166"))
        pen.setWidthF(1.5)
        pen.setStyle(Qt.DashLine)
        pen.setCosmetic(True)
        item = pg.PlotDataItem(
            xs, ys, pen=pen, stepMode="left", antialias=True,
            name="theoretical_best",
        )
        item.setZValue(5)
        row.plot.addItem(item)
        row.theo_best_item = item

    def _sync_sector_markers(self) -> None:
        """Refresh InSim split markers on every visible row."""
        # Clear existing markers everywhere first.
        all_rows: List[_Row] = list(self._rows.values())
        if self._delta_row is not None:
            all_rows.append(self._delta_row)
        for r in all_rows:
            for line in r.sector_lines:
                r.plot.removeItem(line)
            r.sector_lines.clear()
        if self._axis_kind != "distance":
            return
        splits = self._split_distances()
        if not splits:
            return
        pen = pg.mkPen("#ffd166", width=1, style=Qt.DotLine)
        for r in all_rows:
            for d in splits:
                line = pg.InfiniteLine(
                    pos=float(d), angle=90, movable=False, pen=pen,
                )
                line.setZValue(2)
                r.plot.addItem(line, ignoreBounds=True)
                r.sector_lines.append(line)

    def _sync_x_link(self) -> None:
        """Link every row's x-axis to the first row's ViewBox."""
        if not self._row_order:
            return
        anchor = self._rows[self._row_order[0]].plot.getViewBox()
        anchor.enableAutoRange(axis=anchor.XAxis, enable=True)
        anchor.autoRange()
        for col in self._row_order[1:]:
            vb = self._rows[col].plot.getViewBox()
            vb.setXLink(anchor)
        if self._delta_row is not None:
            self._delta_row.plot.getViewBox().setXLink(anchor)

    def _rebuild_traces(self) -> None:
        """Redraw every (lap, channel) trace from the cache."""
        # Clear existing data items per row but keep the row itself.
        for row in self._rows.values():
            for item in list(row.items.values()):
                row.plot.removeItem(item)
            row.items.clear()
        self._rebuild_delta_traces()
        if not self._laps or not self._channels:
            return
        kind = self._axis_kind
        for lap_idx, lap in enumerate(self._laps):
            color = trace_color(lap_idx)
            pen = QPen(pg.mkColor(color))
            pen.setWidthF(1.0)
            pen.setCosmetic(True)
            try:
                x = self._cache.x(lap, kind)
            except Exception:
                continue
            if x.size == 0:
                continue
            for col in self._channels:
                row = self._rows.get(col)
                if row is None:
                    continue
                try:
                    xs, ys = self._cache.xy_decimated(lap, col, kind)
                except KeyError:
                    continue  # column missing on this lap
                item = pg.PlotDataItem(
                    xs, ys, pen=pen, antialias=True,
                    autoDownsample=False, clipToView=False,
                    skipFiniteCheck=True,
                    name=f"lap{lap_idx}:{col}",
                )
                row.plot.addItem(item)
                row.items[lap_idx] = item
        # After the data is in, autoscale both axes; X is propagated
        # through the link to the sibling rows.
        for row in self._rows.values():
            vb = row.plot.getViewBox()
            vb.enableAutoRange(axis=vb.XYAxes, enable=True)
            vb.autoRange()
        self._sync_sector_markers()
        self._link_timer.start()

    # ----- Cursor wiring -----------------------------------------------

    def _wire_cursor(self, row: _Row) -> None:
        plot = row.plot
        scene = plot.scene()

        def on_move(pos):
            vb = plot.getViewBox()
            if not plot.sceneBoundingRect().contains(pos):
                return
            mouse_point = vb.mapSceneToView(pos)
            x = float(mouse_point.x())
            self.set_cursor_x(x)
            self._pending_cursor = x
            if not self._cursor_throttle.isActive():
                self._cursor_throttle.start()

        def on_leave(_event):
            self.hide_cursor()
            self.cursor_left.emit()

        scene.sigMouseMoved.connect(on_move)
        # Hide cursor when the pointer leaves *all* rows; per-row leave
        # is too aggressive because moving between rows briefly leaves
        # the scene.
        plot.installEventFilter(self)

    def eventFilter(self, watched, event):  # type: ignore[override]
        if event.type() in (event.Type.Leave,):
            # Use a short timer so a fast hop between rows doesn't blink
            # the cursor off-then-on.
            QTimer.singleShot(80, self._maybe_hide_cursor)
        return False

    def _maybe_hide_cursor(self) -> None:
        # If the mouse is no longer inside any row's viewport, hide.
        for row in self._rows.values():
            if row.plot.underMouse():
                return
        self.hide_cursor()
        self.cursor_left.emit()

    def _flush_cursor(self) -> None:
        if self._pending_cursor is None:
            return
        x = self._pending_cursor
        self._pending_cursor = None
        self.cursor_moved.emit(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _axis_label(column: str, units: str) -> str:
    info = channel_info(column)
    label = info.label or column
    return f"{label} [{units}]" if units else label


def _x_axis_label(kind: str) -> str:
    return "Distance [m]" if kind == "distance" else "Time [s]"


__all__ = ["MultiChannelChart"]
