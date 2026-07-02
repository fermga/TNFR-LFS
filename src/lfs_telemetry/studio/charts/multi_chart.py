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

import contextlib
from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QPen
from PySide6.QtWidgets import QSizePolicy, QSplitter, QVBoxLayout, QWidget

from ...telemetry import LapTelemetry, channel_info
from ...telemetry.comparison import LapComparison
from ...telemetry.sectors import insim_split_distances_m, sector_times_s
from ..i18n import current_language, tr
from ..theme import CURSOR_COLOR, GRID_COLOR, MUTED_COLOR, TEXT_COLOR, trace_color
from .lap_arrays import LapArrayCache

# Palette used for channels overlayed in the same row (overlay mode
# only). Distinct from `trace_color`, which encodes lap index.
_OVERLAY_CHANNEL_PALETTE: tuple[str, ...] = (
    "#4ea3ff", "#ff6b6b", "#ffd166", "#06d6a0",
    "#a78bfa", "#ff9f43", "#48cae4", "#ef476f",
)
# Line styles used to disambiguate laps when channel colour is taken.
_OVERLAY_LAP_STYLES: tuple = (
    Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine,
)


class _Row(QObject):
    """One PlotWidget + the trace items currently drawn on it.

    In stacked mode a row maps to a single channel. In overlay mode
    a row hosts every channel that shares one unit; the legend lists
    the channels and the y-axis is shared.
    """

    def __init__(
        self,
        group_key: str,
        channels: list[str],
        units: str,
        parent: MultiChannelChart,
    ) -> None:
        super().__init__(parent)
        # ``column`` kept as alias for the group key so existing call
        # sites that read it (e.g. exports) keep working.
        self.group_key = group_key
        self.column = group_key
        self.channels = list(channels)
        self.units = units
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(110)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot.showGrid(x=True, y=True, alpha=0.18)
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
        # (lap_idx, column) → PlotDataItem. In stacked mode the column
        # is always the row's single channel.
        self.items: dict[tuple[int, str], pg.PlotDataItem] = {}
        # InSim split markers (one InfiniteLine per split). Cleared and
        # rebuilt whenever the lap set or x-axis kind changes.
        self.sector_lines: list[pg.InfiniteLine] = []
        # Optional theoretical-best overlay (delta row only).
        self.theo_best_item: pg.PlotDataItem | None = None
        # Lazily created when this row hosts >1 channel.
        self.legend: pg.LegendItem | None = None

    def ensure_legend(self) -> None:
        if self.legend is None:
            self.legend = self.plot.addLegend(
                offset=(8, 4), labelTextColor=TEXT_COLOR,
            )

    def clear_legend(self) -> None:
        if self.legend is not None:
            with contextlib.suppress(Exception):
                self.legend.clear()


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
        # Overlay mode: group channels with the same units into a single
        # row. Normalize: rescale each trace to its own 0–1 range so
        # channels with very different magnitudes stay comparable.
        self._overlay: bool = False
        self._normalize: bool = False
        self._rows: dict[str, _Row] = {}
        self._row_order: list[str] = []
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

    def set_overlay_mode(self, enabled: bool) -> None:
        """Toggle overlay-by-units mode.

        When enabled, channels with the same physical unit are drawn
        on the same row; lap index is encoded by line style instead of
        colour so the channel palette stays readable.
        """
        enabled = bool(enabled)
        if enabled == self._overlay:
            return
        self._overlay = enabled
        self._sync_rows()
        self._rebuild_traces()

    def set_normalize(self, enabled: bool) -> None:
        """Toggle per-trace 0–1 normalisation.

        Useful when overlaying channels with incompatible scales
        (e.g. throttle % vs vertical load N).
        """
        enabled = bool(enabled)
        if enabled == self._normalize:
            return
        self._normalize = enabled
        for row in self._rows.values():
            row.plot.getAxis("left").setLabel(
                self._row_axis_label(row.channels, row.units),
                color=TEXT_COLOR,
            )
        self._rebuild_traces()

    def set_cursor_x(self, x: float, source: MultiChannelChart | None = None) -> None:
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
        """Return visible telemetry rows as ``(name, widget)``.

        Used by the charts dock to export each generated graph as PNG.
        In overlay mode the name joins all channels in the group.
        """
        out: list[tuple[str, QWidget]] = []
        for key in self._row_order:
            row = self._rows.get(key)
            if row is None:
                continue
            name = (
                row.channels[0] if len(row.channels) == 1
                else "+".join(row.channels)
            )
            out.append((name, row.plot))
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_groups(self) -> list[tuple[str, list[str], str]]:
        """Return ``[(group_key, channels, units), ...]`` in display order.

        Stacked mode: one group per channel. Overlay mode: channels with
        the same units share a group, ordered by first-occurrence.
        """
        if not self._overlay:
            out: list[tuple[str, list[str], str]] = []
            for col in self._channels:
                info = channel_info(col)
                out.append((col, [col], info.units))
            return out
        order: list[str] = []
        by_unit: dict[str, list[str]] = {}
        for col in self._channels:
            units = channel_info(col).units or ""
            if units not in by_unit:
                order.append(units)
                by_unit[units] = []
            by_unit[units].append(col)
        return [
            (f"__group__{u or 'nounit'}", by_unit[u], u)
            for u in order
        ]

    def _row_axis_label(self, channels: list[str], units: str) -> str:
        if self._normalize:
            tail = f" [{units}]" if units else ""
            return f"normalized 0–1{tail}"
        if len(channels) == 1:
            return _axis_label(channels[0], units)
        return f"[{units}]" if units else "(mixed)"

    @staticmethod
    def _normalize_array(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        finite = y[np.isfinite(y)] if y.size else y
        if finite.size == 0:
            return y
        lo = float(finite.min())
        hi = float(finite.max())
        if hi - lo < 1e-12:
            return np.zeros_like(y)
        return (y - lo) / (hi - lo)

    def _sync_rows(self) -> None:
        """Add/remove plot rows so they match the channel groups."""
        groups = self._compute_groups()
        wanted_keys = [g[0] for g in groups]
        # Remove rows no longer needed.
        for key in list(self._rows):
            if key not in wanted_keys:
                row = self._rows.pop(key)
                row.plot.setParent(None)
                row.plot.deleteLater()
        existing_widgets = {
            self._splitter.widget(i): i for i in range(self._splitter.count())
        }
        for position, (key, channels, units) in enumerate(groups):
            existing = self._rows.get(key)
            if existing is not None and (
                existing.channels != channels or existing.units != units
            ):
                # Group composition changed (e.g. another channel of
                # the same unit was added): rebuild the row in place.
                existing.plot.setParent(None)
                existing.plot.deleteLater()
                self._rows.pop(key)
                existing = None
            if existing is not None:
                cur_index = existing_widgets.get(existing.plot)
                if cur_index is not None and cur_index != position:
                    self._splitter.insertWidget(position, existing.plot)
                continue
            row = _Row(key, channels, units, self)
            row.plot.setToolTip(
                channel_info(channels[0]).tooltip_html(
                    translate=tr,
                    language=current_language(),
                )
            )
            row.plot.getAxis("left").setLabel(
                self._row_axis_label(channels, units),
                color=TEXT_COLOR,
            )
            if len(channels) > 1:
                row.ensure_legend()
            self._rows[key] = row
            self._splitter.insertWidget(position, row.plot)
            self._wire_cursor(row)
        self._row_order = wanted_keys
        # Bottom row gets x tick labels; everyone else hides them.
        for i, key in enumerate(self._row_order):
            row = self._rows[key]
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
            row = _Row("__delta__", ["__delta__"], "s", self)
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
            row.items[(lap_idx, "__delta__")] = item
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

    def _split_distances(self) -> list[float]:
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
        boundaries: list[float] = list(splits)
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
        sector_ends = [*boundaries, total_d]
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
        all_rows: list[_Row] = list(self._rows.values())
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
            row.clear_legend()
        self._rebuild_delta_traces()
        if not self._laps or not self._channels:
            return
        kind = self._axis_kind
        n_laps = len(self._laps)
        for key in self._row_order:
            row = self._rows.get(key)
            if row is None:
                continue
            channels = row.channels
            single = len(channels) == 1
            for ch_idx, col in enumerate(channels):
                for lap_idx, lap in enumerate(self._laps):
                    try:
                        xs, ys = self._cache.xy_decimated(lap, col, kind)
                    except KeyError:
                        continue  # column missing on this lap
                    if xs.size == 0:
                        continue
                    if self._normalize:
                        ys = self._normalize_array(ys)
                    if single:
                        color = trace_color(lap_idx)
                        style = Qt.SolidLine
                    else:
                        color = _OVERLAY_CHANNEL_PALETTE[
                            ch_idx % len(_OVERLAY_CHANNEL_PALETTE)
                        ]
                        style = _OVERLAY_LAP_STYLES[
                            lap_idx % len(_OVERLAY_LAP_STYLES)
                        ]
                    pen = QPen(pg.mkColor(color))
                    pen.setWidthF(1.0)
                    pen.setStyle(style)
                    pen.setCosmetic(True)
                    if single:
                        name = col
                    elif n_laps > 1:
                        name = f"L{lap_idx + 1} · {col}"
                    else:
                        name = col
                    item = pg.PlotDataItem(
                        xs, ys, pen=pen, antialias=True,
                        autoDownsample=False, clipToView=False,
                        skipFiniteCheck=True,
                        name=name,
                    )
                    row.plot.addItem(item)
                    row.items[(lap_idx, col)] = item
        # After the data is in, autoscale both axes; X is propagated
        # through the link to the sibling rows.
        for row in self._rows.values():
            vb = row.plot.getViewBox()
            if self._normalize:
                vb.enableAutoRange(axis=vb.XAxis, enable=True)
                vb.setYRange(-0.02, 1.02, padding=0)
            else:
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
