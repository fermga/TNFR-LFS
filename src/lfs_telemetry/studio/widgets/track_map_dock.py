"""Track-map dock: top-down XY view with a synchronized cursor dot.

Subscribes to:

* ``laps_selected``  → rebuilds the track outline using the first lap.
* ``cursor_moved``   → moves the cursor dot to ``(x, y)`` at that
  distance along the racing line.
* ``cursor_left``    → hides the cursor dot.
* ``x_axis_changed`` → tracks distance vs time mode; the cursor dot is
  only drawn in distance mode (time → distance mapping per-lap is
  expensive and ambiguous).

Renders with ``pyqtgraph`` so panning/zoom comes for free.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QRectF, QTimer
from PySide6.QtGui import QImage, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...telemetry import LapTelemetry
from ...telemetry.track.overlay import (
    DEFAULT_CALIBRATION,
    OverlayCalibration,
    OverlayExtent,
    compute_overlay_extent,
    compute_overlay_extent_for_image,
    find_overlay_image,
    load_overlay_calibrations,
    save_user_overlay_calibration,
    track_to_environment,
)
from ...telemetry.track_map import TrackMap
from ..models import LapLoader
from ..signals import SignalBus
from ..theme import CURSOR_COLOR, MUTED_COLOR, TEXT_COLOR, trace_color


class TrackMapDock(QWidget):
    """Top-down racing-line view with a moving cursor dot."""

    def __init__(
        self,
        loader: LapLoader,
        signals: SignalBus,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._loader = loader
        self._signals = signals
        self._axis_kind = "distance"
        self._maps: dict[Path, TrackMap] = {}
        self._loaded_laps: dict[Path, LapTelemetry] = {}
        self._anchor_map: TrackMap | None = None  # for cursor dot

        self._plot = pg.PlotWidget(self)
        self._plot.setBackground(None)  # inherit dark theme
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.setAspectLocked(True)
        self._plot.showGrid(x=True, y=True, alpha=0.12)
        self._plot.getAxis("left").setLabel("Y [m]", color=TEXT_COLOR)
        self._plot.getAxis("bottom").setLabel("X [m]", color=TEXT_COLOR)
        for ax in ("left", "bottom"):
            axis = self._plot.getAxis(ax)
            axis.setTextPen(pg.mkPen(MUTED_COLOR))

        # Cursor dot.
        self._dot = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen(CURSOR_COLOR, width=1.5),
            brush=pg.mkBrush(CURSOR_COLOR),
        )
        self._dot.setZValue(20)
        self._dot.hide()
        self._plot.addItem(self._dot)

        self._legend = QLabel(self)
        self._legend.setTextFormat(Qt.TextFormat.RichText)
        self._legend.setWordWrap(True)
        self._legend.setStyleSheet(
            "QLabel {"
            " background-color: rgba(15, 15, 18, 185);"
            " border: 1px solid rgba(120,120,130,150);"
            " border-radius: 6px;"
            " color: #cfd6dc;"
            " padding: 5px 7px;"
            " font-size: 11px;"
            "}"
        )
        self._legend.setText(
            "<b>Legend</b> · "
            "<span style='color:#6ab0ff'>●</span> cursor · "
            "<span style='color:#ffffff'>╌╌</span> ideal line · "
            "<span style='color:#ffd166'>···</span> KNW AI line · "
            "<span style='color:#ff5d6c'>●</span> apex/slowest decile · "
            "lap traces: selected-lap colors"
        )

        # Per-lap polyline items, keyed by capture path.
        self._lines: dict[Path, pg.PlotDataItem] = {}

        # Reference racing line (ideal line from racing_lines/<TRACK>).
        self._rline_item: pg.PlotDataItem | None = None
        self._apex_item: pg.ScatterPlotItem | None = None
        self._rline_cache: dict[str, tuple] = {}
        self._current_track: str | None = None

        # Per-car KNW AI line overlay.
        self._knw_item: pg.PlotDataItem | None = None
        self._knw_cache: dict[tuple, np.ndarray | None] = {}
        self._current_car: str | None = None

        # Track-map TIF overlay (per-environment top-down image).
        self._overlay_item: pg.ImageItem | None = None
        self._overlay_calibrations: dict[str, OverlayCalibration] = (
            load_overlay_calibrations()
        )
        self._overlay_image_cache: dict[Path, np.ndarray | None] = {}
        self._overlay_extent_cache: dict[str, OverlayExtent | None] = {}
        self._overlay_visible: bool = True
        self._overlay_opacity: float = 0.35
        self._current_env: str | None = None

        # UI controls for the overlay (opacity slider + show toggle).
        self._overlay_check = QCheckBox("Track image", self)
        self._overlay_check.setChecked(self._overlay_visible)
        self._overlay_check.toggled.connect(self._on_overlay_toggled)
        self._overlay_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._overlay_slider.setRange(0, 100)
        self._overlay_slider.setValue(int(self._overlay_opacity * 100))
        self._overlay_slider.setFixedWidth(140)
        self._overlay_slider.setToolTip("Track image opacity")
        self._overlay_slider.valueChanged.connect(self._on_overlay_opacity)
        overlay_label = QLabel("Opacity", self)
        overlay_label.setStyleSheet(f"color: {MUTED_COLOR};")
        self._overlay_calib_btn = QPushButton("Calibrate map…", self)
        self._overlay_calib_btn.setToolTip(
            "Nudge / scale the track image so it lines up with the racing"
            " line. Saved per environment under your user profile."
        )
        self._overlay_calib_btn.clicked.connect(self._on_calibrate_overlay)
        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(2, 0, 2, 0)
        controls_row.addWidget(self._overlay_check)
        controls_row.addStretch(1)
        controls_row.addWidget(self._overlay_calib_btn)
        controls_row.addWidget(overlay_label)
        controls_row.addWidget(self._overlay_slider)

        # ----- Replay transport bar -----------------------------------
        # Animate the cursor along the anchor lap (and ghost dots along
        # every other selected lap) at variable speed. Drives the same
        # ``cursor_moved`` signal used by the chart crosshairs, so every
        # dock follows in lockstep — no extra wiring needed.
        self._playback_speeds: tuple[float, ...] = (
            0.25, 0.5, 1.0, 2.0, 4.0, 8.0,
        )
        self._playback_speed_idx: int = 2  # 1.0×
        self._playback_t_s: float = 0.0
        self._playback_loop: bool = False
        self._playback_axis_kind_before: str | None = None
        self._anchor_path: Path | None = None
        # Per-lap monotone (time_s, distance_m) arrays for time→dist
        # interpolation. Populated from raw on lap-load.
        self._lap_t_d: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
        # Ghost dots for non-anchor selected laps, keyed by capture path.
        self._ghost_dots: dict[Path, pg.ScatterPlotItem] = {}

        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(33)  # ~30 Hz
        self._playback_timer.timeout.connect(self._on_playback_tick)

        style = self.style()

        def _tb(icon_enum, tip: str) -> QToolButton:
            btn = QToolButton(self)
            btn.setIcon(style.standardIcon(icon_enum))
            btn.setToolTip(tip)
            btn.setAutoRaise(True)
            return btn

        self._btn_skip_back = _tb(
            QStyle.StandardPixmap.SP_MediaSkipBackward,
            "Back to start (keep paused)",
        )
        self._btn_slow = _tb(
            QStyle.StandardPixmap.SP_MediaSeekBackward,
            "Slow motion (decrease playback speed)",
        )
        self._btn_play = _tb(
            QStyle.StandardPixmap.SP_MediaPlay,
            "Play / Pause animation along the lap",
        )
        self._btn_stop = _tb(
            QStyle.StandardPixmap.SP_MediaStop,
            "Stop and hide cursor",
        )
        self._btn_fast = _tb(
            QStyle.StandardPixmap.SP_MediaSeekForward,
            "Speed up playback",
        )
        self._btn_skip_fwd = _tb(
            QStyle.StandardPixmap.SP_MediaSkipForward,
            "Jump to end of lap",
        )

        self._speed_label = QLabel("1.0×", self)
        self._speed_label.setMinimumWidth(38)
        self._speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._speed_label.setStyleSheet(f"color: {TEXT_COLOR};")

        self._loop_check = QCheckBox("Loop", self)
        self._loop_check.setToolTip(
            "Restart from t=0 when the end of the lap is reached.",
        )

        self._scrub_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._scrub_slider.setRange(0, 1000)
        self._scrub_slider.setValue(0)
        self._scrub_slider.setToolTip(
            "Scrub through the lap. Drag to seek; the cursor and chart"
            " crosshairs follow.",
        )
        # While the user drags, suspend autoplay so the slider position
        # isn't fought by the timer.
        self._scrub_slider.sliderPressed.connect(
            self._on_scrub_pressed,
        )
        self._scrub_slider.sliderReleased.connect(
            self._on_scrub_released,
        )
        self._scrub_slider.valueChanged.connect(self._on_scrub_changed)
        self._scrub_was_playing: bool = False

        self._time_label = QLabel("00:00.000 / 00:00.000", self)
        self._time_label.setStyleSheet(f"color: {MUTED_COLOR};")
        self._time_label.setMinimumWidth(135)
        self._time_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        self._btn_skip_back.clicked.connect(self._on_skip_back)
        self._btn_slow.clicked.connect(self._on_slower)
        self._btn_play.clicked.connect(self._on_play_pause)
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_fast.clicked.connect(self._on_faster)
        self._btn_skip_fwd.clicked.connect(self._on_skip_forward)
        self._loop_check.toggled.connect(self._on_loop_toggled)

        replay_row = QHBoxLayout()
        replay_row.setContentsMargins(2, 0, 2, 0)
        for w in (
            self._btn_skip_back,
            self._btn_slow,
            self._btn_play,
            self._btn_stop,
            self._btn_fast,
            self._btn_skip_fwd,
        ):
            replay_row.addWidget(w)
        replay_row.addWidget(self._speed_label)
        replay_row.addSpacing(8)
        replay_row.addWidget(self._loop_check)
        replay_row.addWidget(self._scrub_slider, 1)
        replay_row.addWidget(self._time_label)
        # Disabled until at least one lap with a usable time-axis lands.
        self._set_replay_enabled(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._plot)
        layout.addLayout(controls_row)
        layout.addLayout(replay_row)
        layout.addWidget(self._legend)

        signals.laps_selected.connect(self._on_laps_selected)
        signals.cursor_moved.connect(self._on_cursor_moved)
        signals.cursor_left.connect(self._on_cursor_left)
        signals.x_axis_changed.connect(self._on_axis_changed)
        loader.lap_loaded.connect(self._on_lap_loaded)

    # ------------------------------------------------------------------
    # Public hooks
    # ------------------------------------------------------------------

    def _on_lap_loaded(self, path: Path, lap: LapTelemetry) -> None:
        path = Path(path)
        if path not in getattr(self, "_selection_order", []):
            return
        self._loaded_laps[path] = lap
        try:
            tmap = TrackMap.from_lap(lap)
        except Exception:
            return
        self._maps[path] = tmap
        # Cache (t_s, distance_m) for playback. Both columns exist in
        # every schema version since 1.0; if either is missing or the
        # lap is too short, leave the entry out — playback will simply
        # ignore that lap.
        try:
            df = lap.raw
            if (
                "time_ms" in df.columns
                and "current_lap_dist_m" in df.columns
                and len(df) >= 2
            ):
                t_ms = np.asarray(df["time_ms"], dtype=float)
                d_m = np.asarray(df["current_lap_dist_m"], dtype=float)
                mask = np.isfinite(t_ms) & np.isfinite(d_m)
                if mask.sum() >= 2:
                    t_s = (t_ms[mask] - t_ms[mask][0]) / 1000.0
                    d = d_m[mask]
                    # Force monotone t_s (sorted by time).
                    order = np.argsort(t_s)
                    self._lap_t_d[path] = (t_s[order], d[order])
        except Exception:  # noqa: BLE001
            pass
        self._redraw()
        self._refresh_replay_ui()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_laps_selected(self, paths: list[Path]) -> None:
        wanted = {Path(p) for p in paths}
        # Drop maps for laps no longer selected.
        for p in list(self._maps):
            if p not in wanted:
                self._maps.pop(p, None)
                self._loaded_laps.pop(p, None)
                self._lap_t_d.pop(p, None)
                ghost = self._ghost_dots.pop(p, None)
                if ghost is not None:
                    self._plot.removeItem(ghost)
        # Track lap order so the first selected becomes the anchor.
        self._selection_order = [Path(p) for p in paths]
        # Selection changed → reset playback so the slider/label reflect
        # the new anchor lap and ghosts vanish until the user replays.
        self._stop_playback(restore_axis=False)
        self._redraw()
        self._refresh_replay_ui()

    def _on_cursor_moved(self, x: float) -> None:
        if self._anchor_map is None or self._axis_kind != "distance":
            self._dot.hide()
            return
        d_arr = self._anchor_map.distance_m
        if d_arr.size < 2:
            return
        # Clamp + nearest neighbour. d_arr is monotonic.
        x = float(np.clip(x, d_arr[0], d_arr[-1]))
        i = int(np.searchsorted(d_arr, x))
        if i >= d_arr.size:
            i = d_arr.size - 1
        # Pick the closer of i-1, i.
        if i > 0 and (x - d_arr[i - 1]) < (d_arr[i] - x):
            i -= 1
        xm = float(self._anchor_map.x_m[i])
        ym = float(self._anchor_map.y_m[i])
        self._dot.setData([xm], [ym])
        self._dot.show()

    def _on_cursor_left(self) -> None:
        self._dot.hide()

    def _on_axis_changed(self, kind: str) -> None:
        self._axis_kind = kind
        if kind != "distance":
            self._dot.hide()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        # Remove existing polylines.
        for line in self._lines.values():
            self._plot.removeItem(line)
        self._lines.clear()
        order = getattr(self, "_selection_order", list(self._maps))
        # Anchor = first selected lap that has a map.
        self._anchor_map = None
        anchor_path: Path | None = None
        for path in order:
            tmap = self._maps.get(path)
            if tmap is None:
                continue
            idx = order.index(path)
            color = trace_color(idx)
            pen = QPen(pg.mkColor(color))
            pen.setWidthF(1.4)
            pen.setCosmetic(True)
            line = pg.PlotDataItem(
                tmap.x_m, tmap.y_m, pen=pen, antialias=True,
                skipFiniteCheck=True,
                name=f"trackline_{idx}",
            )
            self._plot.addItem(line)
            self._lines[path] = line
            if self._anchor_map is None:
                self._anchor_map = tmap
                anchor_path = path
        if self._anchor_map is None:
            self._dot.hide()
        self._anchor_path = anchor_path

        # Reference racing line overlay (ideal line from precomputed
        # CSV in racing_lines/<TRACK>_racing.csv, if available).
        track = None
        car = None
        if anchor_path is not None:
            lap = self._loaded_laps.get(anchor_path)
            if lap is not None:
                try:
                    track = str(lap.summary.get("track") or "") or None
                    car = str(lap.summary.get("car") or "") or None
                except Exception:  # noqa: BLE001
                    track = None
                    car = None
        self._render_racing_line(track)
        self._render_knw_line(track, car)
        self._render_track_overlay(track)

        # Re-fit the view to the new geometry.
        self._plot.getViewBox().enableAutoRange()
        self._plot.getViewBox().autoRange()

    # ------------------------------------------------------------------
    # Replay transport (play / pause / stop / scrub / ghosts)
    # ------------------------------------------------------------------

    def _anchor_duration_s(self) -> float:
        """Lap duration of the anchor lap, or 0 if not playable."""
        path = self._anchor_path
        if path is None:
            return 0.0
        td = self._lap_t_d.get(path)
        if td is None:
            return 0.0
        t_s = td[0]
        if t_s.size < 2:
            return 0.0
        return float(t_s[-1] - t_s[0])

    def _set_replay_enabled(self, enabled: bool) -> None:
        for w in (
            self._btn_skip_back, self._btn_slow, self._btn_play,
            self._btn_stop, self._btn_fast, self._btn_skip_fwd,
            self._loop_check, self._scrub_slider,
        ):
            w.setEnabled(enabled)

    def _refresh_replay_ui(self) -> None:
        """Sync the transport bar with the current anchor lap."""
        dur = self._anchor_duration_s()
        playable = dur > 0.0
        self._set_replay_enabled(playable)
        if not playable:
            self._stop_playback(restore_axis=False)
            self._time_label.setText("00:00.000 / 00:00.000")
            self._scrub_slider.blockSignals(True)
            self._scrub_slider.setValue(0)
            self._scrub_slider.blockSignals(False)
            return
        # Clamp current playback time into the new lap's range.
        self._playback_t_s = max(0.0, min(self._playback_t_s, dur))
        self._update_time_label()
        self._sync_slider_from_time()

    def _update_time_label(self) -> None:
        dur = self._anchor_duration_s()

        def fmt(t: float) -> str:
            t = max(0.0, t)
            m = int(t // 60)
            s = t - 60 * m
            return f"{m:02d}:{s:06.3f}"

        self._time_label.setText(
            f"{fmt(self._playback_t_s)} / {fmt(dur)}"
        )

    def _sync_slider_from_time(self) -> None:
        dur = self._anchor_duration_s()
        if dur <= 0:
            val = 0
        else:
            frac = max(0.0, min(1.0, self._playback_t_s / dur))
            val = int(round(frac * self._scrub_slider.maximum()))
        self._scrub_slider.blockSignals(True)
        self._scrub_slider.setValue(val)
        self._scrub_slider.blockSignals(False)

    def _t_to_distance(self, path: Path, t_s: float) -> float | None:
        td = self._lap_t_d.get(path)
        if td is None:
            return None
        ts, ds = td
        if ts.size < 2:
            return None
        t_clamped = float(np.clip(t_s, ts[0], ts[-1]))
        return float(np.interp(t_clamped, ts, ds))

    def _emit_anchor_cursor(self) -> None:
        """Drive the shared cursor signal from current ``_playback_t_s``.

        Reuses the existing ``cursor_moved`` channel so the chart
        crosshairs and the track-map dot move in lockstep — the dock
        already handles the dot when the x-axis is in distance mode.
        """
        if self._anchor_path is None:
            return
        d = self._t_to_distance(self._anchor_path, self._playback_t_s)
        if d is None:
            return
        # Make sure the dot is visible: distance-mode is required by
        # ``_on_cursor_moved``. Switch the global axis kind on first
        # play; restore on stop.
        if self._axis_kind != "distance":
            self._signals.x_axis_changed.emit("distance")
        self._signals.cursor_moved.emit(d)

    def _update_ghost_dots(self) -> None:
        """Place a ghost dot on each non-anchor selected lap.

        Each lap is sampled at the *same* elapsed playback time on its
        own ``(time_ms, current_lap_dist_m)`` mapping, so the dots
        race along the geometry just like a MoTeC overlay replay.
        """
        order = getattr(self, "_selection_order", [])
        if not order or self._anchor_path is None:
            for p, item in list(self._ghost_dots.items()):
                self._plot.removeItem(item)
                self._ghost_dots.pop(p, None)
            return

        for idx, path in enumerate(order):
            if path == self._anchor_path:
                continue
            tmap = self._maps.get(path)
            d = self._t_to_distance(path, self._playback_t_s)
            if tmap is None or d is None or tmap.distance_m.size < 2:
                ghost = self._ghost_dots.pop(path, None)
                if ghost is not None:
                    self._plot.removeItem(ghost)
                continue
            d_arr = tmap.distance_m
            d = float(np.clip(d, d_arr[0], d_arr[-1]))
            i = int(np.searchsorted(d_arr, d))
            if i >= d_arr.size:
                i = d_arr.size - 1
            if i > 0 and (d - d_arr[i - 1]) < (d_arr[i] - d):
                i -= 1
            xm = float(tmap.x_m[i])
            ym = float(tmap.y_m[i])
            ghost = self._ghost_dots.get(path)
            if ghost is None:
                color = trace_color(idx)
                ghost = pg.ScatterPlotItem(
                    size=9,
                    pen=pg.mkPen(color, width=1.0),
                    brush=pg.mkBrush(color),
                )
                ghost.setZValue(19)  # just below the anchor cursor
                self._plot.addItem(ghost, ignoreBounds=True)
                self._ghost_dots[path] = ghost
            ghost.setData([xm], [ym])
            ghost.show()

        # Drop ghosts for laps no longer selected.
        keep = set(order)
        for p, item in list(self._ghost_dots.items()):
            if p not in keep or p == self._anchor_path:
                self._plot.removeItem(item)
                self._ghost_dots.pop(p, None)

    def _hide_ghost_dots(self) -> None:
        for p, item in list(self._ghost_dots.items()):
            self._plot.removeItem(item)
            self._ghost_dots.pop(p, None)

    def _start_playback(self) -> None:
        if self._anchor_duration_s() <= 0:
            return
        if self._playback_axis_kind_before is None:
            self._playback_axis_kind_before = self._axis_kind
        self._playback_timer.start()
        self._btn_play.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause),
        )
        self._btn_play.setToolTip("Pause animation")
        # Push the cursor immediately so the user sees feedback even
        # before the first timer tick fires.
        self._emit_anchor_cursor()
        self._update_ghost_dots()

    def _pause_playback(self) -> None:
        self._playback_timer.stop()
        self._btn_play.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
        )
        self._btn_play.setToolTip("Play animation along the lap")

    def _stop_playback(self, *, restore_axis: bool = True) -> None:
        self._playback_timer.stop()
        self._btn_play.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
        )
        self._btn_play.setToolTip("Play animation along the lap")
        self._playback_t_s = 0.0
        self._update_time_label()
        self._sync_slider_from_time()
        self._hide_ghost_dots()
        self._signals.cursor_left.emit()
        if (
            restore_axis
            and self._playback_axis_kind_before is not None
            and self._playback_axis_kind_before != self._axis_kind
        ):
            self._signals.x_axis_changed.emit(
                self._playback_axis_kind_before,
            )
        self._playback_axis_kind_before = None

    def _is_playing(self) -> bool:
        return self._playback_timer.isActive()

    def _on_play_pause(self) -> None:
        if self._is_playing():
            self._pause_playback()
        else:
            # If we're at (or beyond) the end, rewind so Play does the
            # natural thing instead of refusing to start.
            if self._playback_t_s >= self._anchor_duration_s() - 1e-3:
                self._playback_t_s = 0.0
                self._update_time_label()
                self._sync_slider_from_time()
            self._start_playback()

    def _on_stop(self) -> None:
        self._stop_playback(restore_axis=True)

    def _on_skip_back(self) -> None:
        was_playing = self._is_playing()
        self._pause_playback()
        self._playback_t_s = 0.0
        self._update_time_label()
        self._sync_slider_from_time()
        self._emit_anchor_cursor()
        self._update_ghost_dots()
        # Skip-back parks at t=0 paused, like a video player rewind.
        # Honour user expectation: don't auto-resume.
        del was_playing

    def _on_skip_forward(self) -> None:
        dur = self._anchor_duration_s()
        if dur <= 0:
            return
        self._pause_playback()
        self._playback_t_s = dur
        self._update_time_label()
        self._sync_slider_from_time()
        self._emit_anchor_cursor()
        self._update_ghost_dots()

    def _set_speed_index(self, idx: int) -> None:
        idx = max(0, min(len(self._playback_speeds) - 1, idx))
        self._playback_speed_idx = idx
        self._speed_label.setText(
            f"{self._playback_speeds[idx]:g}×"
        )

    def _on_slower(self) -> None:
        self._set_speed_index(self._playback_speed_idx - 1)

    def _on_faster(self) -> None:
        self._set_speed_index(self._playback_speed_idx + 1)

    def _on_loop_toggled(self, checked: bool) -> None:
        self._playback_loop = bool(checked)

    def _on_scrub_pressed(self) -> None:
        self._scrub_was_playing = self._is_playing()
        if self._scrub_was_playing:
            self._pause_playback()

    def _on_scrub_released(self) -> None:
        if self._scrub_was_playing:
            self._start_playback()
        self._scrub_was_playing = False

    def _on_scrub_changed(self, value: int) -> None:
        dur = self._anchor_duration_s()
        if dur <= 0:
            return
        frac = float(value) / float(self._scrub_slider.maximum() or 1)
        self._playback_t_s = max(0.0, min(dur, frac * dur))
        self._update_time_label()
        self._emit_anchor_cursor()
        self._update_ghost_dots()

    def _on_playback_tick(self) -> None:
        dur = self._anchor_duration_s()
        if dur <= 0:
            self._stop_playback(restore_axis=True)
            return
        dt = self._playback_timer.interval() / 1000.0
        speed = self._playback_speeds[self._playback_speed_idx]
        self._playback_t_s += dt * speed
        if self._playback_t_s >= dur:
            if self._playback_loop:
                # Wrap modulo lap duration so high speeds don't skip a
                # whole lap when the overshoot exceeds dt × 1.
                self._playback_t_s = self._playback_t_s % dur
            else:
                self._playback_t_s = dur
                self._emit_anchor_cursor()
                self._update_ghost_dots()
                self._update_time_label()
                self._sync_slider_from_time()
                self._pause_playback()
                return
        self._emit_anchor_cursor()
        self._update_ghost_dots()
        self._update_time_label()
        self._sync_slider_from_time()

    # ------------------------------------------------------------------
    # Racing line overlay
    # ------------------------------------------------------------------

    @staticmethod
    def _racing_line_dirs() -> list[Path]:
        """Search dirs for ``racing_lines/<TRACK>_racing.csv``."""
        out: list[Path] = [Path.cwd() / "racing_lines"]
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            out.append(Path(meipass) / "racing_lines")
        out.append(Path(sys.argv[0]).resolve().parent / "racing_lines")
        return out

    def _load_racing_line(
        self, track: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return (x_line_m, y_line_m, v_target_kmh) or None."""
        if track in self._rline_cache:
            return self._rline_cache[track]
        path: Path | None = None
        for d in self._racing_line_dirs():
            cand = d / f"{track}_racing.csv"
            if cand.exists():
                path = cand
                break
        if path is None:
            self._rline_cache[track] = None  # type: ignore[assignment]
            return None
        xs: list[float] = []
        ys: list[float] = []
        vs: list[float] = []
        try:
            with path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        xs.append(float(row["x_line_m"]))
                        ys.append(float(row["y_line_m"]))
                    except (KeyError, TypeError, ValueError):
                        continue
                    try:
                        vs.append(float(row.get("v_target_kmh") or "nan"))
                    except (TypeError, ValueError):
                        vs.append(float("nan"))
        except OSError:
            self._rline_cache[track] = None  # type: ignore[assignment]
            return None
        if not xs:
            self._rline_cache[track] = None  # type: ignore[assignment]
            return None
        arr = (
            np.asarray(xs, dtype=float),
            np.asarray(ys, dtype=float),
            np.asarray(vs, dtype=float),
        )
        self._rline_cache[track] = arr
        return arr

    def _render_racing_line(self, track: str | None) -> None:
        # Clear previous overlay.
        if self._rline_item is not None:
            self._plot.removeItem(self._rline_item)
            self._rline_item = None
        if self._apex_item is not None:
            self._plot.removeItem(self._apex_item)
            self._apex_item = None
        self._current_track = track
        if not track:
            return
        data = self._load_racing_line(track)
        if data is None:
            return
        x_line, y_line, v_target = data
        # Dashed ideal-line polyline, low z so lap traces stay on top.
        pen = QPen(pg.mkColor("#ffffff"))
        pen.setWidthF(1.0)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._rline_item = pg.PlotDataItem(
            x_line, y_line, pen=pen, antialias=True,
            skipFiniteCheck=True, name="racing_line",
        )
        self._rline_item.setOpacity(0.55)
        self._rline_item.setZValue(-1)
        self._plot.addItem(self._rline_item)
        # Apex markers = slowest decile of the target-speed signal.
        v_clean = v_target[np.isfinite(v_target)]
        if v_clean.size >= 10:
            thr = float(np.percentile(v_clean, 10.0))
            mask = np.isfinite(v_target) & (v_target <= thr)
            if mask.any():
                self._apex_item = pg.ScatterPlotItem(
                    x=x_line[mask], y=y_line[mask],
                    size=6,
                    brush=pg.mkBrush("#ff5d6c"),
                    pen=pg.mkPen("#202830", width=0.8),
                )
                self._apex_item.setZValue(-1)
                self._apex_item.setOpacity(0.7)
                self._plot.addItem(self._apex_item)

    # ------------------------------------------------------------------
    # Per-car KNW AI line overlay
    # ------------------------------------------------------------------

    @staticmethod
    def _pth_search_paths(track: str) -> list[Path]:
        out: list[Path] = []
        # Standard LFS install location.
        out.append(Path(r"C:\LFS\data\smx") / f"{track}.pth")
        # Workspace-bundled assets used by tests / dev.
        cwd = Path.cwd()
        for sub in (
            "assets/source",
            "assets",
            "tracks",
        ):
            out.append(cwd / sub / f"{track}.pth")
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            out.append(Path(meipass) / "tracks" / f"{track}.pth")
        return out

    @staticmethod
    def _knw_search_paths(track: str, car: str) -> list[Path]:
        out: list[Path] = []
        stem = f"{track}_{car}.knw"
        out.append(Path(r"C:\LFS\data\knw") / stem)
        cwd = Path.cwd()
        for sub in ("assets/knw", "assets", "tracks"):
            out.append(cwd / sub / stem)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            out.append(Path(meipass) / "knw" / stem)
        return out

    def _load_knw_line(
        self, track: str, car: str,
    ) -> np.ndarray | None:
        """Return shape (N, 2) x/y of the KNW AI line, or None."""
        key = (track.upper(), car.upper())
        if key in self._knw_cache:
            return self._knw_cache[key]
        pth_path: Path | None = next(
            (p for p in self._pth_search_paths(key[0]) if p.exists()),
            None,
        )
        knw_path: Path | None = next(
            (p for p in self._knw_search_paths(*key) if p.exists()),
            None,
        )
        if pth_path is None or knw_path is None:
            self._knw_cache[key] = None
            return None
        try:
            from ...telemetry.track.knw import parse_knw
            from ...telemetry.track.pth import compute_profile
            from ...telemetry.track.racing_line import compute_knw_line
            profile = compute_profile(pth_path)
            knw = parse_knw(knw_path)
            line = compute_knw_line(profile, knw)
            xy = np.asarray(line.line_xy, dtype=float)
        except Exception:  # noqa: BLE001
            self._knw_cache[key] = None
            return None
        if xy.ndim != 2 or xy.shape[0] < 2:
            self._knw_cache[key] = None
            return None
        self._knw_cache[key] = xy
        return xy

    def _render_knw_line(
        self, track: str | None, car: str | None,
    ) -> None:
        if self._knw_item is not None:
            self._plot.removeItem(self._knw_item)
            self._knw_item = None
        self._current_car = car
        if not track or not car:
            return
        xy = self._load_knw_line(track, car)
        if xy is None:
            return
        # Dotted golden line just above the FBM reference line.
        pen = QPen(pg.mkColor("#ffd166"))
        pen.setWidthF(1.2)
        pen.setStyle(Qt.PenStyle.DotLine)
        pen.setCosmetic(True)
        self._knw_item = pg.PlotDataItem(
            xy[:, 0], xy[:, 1], pen=pen, antialias=True,
            skipFiniteCheck=True, name=f"knw_{car}",
        )
        self._knw_item.setOpacity(0.85)
        self._knw_item.setZValue(-0.5)
        self._plot.addItem(self._knw_item)

    # ------------------------------------------------------------------
    # Per-environment top-down image overlay (LFS official .tif)
    # ------------------------------------------------------------------

    @staticmethod
    def _qimage_to_array(qimg: QImage) -> np.ndarray | None:
        """Convert a QImage to an (H, W, 4) RGBA uint8 ndarray."""
        if qimg.isNull():
            return None
        rgba = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
        w = rgba.width()
        h = rgba.height()
        if w <= 0 or h <= 0:
            return None
        bpl = rgba.bytesPerLine()
        ptr = rgba.constBits()
        if ptr is None:
            return None
        buf = bytes(ptr)
        arr = np.frombuffer(buf, dtype=np.uint8)
        try:
            arr = arr.reshape((h, bpl // 4, 4))[:, :w, :]
        except ValueError:
            return None
        return np.ascontiguousarray(arr)

    def _load_overlay_image(self, env: str) -> np.ndarray | None:
        path = find_overlay_image(env)
        if path is None:
            return None
        cached = self._overlay_image_cache.get(path)
        if cached is not None:
            return cached
        qimg = QImage(str(path))
        arr = self._qimage_to_array(qimg)
        # Only cache successful decodes; otherwise a transient failure
        # would poison the cache and the overlay would never recover
        # for the rest of the session.
        if arr is not None:
            self._overlay_image_cache[path] = arr
        return arr

    def _calibration_for(self, env: str) -> OverlayCalibration:
        return self._overlay_calibrations.get(env, DEFAULT_CALIBRATION)

    def _overlay_extent_for(
        self, env: str, image: np.ndarray | None = None,
    ) -> OverlayExtent | None:
        if env in self._overlay_extent_cache:
            return self._overlay_extent_cache[env]
        cal = self._calibration_for(env)
        ext: OverlayExtent | None
        if image is not None and image.ndim >= 2:
            # Canonical placement: LFS renders every track image at
            # exactly 1 m/px centred on the world origin (0, 0). So we
            # derive the world rectangle from image dimensions alone.
            h_px, w_px = int(image.shape[0]), int(image.shape[1])
            ext = compute_overlay_extent_for_image((w_px, h_px), cal)
        else:
            # Fall back to the legacy bbox auto-fit when we don't yet
            # have the image (e.g. preview without bundled TIF).
            ext = compute_overlay_extent(env, cal)
        self._overlay_extent_cache[env] = ext
        return ext

    def _render_track_overlay(self, track: str | None) -> None:
        if self._overlay_item is not None:
            self._plot.removeItem(self._overlay_item)
            self._overlay_item = None
        env = track_to_environment(track or "")
        self._current_env = env
        if not env or not self._overlay_visible:
            return
        img = self._load_overlay_image(env)
        if img is None:
            return
        extent = self._overlay_extent_for(env, image=img)
        if extent is None:
            return
        if extent.flip_y:
            img_to_show = np.ascontiguousarray(img[::-1, :, :])
        else:
            img_to_show = img
        # ``autoDownsample=True`` is required for the 2560×2560 LFS
        # track images: without it pyqtgraph keeps a full-resolution
        # QPixmap cache that silently drops at large zoom factors,
        # which made the overlay vanish after the user panned/zoomed
        # and then changed laps.
        item = pg.ImageItem(
            img_to_show, axisOrder="row-major", autoDownsample=True,
        )
        item.setRect(QRectF(
            extent.x0_m, extent.y0_m, extent.width_m, extent.height_m,
        ))
        item.setOpacity(self._overlay_opacity)
        # Sit underneath every other item (racing line is at z=-1).
        item.setZValue(-10)
        # Keep the image out of pyqtgraph's autoRange computation so a
        # previously-zoomed ViewBox doesn't try to re-fit a 2.5 km box
        # and end up clipping the overlay off-screen.
        self._plot.addItem(item, ignoreBounds=True)
        self._overlay_item = item

    def _on_overlay_toggled(self, checked: bool) -> None:
        self._overlay_visible = bool(checked)
        if not self._overlay_visible and self._overlay_item is not None:
            self._plot.removeItem(self._overlay_item)
            self._overlay_item = None
        elif self._overlay_visible and self._overlay_item is None:
            self._render_track_overlay(self._current_env)

    def _on_overlay_opacity(self, value: int) -> None:
        self._overlay_opacity = max(0.0, min(1.0, float(value) / 100.0))
        if self._overlay_item is not None:
            self._overlay_item.setOpacity(self._overlay_opacity)

    # ------------------------------------------------------------------
    # Interactive overlay calibration
    # ------------------------------------------------------------------

    def _apply_calibration_preview(
        self, env: str, cal: OverlayCalibration,
    ) -> None:
        """Swap in *cal* for *env* and re-render the overlay."""
        self._overlay_calibrations[env] = cal
        self._overlay_extent_cache.pop(env, None)
        if self._overlay_visible:
            self._render_track_overlay(self._current_env)

    def _on_calibrate_overlay(self) -> None:
        env = self._current_env
        if not env:
            QMessageBox.information(
                self,
                "Calibrate map",
                "Load a lap first so the map dock knows which environment"
                " (BL, AS, KY, …) you want to calibrate.",
            )
            return
        if find_overlay_image(env) is None:
            QMessageBox.information(
                self,
                "Calibrate map",
                f"No track image is bundled for environment '{env}'.",
            )
            return
        original = self._calibration_for(env)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Calibrate map · {env}")
        form = QFormLayout()

        def _spin(
            value: float, lo: float, hi: float, step: float, decimals: int,
        ) -> QDoubleSpinBox:
            sp = QDoubleSpinBox(dlg)
            sp.setRange(lo, hi)
            sp.setDecimals(decimals)
            sp.setSingleStep(step)
            sp.setValue(value)
            return sp

        dx_spin = _spin(original.dx_m, -5000.0, 5000.0, 1.0, 2)
        dy_spin = _spin(original.dy_m, -5000.0, 5000.0, 1.0, 2)
        scale_spin = _spin(original.scale, 0.5, 2.0, 0.001, 4)
        flip_chk = QCheckBox("Flip image vertically", dlg)
        flip_chk.setChecked(bool(original.flip_y))

        form.addRow("dx (m, +east)", dx_spin)
        form.addRow("dy (m, +north)", dy_spin)
        form.addRow("scale ×", scale_spin)
        form.addRow("", flip_chk)

        hint = QLabel(
            "Track images are auto-aligned at 1 m/px centred on the"
            " world origin. These spinners apply optional residual"
            " nudges if a particular environment looks off. Save"
            " writes them to your user profile.",
            dlg,
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {MUTED_COLOR};")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Reset,
            parent=dlg,
        )

        layout = QVBoxLayout(dlg)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(buttons)

        def _current_cal() -> OverlayCalibration:
            return replace(
                original,
                dx_m=float(dx_spin.value()),
                dy_m=float(dy_spin.value()),
                scale=float(scale_spin.value()),
                flip_y=bool(flip_chk.isChecked()),
            )

        def _on_change(_=None) -> None:
            self._apply_calibration_preview(env, _current_cal())

        for sp in (dx_spin, dy_spin, scale_spin):
            sp.valueChanged.connect(_on_change)
        flip_chk.toggled.connect(_on_change)

        def _on_reset() -> None:
            dx_spin.setValue(DEFAULT_CALIBRATION.dx_m)
            dy_spin.setValue(DEFAULT_CALIBRATION.dy_m)
            scale_spin.setValue(DEFAULT_CALIBRATION.scale)
            flip_chk.setChecked(DEFAULT_CALIBRATION.flip_y)

        reset_btn = buttons.button(QDialogButtonBox.StandardButton.Reset)
        if reset_btn is not None:
            reset_btn.clicked.connect(_on_reset)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        result = dlg.exec()
        if result == QDialog.DialogCode.Accepted:
            cal = _current_cal()
            self._apply_calibration_preview(env, cal)
            try:
                path = save_user_overlay_calibration(env, cal)
            except OSError as exc:
                QMessageBox.warning(
                    self,
                    "Calibrate map",
                    f"Could not save calibration: {exc}",
                )
                return
            QMessageBox.information(
                self,
                "Calibrate map",
                f"Saved calibration for {env} to:\n{path}",
            )
        else:
            # Roll back the live preview.
            self._apply_calibration_preview(env, original)


__all__ = ["TrackMapDock"]
