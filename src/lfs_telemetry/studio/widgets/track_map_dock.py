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
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QPen
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ...telemetry import LapTelemetry
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._plot)
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
        self._redraw()

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
        # Track lap order so the first selected becomes the anchor.
        self._selection_order = [Path(p) for p in paths]
        self._redraw()

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

        # Re-fit the view to the new geometry.
        self._plot.getViewBox().enableAutoRange()
        self._plot.getViewBox().autoRange()

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


__all__ = ["TrackMapDock"]
