"""Track-elevation dock: side-on Z(s) view with synchronized cursor.

This dock is the 3D counterpart of :class:`TrackMapDock`. Where the
top-down view answers "where on the layout am I?", the elevation view
answers "how high am I and what does the terrain around me look like?".

For every selected lap we draw the racing-line altitude profile
``z(distance)`` directly from the lap's OutSim ``pos_z`` channel.

If a matching SMX mesh ships with the workspace (or is found in
``C:/LFS/data/smx``) we additionally project the mesh vertices onto the
lap's XY trajectory and overlay a grey fill band = silhouette envelope
of the terrain within ``half_width_m`` of the racing line. This shows
the actual 3D shape of the track corridor (kerbs, banking, hills) — not
just a 2D top-down outline.

Signals consumed:

* ``laps_selected``  → rebuild the elevation traces.
* ``cursor_moved``   → move the cursor dot at that distance.
* ``cursor_left``    → hide the cursor dot.
* ``x_axis_changed`` → the cursor dot only follows in ``distance`` mode.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ...telemetry import LapTelemetry
from ...telemetry.comparison import _unwrapped_lap_arrays
from ...telemetry.track.smx import (
    elevation_envelope,
    find_smx_for_track,
    parse_smx,
)
from ...telemetry.track import geom3d
from ..models import LapLoader
from ..signals import SignalBus
from ..theme import CURSOR_COLOR, MUTED_COLOR, TEXT_COLOR, trace_color

# RGBA palette for the surface-class strip drawn under the elevation
# traces. Order matches :data:`geom3d.SURFACE_CLASSES`. ``other`` is
# rendered fully transparent so non-track stations leave the strip blank.
_SURFACE_RGBA: tuple[tuple[int, int, int, int], ...] = (
    (70, 70, 70, 220),     # asphalt — dark grey
    (210, 70, 70, 220),    # kerb    — red
    (210, 175, 110, 220),  # runoff  — sand
    (80, 145, 80, 220),    # grass   — green
    (0, 0, 0, 0),          # other   — transparent
)


def _elevation_arrays(lap: LapTelemetry) -> tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
    """Return ``(distance_m, pos_z_m, pos_x_m, pos_y_m)`` for one lap.

    Uses :func:`_unwrapped_lap_arrays` so the distance axis matches every
    other dock (charts, track map). Empty arrays are returned if the
    capture is missing ``pos_z``.
    """
    idx, d, _ = _unwrapped_lap_arrays(lap)
    df = lap.raw
    if "pos_z" not in df.columns or idx.size == 0:
        return (np.empty(0), np.empty(0), np.empty(0), np.empty(0))
    z = df["pos_z"].to_numpy()[idx].astype(np.float64)
    x = (df["pos_x"].to_numpy()[idx].astype(np.float64)
         if "pos_x" in df.columns else np.full(idx.size, np.nan))
    y = (df["pos_y"].to_numpy()[idx].astype(np.float64)
         if "pos_y" in df.columns else np.full(idx.size, np.nan))
    return d, z, x, y


class TrackElevationDock(QWidget):
    """Side-elevation view: arc length on X, altitude (m) on Y."""

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

        # Per-lap arrays keyed by capture path.
        self._loaded: Dict[Path, LapTelemetry] = {}
        self._arrays: Dict[Path, tuple[np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray]] = {}
        self._lines: Dict[Path, pg.PlotDataItem] = {}
        self._anchor_path: Path | None = None

        # SMX overlay state (one envelope per file; reused across laps
        # that share the same track id). Cache stores XY, Z and the
        # per-vertex surface classification.
        self._smx_cache: Dict[str, tuple[np.ndarray, np.ndarray,
                                         np.ndarray]] = {}
        self._envelope_item: pg.FillBetweenItem | None = None
        self._env_lo: pg.PlotDataItem | None = None
        self._env_hi: pg.PlotDataItem | None = None
        self._strip_item: pg.ImageItem | None = None
        # Banking secondary axis + apex blind-crest markers.
        self._banking_item: pg.PlotDataItem | None = None
        self._banking_zero: pg.InfiniteLine | None = None
        self._blind_apex_item: pg.ScatterPlotItem | None = None
        # BVH barrier scan: narrow-margin markers.
        self._narrow_wall_item: pg.ScatterPlotItem | None = None

        self._plot = pg.PlotWidget(self)
        self._plot.setBackground(None)
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.showGrid(x=True, y=True, alpha=0.12)
        self._plot.getAxis("left").setLabel("Z [m]", color=TEXT_COLOR)
        self._plot.getAxis("bottom").setLabel("distance [m]", color=TEXT_COLOR)
        for ax in ("left", "bottom"):
            axis = self._plot.getAxis(ax)
            axis.setTextPen(pg.mkPen(MUTED_COLOR))

        # Secondary right axis for banking (degrees), with its own
        # ViewBox X-linked to the main plot.
        self._banking_vb = pg.ViewBox()
        self._banking_vb.setMouseEnabled(x=False, y=False)
        self._plot.scene().addItem(self._banking_vb)
        self._banking_axis = pg.AxisItem("right")
        self._banking_axis.setLabel("banking [deg]", color=MUTED_COLOR)
        self._banking_axis.setTextPen(pg.mkPen(MUTED_COLOR))
        self._plot.plotItem.layout.addItem(self._banking_axis, 2, 3)
        self._banking_axis.linkToView(self._banking_vb)
        self._banking_vb.setXLink(self._plot.plotItem.vb)
        self._plot.plotItem.vb.sigResized.connect(self._sync_banking_vb)
        self._sync_banking_vb()

        # Cursor dot.
        self._dot = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen(CURSOR_COLOR, width=1.5),
            brush=pg.mkBrush(CURSOR_COLOR),
        )
        self._dot.setZValue(20)
        self._dot.hide()
        self._plot.addItem(self._dot)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._plot)

        signals.laps_selected.connect(self._on_laps_selected)
        signals.cursor_moved.connect(self._on_cursor_moved)
        signals.cursor_left.connect(self._on_cursor_left)
        signals.x_axis_changed.connect(self._on_axis_changed)
        loader.lap_loaded.connect(self._on_lap_loaded)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_lap_loaded(self, path: Path, lap: LapTelemetry) -> None:
        path = Path(path)
        if path not in getattr(self, "_selection_order", []):
            return
        self._loaded[path] = lap
        self._arrays[path] = _elevation_arrays(lap)
        self._redraw()

    def _on_laps_selected(self, paths: List[Path]) -> None:
        wanted = {Path(p) for p in paths}
        for p in list(self._arrays):
            if p not in wanted:
                self._arrays.pop(p, None)
                self._loaded.pop(p, None)
        self._selection_order = [Path(p) for p in paths]
        self._redraw()

    def _on_cursor_moved(self, x: float) -> None:
        if (self._anchor_path is None
                or self._axis_kind != "distance"):
            self._dot.hide()
            return
        arr = self._arrays.get(self._anchor_path)
        if arr is None:
            return
        d, z, *_ = arr
        if d.size < 2:
            return
        x = float(np.clip(x, d[0], d[-1]))
        i = int(np.searchsorted(d, x))
        if i >= d.size:
            i = d.size - 1
        if i > 0 and (x - d[i - 1]) < (d[i] - x):
            i -= 1
        self._dot.setData([float(d[i])], [float(z[i])])
        self._dot.show()

    def _on_cursor_left(self) -> None:
        self._dot.hide()

    def _on_axis_changed(self, kind: str) -> None:
        self._axis_kind = kind
        if kind != "distance":
            self._dot.hide()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        for line in self._lines.values():
            self._plot.removeItem(line)
        self._lines.clear()
        self._clear_envelope()

        order = getattr(self, "_selection_order", list(self._arrays))
        self._anchor_path = None
        for path in order:
            arr = self._arrays.get(path)
            if arr is None:
                continue
            d, z, _x, _y = arr
            if d.size < 2 or not np.isfinite(z).any():
                continue
            idx = order.index(path)
            color = trace_color(idx)
            pen = QPen(pg.mkColor(color))
            pen.setWidthF(1.6)
            pen.setCosmetic(True)
            line = pg.PlotDataItem(d, z, pen=pen, antialias=True,
                                   skipFiniteCheck=True,
                                   name=f"elev_{idx}")
            self._plot.addItem(line)
            self._lines[path] = line
            if self._anchor_path is None:
                self._anchor_path = path

        # SMX silhouette overlay (anchor lap only — keeps the band readable).
        if self._anchor_path is not None:
            self._draw_envelope(self._anchor_path)

        if self._anchor_path is None:
            self._dot.hide()

        vb = self._plot.getViewBox()
        vb.enableAutoRange()
        vb.autoRange()

    # ------------------------------------------------------------------
    # SMX silhouette
    # ------------------------------------------------------------------

    def _draw_envelope(self, anchor: Path) -> None:
        arr = self._arrays.get(anchor)
        lap = self._loaded.get(anchor)
        if arr is None or lap is None:
            return
        d, _z, x, y = arr
        if d.size < 4 or not (np.isfinite(x).any() and np.isfinite(y).any()):
            return
        track = lap.summary.get("track") if lap.summary else None
        if not track:
            return

        env = self._smx_envelope_for_track(str(track), d, x, y)
        if env is None:
            return
        (d_grid, z_lo, z_hi, classes_pred, z_grid, banking, apex_vis,
         barrier_left, barrier_right) = env

        # FillBetweenItem requires two curves of identical X.
        terrain_pen = pg.mkPen(QColor(150, 150, 150, 160), width=1.0,
                               style=pg.QtCore.Qt.PenStyle.DotLine)
        self._env_lo = pg.PlotDataItem(d_grid, z_lo, pen=terrain_pen,
                                       connect="finite")
        self._env_hi = pg.PlotDataItem(d_grid, z_hi, pen=terrain_pen,
                                       connect="finite")
        self._plot.addItem(self._env_lo)
        self._plot.addItem(self._env_hi)
        brush = pg.mkBrush(QColor(140, 140, 140, 60))
        self._envelope_item = pg.FillBetweenItem(
            self._env_lo, self._env_hi, brush=brush)
        self._envelope_item.setZValue(-10)
        self._plot.addItem(self._envelope_item)

        if classes_pred is not None and classes_pred.size:
            self._draw_surface_strip(d_grid, z_lo, z_hi, classes_pred)

        # Banking trace on the secondary right axis (degrees).
        if banking is not None and np.isfinite(banking).any():
            banking_deg = np.rad2deg(banking)
            pen = pg.mkPen(QColor(120, 200, 255, 220), width=1.4)
            self._banking_item = pg.PlotDataItem(
                d_grid, banking_deg, pen=pen, antialias=True,
                connect="finite", name="banking_deg",
            )
            self._banking_vb.addItem(self._banking_item)
            self._banking_zero = pg.InfiniteLine(
                pos=0.0, angle=0,
                pen=pg.mkPen(QColor(120, 200, 255, 60),
                             style=pg.QtCore.Qt.PenStyle.DashLine),
            )
            self._banking_vb.addItem(self._banking_zero)
            finite = banking_deg[np.isfinite(banking_deg)]
            if finite.size:
                bmax = max(float(np.nanmax(np.abs(finite))), 2.0)
                self._banking_vb.setYRange(-bmax * 1.2, bmax * 1.2,
                                           padding=0)

        # Apex visibility: highlight blind crests as orange markers
        # on the anchor elevation curve.
        if apex_vis is not None and apex_vis.size == z_grid.size:
            thr = 60.0  # m of line of sight
            mask = np.isfinite(z_grid) & (apex_vis < thr) & (apex_vis > 0.0)
            if mask.any():
                self._blind_apex_item = pg.ScatterPlotItem(
                    x=d_grid[mask], y=z_grid[mask],
                    size=7, symbol="t",
                    brush=pg.mkBrush("#ffa040"),
                    pen=pg.mkPen("#202830", width=0.8),
                )
                self._blind_apex_item.setZValue(15)
                self._plot.addItem(self._blind_apex_item)

        # BVH barrier scan: flag stations whose nearest wall (= drivable
        # surface edge) on either side is under the narrow-margin
        # threshold. Drawn as red squares above the elevation curve so
        # the user can spot pinch points without leaving the plot.
        if (barrier_left is not None and barrier_right is not None
                and barrier_left.size == z_grid.size):
            margin = np.minimum(barrier_left, barrier_right)
            narrow_thr = 4.0  # metres of free space on the worst side
            mask = (np.isfinite(z_grid) & np.isfinite(margin)
                    & (margin > 0.0) & (margin < narrow_thr))
            if mask.any():
                self._narrow_wall_item = pg.ScatterPlotItem(
                    x=d_grid[mask], y=z_grid[mask],
                    size=8, symbol="s",
                    brush=pg.mkBrush("#ff5d6c"),
                    pen=pg.mkPen("#202830", width=0.8),
                )
                self._narrow_wall_item.setZValue(15)
                self._plot.addItem(self._narrow_wall_item)

    def _draw_surface_strip(
        self,
        d_grid: np.ndarray,
        z_lo: np.ndarray,
        z_hi: np.ndarray,
        classes_pred: np.ndarray,
    ) -> None:
        """Render a thin RGBA strip of the predominant surface class."""
        finite = np.isfinite(z_lo) | np.isfinite(z_hi)
        if not finite.any():
            return
        z_min = float(np.nanmin(np.where(finite, z_lo, np.nan)))
        z_max = float(np.nanmax(np.where(finite, z_hi, np.nan)))
        if not (np.isfinite(z_min) and np.isfinite(z_max)):
            return
        z_range = max(z_max - z_min, 1.0)
        strip_h = 0.06 * z_range
        y0 = z_min - 0.10 * z_range
        x0 = float(d_grid[0])
        x1 = float(d_grid[-1])
        # Build a (1, N, 4) RGBA row image.
        n = classes_pred.shape[0]
        rgba = np.zeros((1, n, 4), dtype=np.uint8)
        for ci, color in enumerate(_SURFACE_RGBA):
            mask = classes_pred == ci
            if mask.any():
                rgba[0, mask, :] = color
        # Stations with no SMX hit get an explicit transparent value.
        rgba[0, classes_pred < 0, :] = (0, 0, 0, 0)
        img = pg.ImageItem(rgba)
        # ImageItem origin = (0, 0) in image space; map to data coords.
        img.setRect(pg.QtCore.QRectF(x0, y0, x1 - x0, strip_h))
        img.setZValue(-5)
        self._plot.addItem(img)
        self._strip_item = img

    def _clear_envelope(self) -> None:
        for item in (self._envelope_item, self._env_lo, self._env_hi,
                     self._strip_item, self._blind_apex_item,
                     self._narrow_wall_item):
            if item is not None:
                self._plot.removeItem(item)
        if self._banking_item is not None:
            self._banking_vb.removeItem(self._banking_item)
        if self._banking_zero is not None:
            self._banking_vb.removeItem(self._banking_zero)
        self._envelope_item = None
        self._env_lo = None
        self._env_hi = None
        self._strip_item = None
        self._banking_item = None
        self._banking_zero = None
        self._blind_apex_item = None
        self._narrow_wall_item = None

    def _sync_banking_vb(self) -> None:
        """Keep the banking ViewBox aligned with the main plot."""
        vb = self._plot.plotItem.vb
        self._banking_vb.setGeometry(vb.sceneBoundingRect())
        self._banking_vb.linkedViewChanged(
            vb, self._banking_vb.XAxis,
        )

    def _smx_envelope_for_track(
        self,
        track: str,
        distance: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray,
               np.ndarray, np.ndarray,
               np.ndarray, np.ndarray] | None:
        """Best-effort SMX overlay payload for the lap's path.

        Returns ``(d_grid, z_lo, z_hi, classes_pred, z_grid, banking_rad,
        apex_vis_m, barrier_left_m, barrier_right_m)``. ``classes_pred``
        is the per-station argmax over :data:`geom3d.SURFACE_CLASSES`
        (``-1`` for stations with no SMX hit). ``banking_rad`` is the
        transverse asphalt slope from
        :func:`geom3d.compute_banking_profile`, ``apex_vis_m`` from
        :func:`geom3d.apex_visibility_distance`, and
        ``barrier_left/right_m`` from
        :func:`geom3d.compute_barrier_offsets`. Returns ``None`` quietly
        if no SMX file is available for the track id.
        """
        key = track.upper()
        cached = self._smx_cache.get(key)
        if cached is None:
            smx_path = find_smx_for_track(key)
            if smx_path is None:
                self._smx_cache[key] = (
                    np.empty(0), np.empty(0), np.empty(0, dtype=np.uint8),
                )
                return None
            try:
                mesh = parse_smx(smx_path)
            except (OSError, ValueError):
                self._smx_cache[key] = (
                    np.empty(0), np.empty(0), np.empty(0, dtype=np.uint8),
                )
                return None
            # Store mesh XY, Z and per-vertex surface class so envelope
            # + strip can be recomputed quickly for each new lap.
            self._smx_cache[key] = (
                mesh.vertices[:, :2].copy(),
                mesh.vertices[:, 2].copy(),
                geom3d.classify_surface(mesh),
            )
            cached = self._smx_cache[key]
        verts_xy, verts_z, classes = cached
        if verts_xy.size == 0:
            return None

        # Down-sample lap XY to a stable station grid (so the SMX
        # envelope length matches the racing-line plot X-axis).
        finite = np.isfinite(distance) & np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 4:
            return None
        d_lap = distance[finite]
        x_lap = x[finite]
        y_lap = y[finite]
        n = min(400, d_lap.size)
        d_grid = np.linspace(d_lap[0], d_lap[-1], n)
        x_grid = np.interp(d_grid, d_lap, x_lap)
        y_grid = np.interp(d_grid, d_lap, y_lap)
        centreline = np.column_stack((x_grid, y_grid))

        # Build a tiny ad-hoc mesh shim so we can reuse elevation_envelope().
        from types import SimpleNamespace
        verts3 = np.column_stack((verts_xy, verts_z))
        shim = SimpleNamespace(vertices=verts3)
        _, z_lo, z_hi = elevation_envelope(
            shim, centreline, d_grid, half_width_m=25.0)
        if not np.isfinite(z_lo).any():
            return None

        # Surface-class strip: per-station predominant class via
        # nearest-centreline lookup over the (already cached) classes.
        classes_pred = _predominant_surface_per_station(
            centreline, verts_xy, classes, half_width_m=15.0,
        )

        # Centreline elevation along the same stations (interpolated
        # from the lap's pos_z curve) for banking + apex visibility.
        finite_z = np.isfinite(distance) & np.isfinite(x) & np.isfinite(y)
        z_lap = np.full_like(d_grid, np.nan)
        if finite_z.any():
            anchor = self._anchor_path
            arr_for_anchor = self._arrays.get(anchor) if anchor else None
            if arr_for_anchor is not None:
                _d, _z, _x, _y = arr_for_anchor
                msk = np.isfinite(_d) & np.isfinite(_z)
                if msk.sum() >= 2:
                    z_lap = np.interp(d_grid, _d[msk], _z[msk])

        # Banking profile via SMX asphalt cross-section fit.
        try:
            from ...telemetry.track.smx import SmxMesh  # type: ignore
            mesh_full = SmxMesh(
                name=key, track_label=key,
                smx_version=0, game_version=0, game_revision=0,
                resolution=1.0,
                ground_rgb=(0, 0, 0),
                vertices=np.column_stack((verts_xy, verts_z)),
                colors=np.zeros((verts_xy.shape[0], 3), dtype=np.uint8),
                triangles=np.empty((0, 3), dtype=np.uint32),
                objects=[], cp_object_indices=np.empty(0, dtype=np.int64),
            )
            centreline_xyz = np.column_stack((centreline, z_lap))
            banking = geom3d.compute_banking_profile(
                centreline_xyz, mesh_full, classes=classes,
                half_width_m=12.0, slice_thickness_m=3.0,
            )
        except Exception:  # noqa: BLE001
            banking = np.full(d_grid.size, np.nan)

        # Apex visibility (line-of-sight in metres).
        try:
            apex_vis = geom3d.apex_visibility_distance(d_grid, z_lap)
        except Exception:  # noqa: BLE001
            apex_vis = np.zeros(d_grid.size)

        # BVH-style barrier scan: distance to first non-drivable
        # surface on each side of every centreline station.
        try:
            tangents = geom3d._centreline_tangents(centreline)
            barrier_left, barrier_right = geom3d.compute_barrier_offsets(
                centreline, tangents, mesh_full, classes=classes,
                max_search_m=40.0,
            )
        except Exception:  # noqa: BLE001
            barrier_left = np.full(d_grid.size, np.nan)
            barrier_right = np.full(d_grid.size, np.nan)

        return (d_grid, z_lo, z_hi, classes_pred, z_lap, banking, apex_vis,
                barrier_left, barrier_right)


def _predominant_surface_per_station(
    centreline_xy: np.ndarray,
    verts_xy: np.ndarray,
    classes: np.ndarray,
    *,
    half_width_m: float,
) -> np.ndarray:
    """Return the most common surface class at each centreline station.

    ``-1`` flags stations with no SMX vertex within ``half_width_m``.
    """
    n = centreline_xy.shape[0]
    out = np.full(n, -1, dtype=np.int16)
    if verts_xy.size == 0 or classes.size == 0:
        return out
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(centreline_xy)
        dist, idx = tree.query(verts_xy, k=1)
    except ImportError:  # pragma: no cover
        d2 = ((verts_xy[:, None, 0] - centreline_xy[None, :, 0]) ** 2
              + (verts_xy[:, None, 1] - centreline_xy[None, :, 1]) ** 2)
        idx = d2.argmin(axis=1)
        dist = np.sqrt(d2[np.arange(verts_xy.shape[0]), idx])
    keep = dist <= half_width_m
    if not keep.any():
        return out
    bin_idx = idx[keep]
    cls = classes[keep].astype(np.int64)
    n_classes = len(geom3d.SURFACE_CLASSES)
    counts = np.zeros((n, n_classes), dtype=np.int64)
    for ci in range(n_classes):
        m = cls == ci
        if m.any():
            counts[:, ci] = np.bincount(bin_idx[m], minlength=n)
    has_any = counts.sum(axis=1) > 0
    out[has_any] = counts[has_any].argmax(axis=1).astype(np.int16)
    return out


__all__ = ["TrackElevationDock"]
