"""Dampers tab — bump/rebound × low/high-speed histogram per wheel.

For the currently-selected lap (the first selection) this tab plots a
classic damper-velocity histogram per wheel and overlays the four
race-engineering metrics every setup engineer cares about:

* bump average  &  rebound average    (m/s)
* %hi-speed bump  &  %hi-speed rebound

The low-speed boundary defaults to ±25 mm/s, the conventional split
between chassis pitch/roll work and bump/kerb work used by MoTeC, AIM
RaceStudio and Cosworth Pi. The boundary is configurable via the
toolbar so the user can match the convention of their team.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ...telemetry import LapTelemetry
from ...telemetry.damper_histogram import (
    DEFAULT_BIN_WIDTH_MPS,
    DEFAULT_LOW_SPEED_MPS,
    DEFAULT_MAX_ABS_MPS,
    DamperHistogram,
    damper_histogram,
)
from ..models import LapLoader
from ..signals import SignalBus
from ..theme import (
    MUTED_COLOR,
    PANEL_COLOR,
    TEXT_COLOR,
    WHEEL_COLORS as _WHEEL_COLORS,
    WHEEL_GRID_LAYOUT as _GRID_LAYOUT,
)


class _WheelHistogramPlot(QWidget):
    """One wheel's bar histogram + bump/rebound metric overlay."""

    def __init__(self, corner: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._corner = corner
        color = _WHEEL_COLORS[corner]

        self._plot = pg.PlotWidget(self)
        self._plot.setBackground(PANEL_COLOR)
        self._plot.showGrid(x=False, y=True, alpha=0.18)
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setTitle(f"Damper velocity {corner}",
                            color=TEXT_COLOR, size="9pt")
        self._plot.getAxis("left").setLabel("samples", color=MUTED_COLOR)
        self._plot.getAxis("bottom").setLabel("m/s", color=MUTED_COLOR)
        for ax in ("left", "bottom"):
            self._plot.getAxis(ax).setTextPen(MUTED_COLOR)

        self._bars: pg.BarGraphItem | None = None
        self._low_left: pg.InfiniteLine | None = None
        self._low_right: pg.InfiniteLine | None = None
        # Second-lap overlay (compare mode).
        self._compare_curve: pg.PlotDataItem | None = None

        self._summary = QLabel("—", self)
        self._summary.setTextFormat(Qt.TextFormat.RichText)
        self._summary.setStyleSheet(
            f"color:{TEXT_COLOR}; padding:2px 6px;"
        )
        self._color = color

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        layout.addWidget(self._plot, 1)
        layout.addWidget(self._summary)

    def set_histogram(
        self,
        hist: DamperHistogram,
        compare: DamperHistogram | None = None,
    ) -> None:
        # Clear previous items before drawing the new ones.
        if self._bars is not None:
            self._plot.removeItem(self._bars)
            self._bars = None
        for line_attr in ("_low_left", "_low_right"):
            line = getattr(self, line_attr)
            if line is not None:
                self._plot.removeItem(line)
                setattr(self, line_attr, None)
        if self._compare_curve is not None:
            self._plot.removeItem(self._compare_curve)
            self._compare_curve = None

        if hist.bins.size == 0:
            self._summary.setText("<i>no data</i>")
            return

        width = hist.bin_width_mps * 0.9
        # Color: low-speed bins one shade, high-speed bins another.
        base = QColor(self._color)
        light = base.lighter(140).name()
        brushes = []
        for x in hist.bins:
            brushes.append(self._color if abs(x) > hist.low_speed_boundary_mps
                           else light)
        self._bars = pg.BarGraphItem(
            x=hist.bins, height=hist.counts.astype(float),
            width=width, brushes=brushes,
            pen=pg.mkPen(0, 0, 0, 0),
        )
        self._plot.addItem(self._bars)

        # Low-speed boundary markers.
        bnd_pen = pg.mkPen(MUTED_COLOR, width=1, style=Qt.PenStyle.DashLine)
        self._low_left = pg.InfiniteLine(
            pos=-hist.low_speed_boundary_mps, angle=90, pen=bnd_pen,
        )
        self._low_right = pg.InfiniteLine(
            pos=hist.low_speed_boundary_mps, angle=90, pen=bnd_pen,
        )
        self._plot.addItem(self._low_left)
        self._plot.addItem(self._low_right)

        # Tighten ranges so the histogram fills the plot.
        ymax = float(hist.counts.max()) if hist.counts.size else 1.0
        if compare is not None and compare.bins.size:
            ymax = max(ymax, float(compare.counts.max()))
        self._plot.setYRange(0, ymax * 1.10, padding=0)
        if hist.bins.size:
            xrange = float(hist.bins[-1]) + hist.bin_width_mps
            self._plot.setXRange(-xrange, xrange, padding=0)

        # Compare lap overlay as a thin stepped outline.
        if compare is not None and compare.bins.size:
            cpen = pg.mkPen("#ffffff", width=1.5,
                            style=Qt.PenStyle.DashLine)
            # Build a stepped curve aligned with bin centres.
            x_edges = np.concatenate([
                compare.bins - compare.bin_width_mps / 2.0,
                compare.bins[-1:] + compare.bin_width_mps / 2.0,
            ])
            y_step = np.concatenate([compare.counts.astype(float),
                                     compare.counts[-1:].astype(float)])
            self._compare_curve = pg.PlotDataItem(
                x_edges, y_step, pen=cpen,
                stepMode="left", antialias=True,
            )
            self._compare_curve.setZValue(5)
            self._plot.addItem(self._compare_curve)

        # Metrics summary.
        if compare is None:
            self._summary.setText(
                "<span style='color:#cfd6dd'>"
                f"Reb avg <b>{hist.rebound_avg_mps * 1000:.1f}</b> mm/s "
                f"&nbsp;|&nbsp; Hi-reb <b>{hist.rebound_high_pct:.1f}</b>% "
                f"&nbsp;||&nbsp; "
                f"Bump avg <b>{hist.bump_avg_mps * 1000:.1f}</b> mm/s "
                f"&nbsp;|&nbsp; Hi-bump <b>{hist.bump_high_pct:.1f}</b>%"
                "</span>"
            )
        else:
            d_reb = (compare.rebound_avg_mps - hist.rebound_avg_mps) * 1000
            d_bump = (compare.bump_avg_mps - hist.bump_avg_mps) * 1000
            d_hr = compare.rebound_high_pct - hist.rebound_high_pct
            d_hb = compare.bump_high_pct - hist.bump_high_pct
            self._summary.setText(
                "<span style='color:#cfd6dd'>"
                f"A reb <b>{hist.rebound_avg_mps * 1000:.1f}</b>"
                f" / hi <b>{hist.rebound_high_pct:.0f}%</b>"
                f" &nbsp;│&nbsp; B reb <b>"
                f"{compare.rebound_avg_mps * 1000:.1f}</b>"
                f" / hi <b>{compare.rebound_high_pct:.0f}%</b>"
                f" &nbsp;(<i>Δ{d_reb:+.1f} mm/s, Δ{d_hr:+.0f}%</i>)"
                "<br/>"
                f"A bump <b>{hist.bump_avg_mps * 1000:.1f}</b>"
                f" / hi <b>{hist.bump_high_pct:.0f}%</b>"
                f" &nbsp;│&nbsp; B bump <b>"
                f"{compare.bump_avg_mps * 1000:.1f}</b>"
                f" / hi <b>{compare.bump_high_pct:.0f}%</b>"
                f" &nbsp;(<i>Δ{d_bump:+.1f} mm/s, Δ{d_hb:+.0f}%</i>)"
                "</span>"
            )


class DampersTab(QWidget):
    """Per-wheel damper velocity histograms for the first selected lap."""

    def __init__(
        self,
        loader: LapLoader,
        signals: SignalBus,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._loader = loader
        self._signals = signals
        self._requested: List[Path] = []
        self._loaded: Dict[Path, LapTelemetry] = {}
        self._low_speed_mps = DEFAULT_LOW_SPEED_MPS
        self._bin_width_mps = DEFAULT_BIN_WIDTH_MPS
        self._max_abs_mps = DEFAULT_MAX_ABS_MPS

        # Toolbar: low-speed boundary spinbox.
        toolbar = QToolBar(self)
        toolbar.addWidget(QLabel("Low-speed boundary: "))
        self._lo_spin = QDoubleSpinBox(self)
        self._lo_spin.setSuffix(" mm/s")
        self._lo_spin.setDecimals(0)
        self._lo_spin.setRange(5, 100)
        self._lo_spin.setSingleStep(5)
        self._lo_spin.setValue(self._low_speed_mps * 1000.0)
        self._lo_spin.valueChanged.connect(self._on_lo_changed)
        toolbar.addWidget(self._lo_spin)
        toolbar.addSeparator()
        self._caption = QLabel("No lap loaded.", self)
        self._caption.setStyleSheet(f"color:{MUTED_COLOR};")
        toolbar.addWidget(self._caption)

        # Grid of four histogram tiles, layout matches driver's view.
        self._tiles: Dict[str, _WheelHistogramPlot] = {}
        grid_widget = QWidget(self)
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        for corner, row, col in _GRID_LAYOUT:
            tile = _WheelHistogramPlot(corner, grid_widget)
            self._tiles[corner] = tile
            grid.addWidget(tile, row, col)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(toolbar)
        layout.addWidget(grid_widget, 1)

        # Wiring
        signals.laps_selected.connect(self._on_laps_selected)
        loader.lap_loaded.connect(self._on_lap_loaded)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_laps_selected(self, paths: List[Path]) -> None:
        self._requested = list(paths)
        keep = set(paths)
        self._loaded = {
            p: lap for p, lap in self._loaded.items() if p in keep
        }
        if not paths:
            self._caption.setText("No lap loaded.")
            self._clear_tiles()
            return
        # Request the first two laps (A and optional B for compare).
        for path in paths[:2]:
            if path not in self._loaded:
                self._caption.setText(f"Loading {path.name}…")
                self._loader.request(path)
        if paths[0] in self._loaded:
            self._refresh()

    def _on_lap_loaded(self, path: Path, lap: LapTelemetry) -> None:
        if path not in self._requested:
            return
        self._loaded[path] = lap
        # Refresh once the primary (A) lap is loaded; refresh again if B
        # arrives later so its overlay appears.
        if self._requested and self._requested[0] in self._loaded:
            self._refresh()

    def _on_lo_changed(self, mm_per_s: float) -> None:
        self._low_speed_mps = float(mm_per_s) / 1000.0
        self._refresh()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _clear_tiles(self) -> None:
        empty = damper_histogram(
            np.zeros(0), low_speed_mps=self._low_speed_mps,
            bin_width_mps=self._bin_width_mps,
            max_abs_mps=self._max_abs_mps,
        )
        for tile in self._tiles.values():
            tile.set_histogram(empty)

    def _refresh(self) -> None:
        if not self._requested:
            return
        first = self._requested[0]
        lap = self._loaded.get(first)
        if lap is None:
            return
        # Optional compare lap.
        compare_path: Path | None = (
            self._requested[1] if len(self._requested) >= 2 else None
        )
        compare_lap = (
            self._loaded.get(compare_path) if compare_path else None
        )
        df_b = compare_lap.enriched if compare_lap is not None else None
        df = lap.enriched
        missing = []
        for corner in self._tiles:
            col = f"wheel_{corner}_susp_speed_mps"
            if col not in df.columns:
                missing.append(corner)
                continue
            speeds = df[col].to_numpy(dtype=float)
            hist = damper_histogram(
                speeds,
                low_speed_mps=self._low_speed_mps,
                bin_width_mps=self._bin_width_mps,
                max_abs_mps=self._max_abs_mps,
            )
            compare_hist: DamperHistogram | None = None
            if df_b is not None and col in df_b.columns:
                speeds_b = df_b[col].to_numpy(dtype=float)
                compare_hist = damper_histogram(
                    speeds_b,
                    low_speed_mps=self._low_speed_mps,
                    bin_width_mps=self._bin_width_mps,
                    max_abs_mps=self._max_abs_mps,
                )
            self._tiles[corner].set_histogram(hist, compare=compare_hist)
        if missing:
            self._caption.setText(
                f"{first.name} — missing damper data for: "
                f"{', '.join(missing)}"
            )
        else:
            base = (
                f"{first.name} — low-speed ±"
                f"{self._low_speed_mps * 1000:.0f} mm/s"
            )
            if compare_lap is not None and compare_path is not None:
                base += f"  │  compare B: {compare_path.name}"
            self._caption.setText(base)


__all__ = ["DampersTab"]
