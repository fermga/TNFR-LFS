"""Sector Analysis tab — per-sector timing across selected laps.

Mirrors :class:`StintTab` wiring: subscribes to ``laps_selected`` on the
shared :class:`SignalBus` and, as each lap finishes loading, recomputes
the sector decomposition via :func:`telemetry.sectors.lap_sectors`.

Sector boundaries source (in order of preference):

1. InSim per-lap geometric splits, when at least two laps agree on the
   distance offsets (``insim_split_distances_m``).
2. Uniform 3-equal sectors (``n_equal=3``).

The view shows:

* a header line per lap with sector times (best sector in bold) plus
  the theoretical-best sector sum across the stint,
* a grouped bar chart (one bar per sector × lap) with one stable colour
  per sector,
* a thin dashed line per sector marking the best time in that sector
  across the stint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QPen
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ...telemetry import LapTelemetry
from ...telemetry.sectors import insim_split_distances_m, lap_sectors
from ..models import LapLoader
from ..signals import SignalBus
from ..i18n import tr
from ..theme import MUTED_COLOR, PANEL_COLOR, TEXT_COLOR
from ._format import format_finite


# Stable per-sector colours (cycled if more than 6 sectors).
_SECTOR_COLORS = (
    "#4ea3ff", "#ffa040", "#7ed957",
    "#ff5d6c", "#c58bff", "#ffe066",
)


def _fmt_s(v: float) -> str:
    return format_finite(v, digits=3)


class SectorsTab(QWidget):
    """Per-sector timing table + grouped bar chart for selected laps."""

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

        self._summary = QLabel(tr("No laps selected."), self)
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet(
            f"color:{MUTED_COLOR}; padding:4px 6px;"
        )
        self._summary.setTextFormat(Qt.TextFormat.RichText)

        self._gfx = pg.GraphicsLayoutWidget(parent=self)
        self._gfx.setBackground(PANEL_COLOR)
        self._gfx.ci.layout.setSpacing(2)
        self._gfx.ci.layout.setContentsMargins(2, 2, 2, 2)

        self._plot: pg.PlotItem = self._gfx.addPlot(row=0, col=0)
        self._plot.setTitle(tr("Sector times"), color=TEXT_COLOR, size="9pt")
        self._plot.showGrid(x=False, y=True, alpha=0.12)
        self._plot.getAxis("left").setLabel("s", color=TEXT_COLOR)
        self._plot.getAxis("bottom").setLabel(tr("Lap #"), color=TEXT_COLOR)
        self._plot.getAxis("left").setTextPen(TEXT_COLOR)
        self._plot.getAxis("bottom").setTextPen(TEXT_COLOR)
        self._plot.setMinimumHeight(220)

        self._dyn_items: List[pg.GraphicsObject] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self._summary)
        layout.addWidget(self._gfx, 1)

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
            self._reset()
            return
        for p in paths:
            if p not in self._loaded:
                self._loader.request(p)
        self._refresh()

    def _on_lap_loaded(self, path: Path, lap: LapTelemetry) -> None:
        if path not in self._requested:
            return
        self._loaded[path] = lap
        self._refresh()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _clear_plot(self) -> None:
        for it in self._dyn_items:
            try:
                self._plot.removeItem(it)
            except Exception:  # noqa: BLE001
                pass
        self._dyn_items.clear()
        legend = self._plot.legend
        if legend is not None:
            try:
                legend.scene().removeItem(legend)
            except Exception:  # noqa: BLE001
                pass
            self._plot.legend = None

    def _reset(self) -> None:
        self._summary.setText(tr("No laps selected."))
        self._clear_plot()

    def _shared_boundaries(
        self, laps: List[LapTelemetry]
    ) -> list[float] | None:
        """Return median of per-lap InSim split distances if available."""
        per_lap = [insim_split_distances_m(lap) for lap in laps]
        per_lap = [p for p in per_lap if p]
        if len(per_lap) < max(1, len(laps) // 2 + 1):
            return None
        # All laps must agree on the number of splits.
        n = len(per_lap[0])
        if any(len(p) != n for p in per_lap):
            return None
        arr = np.array(per_lap, dtype=float)
        return [float(x) for x in np.median(arr, axis=0)]

    def _refresh(self) -> None:
        ordered = [
            self._loaded[p] for p in self._requested if p in self._loaded
        ]
        if not ordered:
            self._summary.setText(
                tr("Loading {n} lap(s)\u2026").format(
                    n=len(self._requested),
                ),
            )
            return

        boundaries = self._shared_boundaries(ordered)
        per_lap_sectors: list[list] = []
        lap_indices: list[int] = []
        for lap in ordered:
            try:
                secs = lap_sectors(
                    lap,
                    boundaries_m=boundaries,
                    n_equal=3,
                )
            except Exception:  # noqa: BLE001
                secs = []
            if not secs:
                continue
            per_lap_sectors.append(secs)
            lap_indices.append(int(lap.summary.get("lap_index", 0) or 0))

        if not per_lap_sectors:
            self._summary.setText(
                tr("Sectors unavailable (no usable distance/time data)."),
            )
            self._clear_plot()
            return

        # Normalise sector count across laps (drop laps with mismatching
        # sector counts — usually means a partial lap snuck in).
        n_secs = min(len(s) for s in per_lap_sectors)
        per_lap_sectors = [s[:n_secs] for s in per_lap_sectors]

        # Build matrix: rows=laps, cols=sectors.
        times = np.array(
            [[s.time_s for s in row] for row in per_lap_sectors],
            dtype=float,
        )
        # Best per sector across laps.
        best_per_sec = np.nanmin(times, axis=0)
        theo_best = float(np.nansum(best_per_sec))

        src = (
            tr("InSim splits") if boundaries
            else tr("uniform \u00d7{n}").format(n=n_secs)
        )

        # Header.
        head_lines: list[str] = []
        head_lines.append(
            tr(
                "<b>{n}</b> lap(s) \u00b7 <b>{secs}</b> sectors ({src}) "
                "\u00b7 theoretical best <b>{best} s</b>",
            ).format(
                n=len(per_lap_sectors),
                secs=n_secs,
                src=src,
                best=_fmt_s(theo_best),
            )
        )
        for li, (lap_idx, secs) in enumerate(
            zip(lap_indices, per_lap_sectors)
        ):
            parts = [f"L{lap_idx}"]
            total = 0.0
            for si, sec in enumerate(secs):
                is_best = np.isclose(sec.time_s, best_per_sec[si])
                txt = _fmt_s(sec.time_s)
                if is_best:
                    parts.append(
                        f"S{si + 1}=<b style='color:#7ed957'>{txt}</b>"
                    )
                else:
                    parts.append(f"S{si + 1}={txt}")
                if np.isfinite(sec.time_s):
                    total += sec.time_s
            parts.append(f"Σ={_fmt_s(total)}")
            head_lines.append("  ·  ".join(parts))
        self._summary.setText("<br/>".join(head_lines))

        # Plot — grouped bars: each lap is a cluster of n_secs bars.
        self._clear_plot()
        self._plot.addLegend(offset=(8, 4), labelTextColor=TEXT_COLOR)
        n_laps = len(per_lap_sectors)
        group_w = 0.8
        bar_w = group_w / n_secs
        x_lap = np.arange(n_laps, dtype=float)
        for si in range(n_secs):
            colour = _SECTOR_COLORS[si % len(_SECTOR_COLORS)]
            offsets = (si - (n_secs - 1) / 2.0) * bar_w
            y_sec = times[:, si]
            bars = pg.BarGraphItem(
                x=x_lap + offsets,
                height=y_sec,
                width=bar_w * 0.9,
                brush=pg.mkBrush(colour),
                pen=pg.mkPen(color="#202830", width=1),
            )
            self._plot.addItem(bars)
            self._dyn_items.append(bars)
            # Legend proxy (BarGraphItem isn't picked up automatically).
            sample = pg.PlotDataItem(pen=pg.mkPen(colour, width=6))
            self._plot.legend.addItem(sample, f"S{si + 1}")
            # Best-in-sector dashed reference line.
            best = float(best_per_sec[si])
            if np.isfinite(best):
                pen = QPen(pg.mkColor(colour))
                pen.setStyle(Qt.PenStyle.DashLine)
                pen.setCosmetic(True)
                line = pg.InfiniteLine(
                    pos=best, angle=0, pen=pen, movable=False,
                )
                self._plot.addItem(line)
                self._dyn_items.append(line)

        # Replace numeric x-axis with lap labels.
        ax = self._plot.getAxis("bottom")
        ax.setTicks(
            [list(zip(x_lap.tolist(),
                      [f"L{i}" for i in lap_indices]))]
        )
        self._plot.getViewBox().autoRange()


__all__ = ["SectorsTab"]
