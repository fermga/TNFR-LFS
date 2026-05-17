"""Stint tab — per-lap trends across the currently selected laps.

Listens for ``laps_selected`` on the shared :class:`SignalBus` and, as
the :class:`LapLoader` finishes loading each lap, rebuilds a stack of
charts:

* lap times (bar chart, mean dashed)
* fuel: % used per lap (bars) and % remaining at lap end (line)
* tyre temperature at end-of-lap, per wheel (FL/FR/RL/RR)
* peak vertical load per wheel — proxy for suspension / chassis stress
  (LFS OutSim does not expose dynamic tyre pressures)

A text header above the plots aggregates the most useful stint-wide
numbers: best / mean / theoretical lap, pace drop-off, total fuel
used and laps remaining, peak Gs and the per-wheel tyre-temperature
trend across the stint (warm-up vs heat soak).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QPen
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ...telemetry import LapTelemetry, StintTelemetry
from ..models import LapLoader
from ..signals import SignalBus
from ..i18n import tr
from ..theme import (
    MUTED_COLOR,
    PANEL_COLOR,
    TEXT_COLOR,
    WHEEL_COLORS as _WHEEL_COLORS,
    WHEEL_ORDER_UI as _UI_WHEEL_ORDER,
    trace_color,
)
from ._format import format_lap_time_s, format_signed_delta_s


def _fmt_time(s: float | None) -> str:
    return format_lap_time_s(s)


def _fmt_gap(s: float | None) -> str:
    return format_signed_delta_s(s)


def _styled_plot(
    parent_layout: pg.GraphicsLayoutWidget,
    title: str,
    ylabel: str,
    row: int,
) -> pg.PlotItem:
    plot = parent_layout.addPlot(row=row, col=0)
    plot.setTitle(title, color=TEXT_COLOR, size="9pt")
    plot.showGrid(x=False, y=True, alpha=0.12)
    plot.getAxis("left").setLabel(ylabel, color=TEXT_COLOR)
    plot.getAxis("bottom").setLabel(tr("Lap #"), color=TEXT_COLOR)
    plot.getAxis("left").setTextPen(TEXT_COLOR)
    plot.getAxis("bottom").setTextPen(TEXT_COLOR)
    return plot


class StintTab(QWidget):
    """Per-stint summary header + stacked per-lap trend charts."""

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

        # Header summary
        self._summary = QLabel(tr("No laps selected."), self)
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet(
            f"color:{MUTED_COLOR}; padding:4px 6px;"
        )

        # Stacked plots in a single GraphicsLayoutWidget so they share
        # width and align cleanly.
        self._gfx = pg.GraphicsLayoutWidget(parent=self)
        self._gfx.setBackground(PANEL_COLOR)
        self._gfx.ci.layout.setSpacing(2)
        self._gfx.ci.layout.setContentsMargins(2, 2, 2, 2)

        self._p_times = _styled_plot(self._gfx, tr("Lap times"), "s", row=0)
        self._p_fuel = _styled_plot(self._gfx, tr("Fuel"), "%", row=1)
        self._p_tyre = _styled_plot(
            self._gfx, tr("Tyre temp end-of-lap"), "\u00b0C", row=2,
        )
        self._p_susp = _styled_plot(
            self._gfx, tr("Peak vertical load (suspension)"), "kN", row=3,
        )
        self._p_friction = _styled_plot(
            self._gfx,
            tr("Friction use p95 (circle saturation)"), "", row=4,
        )
        self._p_damper = _styled_plot(
            self._gfx, tr("Damper work \u2014 RMS shaft speed"),
            "mm/s", row=5,
        )
        # Link x-axes so zoom/pan stays in sync per lap index.
        for p in (self._p_fuel, self._p_tyre, self._p_susp,
                  self._p_friction, self._p_damper):
            p.setXLink(self._p_times)
        for p in (self._p_times, self._p_fuel, self._p_tyre,
                  self._p_susp, self._p_friction, self._p_damper):
            p.setMinimumHeight(140)

        # Items recreated each refresh; tracked so we can clear them.
        self._mean_line: pg.InfiniteLine | None = None
        self._dyn_items: List[pg.GraphicsObject] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self._summary)
        layout.addWidget(self._gfx, 1)

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

    def _all_plots(self) -> List[pg.PlotItem]:
        return [self._p_times, self._p_fuel, self._p_tyre,
                self._p_susp, self._p_friction, self._p_damper]

    def _clear_plots(self) -> None:
        for it in self._dyn_items:
            for plot in self._all_plots():
                try:
                    plot.removeItem(it)
                except Exception:  # noqa: BLE001
                    pass
        self._dyn_items.clear()
        if self._mean_line is not None:
            try:
                self._p_times.removeItem(self._mean_line)
            except Exception:  # noqa: BLE001
                pass
            self._mean_line = None
        for plot in self._all_plots():
            legend = plot.legend
            if legend is not None:
                try:
                    legend.scene().removeItem(legend)
                except Exception:  # noqa: BLE001
                    pass
                plot.legend = None

    def _reset(self) -> None:
        self._summary.setText(tr("No laps selected."))
        self._clear_plots()

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
        try:
            stint = StintTelemetry.from_laps(ordered)
            df = stint.per_lap
            trends = stint.trends
        except Exception as exc:  # noqa: BLE001
            self._summary.setText(
                tr("Stint build failed: {error}").format(error=exc),
            )
            return

        # Header summary — stint-wide overview.
        n = trends.get("num_laps", len(ordered))
        car = trends.get("car") or "?"
        track = trends.get("track") or "?"
        line1 = [f"<b>{n}</b> lap(s) — {car} @ {track}"]
        if "lap_time_best_s" in trends:
            line1.append(
                f"best <b>{_fmt_time(trends['lap_time_best_s'])}</b>"
            )
        if "lap_time_mean_s" in trends:
            line1.append(f"mean {_fmt_time(trends['lap_time_mean_s'])}")
        try:
            tb = stint.theoretical_best_lap()
        except Exception:  # noqa: BLE001
            tb = {}
        if tb and np.isfinite(tb.get("theoretical_best_s", float("nan"))):
            line1.append(
                f"theoretical {_fmt_time(tb['theoretical_best_s'])}"
                f" (gap {_fmt_gap(tb.get('gap_s'))})"
            )
        if "pace_dropoff_s_per_lap" in trends:
            line1.append(
                f"drop-off {trends['pace_dropoff_s_per_lap']:+.3f} s/lap"
            )

        # Line 2 — fuel + Gs.
        line2: List[str] = []
        if "fuel_pct_used" in df.columns:
            total = float(df["fuel_pct_used"].fillna(0.0).sum())
            line2.append(f"fuel total <b>{total:.2f}</b> %")
        if "fuel_pct_per_lap_mean" in trends:
            line2.append(
                f"avg {trends['fuel_pct_per_lap_mean'] * 100:.2f} %/lap"
            )
        if "fuel_laps_remaining" in trends:
            line2.append(
                f"≈{trends['fuel_laps_remaining']:.1f} laps left"
            )
        if "peak_long_g" in df.columns:
            v = df["peak_long_g"].dropna()
            if not v.empty:
                line2.append(f"peak long {float(v.abs().max()):.2f} g")
        if "peak_lat_g" in df.columns:
            v = df["peak_lat_g"].dropna()
            if not v.empty:
                line2.append(f"peak lat {float(v.abs().max()):.2f} g")
        if "top_speed_kmh" in df.columns:
            v = df["top_speed_kmh"].dropna()
            if not v.empty:
                line2.append(f"top {float(v.max()):.1f} km/h")
        if "pit_in_lap" in df.columns:
            pit_laps = df.loc[df["pit_in_lap"].astype(bool),
                              "lap_index"].tolist()
            if pit_laps:
                laps_txt = ", ".join(f"L{int(i)}" for i in pit_laps)
                line2.append(
                    f"🔧 pit ×<b>{len(pit_laps)}</b> ({laps_txt})"
                )

        # Line 3 — per-wheel tyre temperature trend across the stint
        # (last lap end vs first lap end). Positive = heat soak, negative
        # = tyres cooled down (e.g. wet patch, slowing pace).
        line3: List[str] = []
        if len(df) >= 2:
            for c in _UI_WHEEL_ORDER:
                col = f"tyre_temp_end_c_{c}"
                if col not in df.columns:
                    continue
                arr = df[col].dropna()
                if len(arr) < 2:
                    continue
                delta = float(arr.iloc[-1]) - float(arr.iloc[0])
                arrow = "↑" if delta > 0.5 else ("↓" if delta < -0.5 else "→")
                line3.append(f"{c} {arrow}{delta:+.1f}")
        tyre_trend = (
            "tyres Δ°C (last − first):  " + "   ".join(line3)
            if line3 else ""
        )

        html = "<br/>".join(
            seg for seg in (
                "  •  ".join(line1),
                "  •  ".join(line2) if line2 else "",
                tyre_trend,
            ) if seg
        )
        self._summary.setText(html)

        self._clear_plots()
        self._render_lap_times(df)
        self._render_fuel(df)
        self._render_tyre_temps(df)
        self._render_suspension(df)
        self._render_friction(df)
        self._render_dampers(df, ordered)

    # ------------------------------------------------------------------
    # Plot builders
    # ------------------------------------------------------------------

    def _x_axis(self, df) -> np.ndarray:
        return df["lap_index"].to_numpy(dtype=float)

    def _render_lap_times(self, df) -> None:
        if "lap_time_s" not in df or df["lap_time_s"].dropna().empty:
            return
        x = self._x_axis(df)
        y = df["lap_time_s"].to_numpy(dtype=float)
        if "is_race_start" in df:
            mask = ~df["is_race_start"].fillna(False).astype(bool).to_numpy()
            mean_y = (
                float(np.nanmean(y[mask])) if mask.any()
                else float(np.nanmean(y))
            )
        else:
            mean_y = float(np.nanmean(y))
        brushes = [pg.mkBrush(trace_color(int(i) - 1)) for i in x]
        bars = pg.BarGraphItem(
            x=x, height=y, width=0.6, brushes=brushes,
            pen=pg.mkPen(color="#202830", width=1),
        )
        self._p_times.addItem(bars)
        self._dyn_items.append(bars)
        if np.isfinite(mean_y):
            pen = QPen(pg.mkColor(MUTED_COLOR))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._mean_line = pg.InfiniteLine(
                pos=mean_y, angle=0, pen=pen, movable=False,
            )
            self._p_times.addItem(self._mean_line)
        # Pit-stop markers: red wrench above bars of laps where a stop
        # happened (count_end > count_start in the lap's CSV).
        if "pit_in_lap" in df.columns:
            pit_mask = df["pit_in_lap"].fillna(False).astype(bool).to_numpy()
            if pit_mask.any():
                y_top = float(np.nanmax(y)) if np.isfinite(y).any() else 0.0
                y_marker = y_top * 1.06 if y_top > 0 else 1.0
                pit_x = x[pit_mask]
                pit_scatter = pg.ScatterPlotItem(
                    pit_x, np.full(pit_x.shape, y_marker),
                    symbol="t1", size=14,
                    brush=pg.mkBrush("#ff5d6c"),
                    pen=pg.mkPen("#202830", width=1),
                )
                self._p_times.addItem(pit_scatter)
                self._dyn_items.append(pit_scatter)
                for xi in pit_x:
                    txt = pg.TextItem("PIT", color="#ff5d6c",
                                      anchor=(0.5, 1.4))
                    txt.setPos(float(xi), y_marker)
                    self._p_times.addItem(txt)
                    self._dyn_items.append(txt)
        self._p_times.getViewBox().autoRange()

    def _render_fuel(self, df) -> None:
        x = self._x_axis(df)
        plot = self._p_fuel
        plot.addLegend(offset=(8, 4), labelTextColor=TEXT_COLOR)
        if "fuel_pct_used" in df:
            used = df["fuel_pct_used"].to_numpy(dtype=float)
            if np.isfinite(used).any():
                bars = pg.BarGraphItem(
                    x=x, height=used, width=0.55,
                    brush=pg.mkBrush("#4ea3ff"),
                    pen=pg.mkPen(color="#202830", width=1),
                )
                plot.addItem(bars)
                self._dyn_items.append(bars)
                # BarGraphItem isn't picked up automatically by legend.
                sample = pg.PlotDataItem(
                    pen=pg.mkPen("#4ea3ff", width=6),
                )
                plot.legend.addItem(sample, tr("used / lap"))
        if "fuel_pct_end" in df:
            end = df["fuel_pct_end"].to_numpy(dtype=float)
            if np.isfinite(end).any():
                curve = plot.plot(
                    x, end,
                    pen=pg.mkPen("#ffa040", width=2),
                    symbol="o", symbolSize=6,
                    symbolBrush="#ffa040", symbolPen=None,
                    name=tr("remaining @ end"),
                )
                self._dyn_items.append(curve)
        plot.getViewBox().autoRange()

    def _render_tyre_temps(self, df) -> None:
        x = self._x_axis(df)
        plot = self._p_tyre
        plot.addLegend(offset=(8, 4), labelTextColor=TEXT_COLOR)
        for c in _UI_WHEEL_ORDER:
            col = f"tyre_temp_end_c_{c}"
            if col not in df.columns:
                continue
            y = df[col].to_numpy(dtype=float)
            if not np.isfinite(y).any():
                continue
            color = _WHEEL_COLORS[c]
            curve = plot.plot(
                x, y,
                pen=pg.mkPen(color, width=2),
                symbol="o", symbolSize=5,
                symbolBrush=color, symbolPen=None,
                name=c,
            )
            self._dyn_items.append(curve)
        plot.getViewBox().autoRange()

    def _render_suspension(self, df) -> None:
        """Peak vertical load per wheel as a proxy for chassis stress.

        OutSim does not expose dynamic tyre pressures, so this is the
        closest "suspension" indicator we can compute per lap.
        """
        x = self._x_axis(df)
        plot = self._p_susp
        plot.addLegend(offset=(8, 4), labelTextColor=TEXT_COLOR)
        for c in _UI_WHEEL_ORDER:
            col = f"vert_load_max_n_{c}"
            if col not in df.columns:
                continue
            y_n = df[col].to_numpy(dtype=float)
            if not np.isfinite(y_n).any():
                continue
            color = _WHEEL_COLORS[c]
            curve = plot.plot(
                x, y_n / 1000.0,  # display in kN
                pen=pg.mkPen(color, width=2),
                symbol="o", symbolSize=5,
                symbolBrush=color, symbolPen=None,
                name=c,
            )
            self._dyn_items.append(curve)
        plot.getViewBox().autoRange()

    def _render_friction(self, df) -> None:
        """Friction-circle utilisation p95 per wheel across laps.

        Saturation toward 1.0 means the tyre is consistently working
        near its grip envelope on that wheel. A rising trend across
        laps signals overdriving or grip loss.
        """
        x = self._x_axis(df)
        plot = self._p_friction
        plot.addLegend(offset=(8, 4), labelTextColor=TEXT_COLOR)
        any_data = False
        for c in _UI_WHEEL_ORDER:
            col = f"friction_use_p95_{c}"
            if col not in df.columns:
                continue
            y = df[col].to_numpy(dtype=float)
            if not np.isfinite(y).any():
                continue
            any_data = True
            color = _WHEEL_COLORS[c]
            curve = plot.plot(
                x, y,
                pen=pg.mkPen(color, width=2),
                symbol="o", symbolSize=5,
                symbolBrush=color, symbolPen=None,
                name=c,
            )
            self._dyn_items.append(curve)
        if any_data:
            # Reference line at 1.0 = full friction-circle saturation.
            ref_pen = QPen(pg.mkColor(MUTED_COLOR))
            ref_pen.setStyle(Qt.PenStyle.DashLine)
            ref_pen.setCosmetic(True)
            ref = pg.InfiniteLine(pos=1.0, angle=0, pen=ref_pen,
                                  movable=False)
            plot.addItem(ref)
            self._dyn_items.append(ref)
        plot.getViewBox().autoRange()

    def _render_dampers(
        self, df, ordered: List[LapTelemetry],
    ) -> None:
        """Per-lap RMS damper shaft speed per wheel (mm/s).

        Proxy for the work done by the dampers over the lap — high
        values flag harsh kerb usage / rough surface. Computed live
        from each lap's ``wheel_<c>_susp_speed_mps`` (m/s) channel.
        """
        x = self._x_axis(df)
        plot = self._p_damper
        plot.addLegend(offset=(8, 4), labelTextColor=TEXT_COLOR)
        # Build a {lap_index: {wheel: rms_mm_s}} table.
        per_lap_rms: Dict[int, Dict[str, float]] = {}
        for lap in ordered:
            idx = int(lap.summary.get("lap_index", 0))
            raw = lap.raw
            row: Dict[str, float] = {}
            for c in _UI_WHEEL_ORDER:
                col = f"wheel_{c}_susp_speed_mps"
                if col not in raw.columns:
                    continue
                arr = raw[col].to_numpy(dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                # RMS in mm/s (×1000 to go from m/s).
                row[c] = float(np.sqrt(np.mean(arr * arr))) * 1000.0
            if row:
                per_lap_rms[idx] = row
        if not per_lap_rms:
            return
        for c in _UI_WHEEL_ORDER:
            ys = np.array(
                [per_lap_rms.get(int(i), {}).get(c, np.nan) for i in x],
                dtype=float,
            )
            if not np.isfinite(ys).any():
                continue
            color = _WHEEL_COLORS[c]
            curve = plot.plot(
                x, ys,
                pen=pg.mkPen(color, width=2),
                symbol="o", symbolSize=5,
                symbolBrush=color, symbolPen=None,
                name=c,
            )
            self._dyn_items.append(curve)
        plot.getViewBox().autoRange()


__all__ = ["StintTab"]
