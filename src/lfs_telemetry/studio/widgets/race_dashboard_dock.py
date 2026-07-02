"""Live race dashboard dock — at-a-glance race state inside the Studio.

Mirrors the most important fields published to ``live.json`` by the
capture CLI (see :mod:`lfs_telemetry.telemetry.live_publisher`) but
keeps everything inside a single non-floating dock so the user does
not have to enable individual floating overlay windows just to see
position / gaps / fuel / predicted lap.

The dock owns its own :class:`LiveDataSource` and polls the shared
:class:`CaptureRunner` for the path to ``live.json`` — the exact same
pattern used by :class:`LiveTab`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...app.capture_runner import CaptureRunner
from ..i18n import tr
from ..signals import SignalBus
from ..theme import MUTED_COLOR, TEXT_COLOR
from ._format import (
    format_gap_meters,
    format_lap_time_ms,
    format_signed_delta_s,
)
from .live_data_source import LiveDataSource

_GOOD = "#7ed957"
_WARN = "#ffe066"
_BAD = "#ff5d6c"
_NEUTRAL = TEXT_COLOR


def _fmt_lap_ms(ms: int | None) -> str:
    return format_lap_time_ms(ms)


def _fmt_delta_ms(ms: int | None) -> tuple[str, str]:
    if ms is None:
        return "—", _NEUTRAL
    s = float(ms) / 1000.0
    colour = _GOOD if s < 0 else (_BAD if s > 0 else _NEUTRAL)
    return format_signed_delta_s(s), colour


def _fmt_gap_m(m: float | None) -> str:
    return format_gap_meters(m)


class _BigValue(QFrame):
    """A title + large-value label pair, styled as a small card."""

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        *,
        size_pt: int = 22,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred,
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(0)
        self._title = QLabel(title, self)
        self._title.setStyleSheet(
            f"color:{MUTED_COLOR}; font-size: 8pt;"
        )
        self._value = QLabel("—", self)
        self._value.setStyleSheet(
            f"color:{TEXT_COLOR};"
            f" font-size:{size_pt}pt;"
            f" font-weight:600;"
        )
        self._value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._title)
        lay.addWidget(self._value)

    def set_value(self, text: str, colour: str | None = None) -> None:
        self._value.setText(text)
        if colour is not None:
            self._value.setStyleSheet(
                f"color:{colour};"
                f" font-size:{self._value.font().pointSize()}pt;"
                f" font-weight:600;"
            )

    def set_title(self, title: str) -> None:
        self._title.setText(title)


class RaceDashboardDock(QWidget):
    """At-a-glance race dashboard reading live.json."""

    def __init__(
        self,
        runner: CaptureRunner,
        signals: SignalBus,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._signals = signals

        self._source = LiveDataSource(parent=self)
        self._source.start()

        # ----- Widgets -------------------------------------------------
        self._w_position = _BigValue(tr("Position"), self, size_pt=28)
        self._w_lap = _BigValue(tr("Lap"), self, size_pt=20)
        self._w_current = _BigValue(tr("Current lap"), self, size_pt=22)
        self._w_last = _BigValue(tr("Last lap"), self, size_pt=20)
        self._w_best = _BigValue(tr("Best lap"), self, size_pt=20)
        self._w_predicted = _BigValue(tr("Predicted"), self, size_pt=20)
        self._w_delta = _BigValue(tr("\u0394 vs best"), self, size_pt=22)
        self._w_spb = _BigValue(tr("SPB"), self, size_pt=18)
        self._w_avg = _BigValue(tr("Avg (stint)"), self, size_pt=18)
        # Rotation mode for the Avg card: cycles stint→clean→total
        # every ~5 s, mirroring D&M's average_auto_interval. The
        # initial mode is the most useful one for race pace.
        self._avg_modes: tuple[str, ...] = ("stint", "clean", "total")
        self._avg_idx: int = 0
        self._avg_timer = QTimer(self)
        self._avg_timer.setInterval(5000)
        self._avg_timer.timeout.connect(self._rotate_avg_mode)
        self._avg_timer.start()
        self._last_avg_snap: dict[str, int | None] = {
            "stint": None, "clean": None, "total": None,
        }
        self._w_gap_ahead = _BigValue(tr("Gap ahead"), self, size_pt=20)
        self._w_gap_behind = _BigValue(tr("Gap behind"), self, size_pt=20)
        self._w_fuel_pct = _BigValue(tr("Fuel"), self, size_pt=20)
        self._w_fuel_laps = _BigValue(
            tr("Fuel laps left"), self, size_pt=20,
        )
        self._w_speed = _BigValue(tr("Speed"), self, size_pt=20)
        self._w_gear = _BigValue(tr("Gear"), self, size_pt=28)

        self._standings_table = QLabel(self)
        self._standings_table.setTextFormat(Qt.TextFormat.PlainText)
        self._standings_table.setWordWrap(False)
        self._standings_table.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._standings_table.setStyleSheet(
            f"color:{TEXT_COLOR};"
            "font-family: Consolas, 'Courier New', monospace;"
            "font-size: 10pt;"
            "padding: 4px 6px;"
        )

        # Group: timing
        timing = QGroupBox(tr("Timing"), self)
        gt = QGridLayout(timing)
        gt.setSpacing(4)
        gt.addWidget(self._w_position, 0, 0)
        gt.addWidget(self._w_lap, 0, 1)
        gt.addWidget(self._w_delta, 0, 2)
        gt.addWidget(self._w_current, 1, 0)
        gt.addWidget(self._w_last, 1, 1)
        gt.addWidget(self._w_best, 1, 2)
        gt.addWidget(self._w_predicted, 2, 0)
        gt.addWidget(self._w_spb, 2, 1)
        gt.addWidget(self._w_avg, 2, 2)

        # Group: gaps
        gaps = QGroupBox(tr("Gaps to rivals"), self)
        gg = QGridLayout(gaps)
        gg.setSpacing(4)
        gg.addWidget(self._w_gap_ahead, 0, 0)
        gg.addWidget(self._w_gap_behind, 0, 1)

        # Group: fuel + drive
        fuel = QGroupBox(tr("Fuel / Drive"), self)
        gf = QGridLayout(fuel)
        gf.setSpacing(4)
        gf.addWidget(self._w_fuel_pct, 0, 0)
        gf.addWidget(self._w_fuel_laps, 0, 1)
        gf.addWidget(self._w_speed, 1, 0)
        gf.addWidget(self._w_gear, 1, 1)

        standings = QGroupBox(tr("Race classification"), self)
        self._standings_group = standings
        gs = QVBoxLayout(standings)
        gs.setContentsMargins(4, 4, 4, 4)
        gs.addWidget(self._standings_table)

        # Context strip (track, weather, race status, capture state)
        self._context = QLabel(tr("Waiting for capture\u2026"), self)
        self._context.setWordWrap(True)
        self._context.setStyleSheet(
            f"color:{MUTED_COLOR}; padding:4px 6px;"
        )
        self._context.setTextFormat(Qt.TextFormat.RichText)

        # ----- Layout --------------------------------------------------
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)
        lay.addWidget(self._context)
        lay.addWidget(timing)
        lay.addWidget(gaps)
        lay.addWidget(fuel)
        lay.addWidget(standings)
        lay.addStretch(1)

        # ----- Wiring --------------------------------------------------
        self._source.snapshot_changed.connect(self._on_snapshot)
        self._source.available_changed.connect(self._on_available_changed)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._poll_runner)
        self._timer.start()
        self._poll_runner()

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _poll_runner(self) -> None:
        st = self._runner.status()
        path_str = st.get("live_file") or ""
        path = Path(path_str) if path_str else None
        self._source.set_path(path)

    def _rotate_avg_mode(self) -> None:
        self._avg_idx = (self._avg_idx + 1) % len(self._avg_modes)
        self._refresh_avg_card()

    def _refresh_avg_card(self) -> None:
        mode = self._avg_modes[self._avg_idx]
        self._w_avg.set_title(
            tr("Avg ({mode})").format(mode=tr(mode)),
        )
        self._w_avg.set_value(_fmt_lap_ms(self._last_avg_snap.get(mode)))

    def _on_available_changed(self, available: bool) -> None:
        if not available:
            self._reset_values()

    def _reset_values(self) -> None:
        for w in (
            self._w_position, self._w_lap, self._w_current, self._w_last,
            self._w_best, self._w_predicted, self._w_delta, self._w_spb,
            self._w_avg,
            self._w_gap_ahead, self._w_gap_behind, self._w_fuel_pct,
            self._w_fuel_laps, self._w_speed, self._w_gear,
        ):
            w.set_value("—", _NEUTRAL)
        self._standings_table.setText(tr("No classification data yet."))

    def _set_standings(self, snap: dict[str, Any]) -> None:
        mode = str(snap.get("session_mode") or "practice")
        if mode == "race":
            self._standings_group.setTitle(tr("Race classification"))
        elif mode == "qualifying":
            self._standings_group.setTitle(tr("Qualifying leaderboard"))
        else:
            self._standings_group.setTitle(tr("Session leaderboard"))

        standings = snap.get("standings")
        if not isinstance(standings, list) or not standings:
            self._standings_table.setText(tr("No classification data yet."))
            return
        head = "{pos:>3} {drv:<14} {lap:>3} {lst:>10} {bst:>10}".format(
            pos=tr("Pos"),
            drv=tr("Driver"),
            lap=tr("Lap"),
            lst=tr("Last"),
            bst=tr("Best"),
        )
        lines: list[str] = [head]
        max_rows = 10
        for row in standings[:max_rows]:
            try:
                pos = int(row.get("pos", 0))
                name = str(row.get("name") or "?")
                lap = int(row.get("lap", 0))
                last = _fmt_lap_ms(row.get("last_lap_ms"))
                best = _fmt_lap_ms(row.get("best_lap_ms"))
                mark = ">" if bool(row.get("view")) else " "
                lines.append(
                    f"{mark}{pos:>2} {name[:14]:<14} {lap:>3}"
                    f" {last:>10} {best:>10}"
                )
            except (TypeError, ValueError):
                continue
        self._standings_table.setText("\n".join(lines))

    # ------------------------------------------------------------------
    # Snapshot handler
    # ------------------------------------------------------------------

    @staticmethod
    def _gap_to_nearest(
        snap: dict[str, Any],
        *,
        ahead: bool,
    ) -> float | None:
        """Return nearest rival gap in metres from the published
        traffic snapshot.

        ``live_publisher`` already computes robust ahead/behind gaps
        from race-position and spatial fallbacks. Reusing those values
        keeps the dock coherent with the capture-side telemetry logic.
        """
        traffic = snap.get("traffic")
        if isinstance(traffic, dict):
            key = "ahead_gap_m" if ahead else "behind_gap_m"
            raw = traffic.get(key)
            if raw is not None:
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    value = None
                if value is not None and value >= 0.0:
                    return value
        return None

    def _on_snapshot(self, snap: dict[str, Any]) -> None:
        if not snap:
            self._reset_values()
            return

        # ----- Context strip ------------------------------------------
        track = snap.get("track") or "?"
        weather = snap.get("weather")
        in_progress = snap.get("race_in_progress")
        armed = bool(snap.get("armed"))
        samples = int(snap.get("samples") or 0)
        bits: list[str] = [f"<b>{track}</b>"]
        mode = str(snap.get("session_mode") or "practice")
        bits.append(tr(mode))
        if weather:
            bits.append(tr("weather {value}").format(value=weather))
        if in_progress is True:
            bits.append(
                "<span style='color:{c}'>{txt}</span>".format(
                    c=_GOOD, txt=tr("race"),
                )
            )
        elif in_progress is False:
            bits.append(
                "<span style='color:{c}'>{txt}</span>".format(
                    c=_BAD, txt=tr("idle"),
                )
            )
        bits.append(
            tr("capture {state} \u00b7 {n} samples").format(
                state=tr("ARMED") if armed else tr("off"),
                n=samples,
            )
        )
        self._context.setText(" \u00b7 ".join(bits))

        # ----- Position / lap -----------------------------------------
        pos = snap.get("view_position")
        cars = snap.get("cars") or []
        n_cars = len(cars) if cars else 0
        traffic = snap.get("traffic")
        if isinstance(traffic, dict):
            n_from_traffic = traffic.get("num_cars")
            if isinstance(n_from_traffic, int) and n_from_traffic > 0:
                n_cars = n_from_traffic
        if pos is not None:
            tag = f"P{int(pos)}" if n_cars == 0 else f"P{int(pos)}/{n_cars}"
            colour = _GOOD if int(pos) == 1 else _NEUTRAL
            self._w_position.set_value(tag, colour)
        else:
            self._w_position.set_value("—", _NEUTRAL)
        lap = snap.get("view_lap")
        self._w_lap.set_value(str(int(lap)) if lap is not None else "—")

        # ----- Timing -------------------------------------------------
        self._w_current.set_value(_fmt_lap_ms(snap.get("current_lap_ms")))
        self._w_last.set_value(_fmt_lap_ms(snap.get("last_lap_ms")))
        self._w_best.set_value(_fmt_lap_ms(snap.get("best_lap_ms")), _GOOD)
        self._w_predicted.set_value(_fmt_lap_ms(snap.get("predicted_lap_ms")))
        self._w_spb.set_value(_fmt_lap_ms(snap.get("spb_ms")))

        averages = snap.get("lap_averages_ms") or {}
        self._last_avg_snap = {
            "stint": averages.get("stint"),
            "clean": averages.get("clean"),
            "total": averages.get("total"),
        }
        self._refresh_avg_card()

        delta_txt, delta_col = _fmt_delta_ms(snap.get("delta_vs_best_ms"))
        self._w_delta.set_value(delta_txt, delta_col)

        # ----- Gaps ---------------------------------------------------
        gap_a = self._gap_to_nearest(snap, ahead=True)
        gap_b = self._gap_to_nearest(snap, ahead=False)
        self._w_gap_ahead.set_value(_fmt_gap_m(gap_a))
        self._w_gap_behind.set_value(_fmt_gap_m(gap_b))

        # ----- Classification -----------------------------------------
        self._set_standings(snap)

        # ----- Fuel / drive -------------------------------------------
        fpct = snap.get("view_fuel_pct")
        if fpct is not None:
            fcol = _GOOD
            if float(fpct) < 8.0:
                fcol = _BAD
            elif float(fpct) < 20.0:
                fcol = _WARN
            self._w_fuel_pct.set_value(f"{float(fpct):.1f}%", fcol)
        else:
            self._w_fuel_pct.set_value("—", _NEUTRAL)
        flaps = snap.get("fuel_laps_remaining")
        if flaps is not None:
            lcol = _GOOD
            if float(flaps) < 1.0:
                lcol = _BAD
            elif float(flaps) < 3.0:
                lcol = _WARN
            self._w_fuel_laps.set_value(f"{float(flaps):.2f}", lcol)
        else:
            self._w_fuel_laps.set_value("—", _NEUTRAL)

        sp = snap.get("view_speed_kmh")
        self._w_speed.set_value(
            f"{float(sp):.0f} km/h" if sp is not None else "—",
        )
        gear = snap.get("view_gear")
        if gear is None:
            self._w_gear.set_value("—")
        else:
            g = int(gear)
            label = "R" if g < 0 else ("N" if g == 0 else str(g))
            self._w_gear.set_value(label)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._avg_timer.stop()
        self._timer.stop()
        self._source.stop()
        super().closeEvent(event)


__all__ = ["RaceDashboardDock"]
