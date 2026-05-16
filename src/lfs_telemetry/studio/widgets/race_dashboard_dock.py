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
from ..signals import SignalBus
from ..theme import MUTED_COLOR, TEXT_COLOR
from .live_data_source import LiveDataSource
from ._format import (
    format_gap_meters,
    format_lap_time_ms,
    format_signed_delta_s,
)


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
        self._w_position = _BigValue("Position", self, size_pt=28)
        self._w_lap = _BigValue("Lap", self, size_pt=20)
        self._w_current = _BigValue("Current lap", self, size_pt=22)
        self._w_last = _BigValue("Last lap", self, size_pt=20)
        self._w_best = _BigValue("Best lap", self, size_pt=20)
        self._w_predicted = _BigValue("Predicted", self, size_pt=20)
        self._w_delta = _BigValue("Δ vs best", self, size_pt=22)
        self._w_spb = _BigValue("SPB", self, size_pt=18)
        self._w_gap_ahead = _BigValue("Gap ahead", self, size_pt=20)
        self._w_gap_behind = _BigValue("Gap behind", self, size_pt=20)
        self._w_fuel_pct = _BigValue("Fuel", self, size_pt=20)
        self._w_fuel_laps = _BigValue("Fuel laps left", self, size_pt=20)
        self._w_speed = _BigValue("Speed", self, size_pt=20)
        self._w_gear = _BigValue("Gear", self, size_pt=28)

        # Group: timing
        timing = QGroupBox("Timing", self)
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

        # Group: gaps
        gaps = QGroupBox("Gaps to rivals", self)
        gg = QGridLayout(gaps)
        gg.setSpacing(4)
        gg.addWidget(self._w_gap_ahead, 0, 0)
        gg.addWidget(self._w_gap_behind, 0, 1)

        # Group: fuel + drive
        fuel = QGroupBox("Fuel / Drive", self)
        gf = QGridLayout(fuel)
        gf.setSpacing(4)
        gf.addWidget(self._w_fuel_pct, 0, 0)
        gf.addWidget(self._w_fuel_laps, 0, 1)
        gf.addWidget(self._w_speed, 1, 0)
        gf.addWidget(self._w_gear, 1, 1)

        # Context strip (track, weather, race status, capture state)
        self._context = QLabel("Waiting for capture…", self)
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

    def _on_available_changed(self, available: bool) -> None:
        if not available:
            self._reset_values()

    def _reset_values(self) -> None:
        for w in (
            self._w_position, self._w_lap, self._w_current, self._w_last,
            self._w_best, self._w_predicted, self._w_delta, self._w_spb,
            self._w_gap_ahead, self._w_gap_behind, self._w_fuel_pct,
            self._w_fuel_laps, self._w_speed, self._w_gear,
        ):
            w.set_value("—", _NEUTRAL)

    # ------------------------------------------------------------------
    # Snapshot handler
    # ------------------------------------------------------------------

    @staticmethod
    def _gap_to_nearest(
        cars: list[dict[str, Any]],
        view_plid: int | None,
        *,
        ahead: bool,
    ) -> float | None:
        """Return the smallest positive gap (m) to a rival in front
        or behind the camera car, derived from per-car ``progress_m``
        values when present.
        """
        if view_plid is None or not cars:
            return None
        own_prog: float | None = None
        for c in cars:
            if int(c.get("plid", -1)) == view_plid:
                own_prog = c.get("progress_m")
                break
        if own_prog is None:
            return None
        best: float | None = None
        for c in cars:
            if int(c.get("plid", -1)) == view_plid:
                continue
            p = c.get("progress_m")
            if p is None:
                continue
            diff = float(p) - float(own_prog)
            if ahead and diff > 0 and (best is None or diff < best):
                best = diff
            elif (not ahead) and diff < 0 and (
                best is None or -diff < best
            ):
                best = -diff
        return best

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
        if weather:
            bits.append(f"weather {weather}")
        if in_progress is True:
            bits.append("<span style='color:#7ed957'>race</span>")
        elif in_progress is False:
            bits.append("<span style='color:#ff5d6c'>idle</span>")
        bits.append(
            f"capture {'ARMED' if armed else 'off'} · {samples} samples"
        )
        self._context.setText(" · ".join(bits))

        # ----- Position / lap -----------------------------------------
        pos = snap.get("view_position")
        cars = snap.get("cars") or []
        n_cars = len(cars) if cars else 0
        if pos:
            tag = f"P{int(pos)}" if n_cars == 0 else f"P{int(pos)}/{n_cars}"
            colour = _GOOD if int(pos) == 1 else _NEUTRAL
            self._w_position.set_value(tag, colour)
        else:
            self._w_position.set_value("—", _NEUTRAL)
        lap = snap.get("view_lap")
        self._w_lap.set_value(str(int(lap)) if lap else "—")

        # ----- Timing -------------------------------------------------
        self._w_current.set_value(_fmt_lap_ms(snap.get("current_lap_ms")))
        self._w_last.set_value(_fmt_lap_ms(snap.get("last_lap_ms")))
        self._w_best.set_value(_fmt_lap_ms(snap.get("best_lap_ms")), _GOOD)
        self._w_predicted.set_value(_fmt_lap_ms(snap.get("predicted_lap_ms")))
        self._w_spb.set_value(_fmt_lap_ms(snap.get("spb_ms")))

        delta_txt, delta_col = _fmt_delta_ms(snap.get("delta_vs_best_ms"))
        self._w_delta.set_value(delta_txt, delta_col)

        # ----- Gaps ---------------------------------------------------
        view_plid = snap.get("view_plid")
        gap_a = self._gap_to_nearest(cars, view_plid, ahead=True)
        gap_b = self._gap_to_nearest(cars, view_plid, ahead=False)
        self._w_gap_ahead.set_value(_fmt_gap_m(gap_a))
        self._w_gap_behind.set_value(_fmt_gap_m(gap_b))

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

    def closeEvent(self, event) -> None:  # noqa: N802
        self._source.stop()
        super().closeEvent(event)


__all__ = ["RaceDashboardDock"]
