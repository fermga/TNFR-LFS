"""Independent overlay modules driven by :class:`LiveDataSource`.

Every datum the live snapshot carries is exposed as its own toggleable,
draggable, **resizable** frameless window. The Studio Live tab toggles
each module on/off independently so users can build whatever overlay
layout they want.

All windows share :class:`_LiveModuleWindow`, which provides:
* frameless / always-on-top / translucent chrome
* configurable opacity
* drag-anywhere-to-move (left-click anywhere on the window body)
* drag-bottom-right-corner-to-resize (within ``MIN_W/MIN_H``..)
* automatic font/element scaling driven by current widget dimensions

Painting helpers (``_scale_pt``, ``_paint_card``) keep every module
visually consistent regardless of size.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

from PySide6.QtCore import QPoint, QPointF, QRectF, QSettings, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QWidget

from ...lfs_paths import QSETTINGS_APP as APP
from ...lfs_paths import QSETTINGS_ORG as ORG
from ...telemetry.constants import GRAVITY
from ._format import (
    format_clock_ms,
    format_gap_seconds,
    format_signed_delta_ms,
)
from .live_data_source import LiveDataSource
from .racing_line_loader import RacingLine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MIN_W = 60
MIN_H = 40
RESIZE_GRIP_PX = 14


def _fmt_clock(ms: int | None) -> str:
    return format_clock_ms(ms)


def _fmt_delta(ms: int | None) -> str:
    return format_signed_delta_ms(ms)


def _fmt_gap(seconds: float | None) -> str:
    return format_gap_seconds(seconds)


def proximity_color(
    distance_m: float, *, red_m: float, yellow_m: float, white_m: float
) -> QColor:
    """Detect&Monitor / helicorsa proximity ramp."""
    if distance_m <= red_m:
        return QColor(255, 60, 60)
    if distance_m <= yellow_m:
        return QColor(255, 220, 60)
    if distance_m <= white_m:
        return QColor(230, 230, 230)
    return QColor(140, 140, 140)


# ---------------------------------------------------------------------------
# Base window: frameless, top-most, draggable, RESIZABLE
# ---------------------------------------------------------------------------


class _LiveModuleWindow(QWidget):
    """Common chrome + drag/resize behaviour for every overlay module."""

    MODULE_ID: str = ""

    def __init__(
        self,
        source: LiveDataSource,
        *,
        size: tuple[int, int],
        title: str,
        opacity: float = 0.85,
    ) -> None:
        super().__init__()
        self._source = source
        self._snap: dict[str, Any] = source.snapshot
        self._drag_offset: QPoint | None = None
        self._resizing = False
        self._default_size = size
        self._fullscreen_compat = self._load_fullscreen_compat()

        win_kind = (
            Qt.WindowType.Window
            if self._fullscreen_compat
            else Qt.WindowType.Tool
        )
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | win_kind
        )
        if self._fullscreen_compat:
            flags |= Qt.WindowType.WindowDoesNotAcceptFocus
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setMinimumSize(MIN_W, MIN_H)
        self.resize(*size)
        self.setWindowTitle(title)

        # Restore previously-saved geometry + opacity (per module id).
        restored_opacity = self._load_opacity(opacity)
        self.setWindowOpacity(restored_opacity)
        self._restore_geometry()

        source.snapshot_changed.connect(self._on_snapshot)

    # ----- Persistence -------------------------------------------------

    def _settings(self) -> QSettings:
        return QSettings(ORG, APP)

    def _settings_key(self, suffix: str) -> str:
        mid = self.MODULE_ID or self.__class__.__name__
        return f"overlay/{mid}/{suffix}"

    def _load_opacity(self, default: float) -> float:
        if not self.MODULE_ID:
            return default
        raw = self._settings().value(self._settings_key("opacity"), None)
        if raw is None:
            return default
        try:
            return max(0.1, min(1.0, float(raw)))
        except (TypeError, ValueError):
            return default

    def _load_fullscreen_compat(self) -> bool:
        raw = self._settings().value("overlay/fullscreen_compat", True)
        if isinstance(raw, bool):
            return raw
        txt = str(raw).strip().lower()
        return txt in {"1", "true", "yes", "on"}

    def _save_opacity(self) -> None:
        if not self.MODULE_ID:
            return
        self._settings().setValue(
            self._settings_key("opacity"), self.windowOpacity(),
        )

    def _restore_geometry(self) -> None:
        if not self.MODULE_ID:
            return
        geo = self._settings().value(self._settings_key("geometry"))
        if geo is not None:
            with contextlib.suppress(TypeError, ValueError):
                self.restoreGeometry(geo)

    def _save_geometry(self) -> None:
        if not self.MODULE_ID:
            return
        self._settings().setValue(
            self._settings_key("geometry"), self.saveGeometry(),
        )

    # ----- API ---------------------------------------------------------

    def set_opacity_pct(self, pct: int) -> None:
        self.setWindowOpacity(max(0.1, min(1.0, pct / 100.0)))
        self._save_opacity()

    def current_opacity_pct(self) -> int:
        return int(round(self.windowOpacity() * 100))

    def reset_size(self) -> None:
        self.resize(*self._default_size)

    # ----- Drag + resize ----------------------------------------------

    def _in_resize_zone(self, pos: QPoint) -> bool:
        return (
            pos.x() >= self.width() - RESIZE_GRIP_PX
            and pos.y() >= self.height() - RESIZE_GRIP_PX
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            if self._in_resize_zone(event.position().toPoint()):
                self._resizing = True
            else:
                self._drag_offset = (
                    event.globalPosition().toPoint()
                    - self.frameGeometry().topLeft()
                )
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self.reset_size()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._resizing and event.buttons() & Qt.MouseButton.LeftButton:
            local = event.position().toPoint()
            new_w = max(MIN_W, local.x())
            new_h = max(MIN_H, local.y())
            self.resize(new_w, new_h)
            event.accept()
        elif (
            self._drag_offset is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
        else:
            if self._in_resize_zone(event.position().toPoint()):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            else:
                self.unsetCursor()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_offset is not None or self._resizing:
            self._save_geometry()
        self._drag_offset = None
        self._resizing = False
        self.unsetCursor()

    def closeEvent(self, event) -> None:  # noqa: N802
        # Persist the final spot so re-enabling brings the module back
        # to where the user last left it.
        self._save_geometry()
        super().closeEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        if self.isVisible() or self.geometry().isValid():
            self._save_geometry()
        super().hideEvent(event)

    # ----- Data hook ---------------------------------------------------

    def _on_snapshot(self, snap: dict[str, Any]) -> None:
        self._snap = snap
        self.update()

    # ----- Painting helpers -------------------------------------------

    def _paint_card(self, p: QPainter) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(15, 15, 18, 230))
        p.drawRoundedRect(self.rect(), 12, 12)
        p.setPen(QPen(QColor(120, 120, 130, 180), 1))
        x0 = self.width() - 4
        y0 = self.height() - 4
        for d in (3, 6, 9):
            p.drawLine(x0 - d, y0, x0, y0 - d)

    def _scale_pt(self, base_pt: int, ref_dim: int = 160) -> int:
        cur = min(self.width(), self.height())
        return max(6, int(round(base_pt * cur / ref_dim)))

    def _font(self, base_pt: int, weight=QFont.Weight.Bold,
              family: str = "Segoe UI") -> QFont:
        return QFont(family, self._scale_pt(base_pt), weight)


# ---------------------------------------------------------------------------
# Generic LABEL + VALUE module (used by most atomic modules)
# ---------------------------------------------------------------------------


class _LabeledValueWindow(_LiveModuleWindow):
    LABEL: str = ""
    DEFAULT_SIZE: tuple[int, int] = (140, 80)

    def __init__(
        self, source: LiveDataSource, *, opacity: float = 0.85,
    ) -> None:
        super().__init__(
            source,
            size=self.DEFAULT_SIZE,
            title=f"LFS Live - {self.LABEL or self.__class__.__name__}",
            opacity=opacity,
        )

    def _value_text(self) -> str:
        return "--"

    def _value_color(self) -> QColor:
        return QColor(235, 235, 245)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)
        p.setPen(QPen(QColor(150, 150, 160)))
        p.setFont(self._font(11, QFont.Weight.Normal))
        p.drawText(
            QRectF(8, 4, self.width() - 16, self.height() * 0.30),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self.LABEL,
        )
        p.setPen(QPen(self._value_color()))
        p.setFont(QFont(
            "Consolas", self._scale_pt(28), QFont.Weight.Bold,
        ))
        p.drawText(
            QRectF(8, self.height() * 0.28,
                   self.width() - 16, self.height() * 0.70),
            int(Qt.AlignmentFlag.AlignCenter),
            self._value_text(),
        )


# ---------------------------------------------------------------------------
# Atomic value modules
# ---------------------------------------------------------------------------


class PositionWindow(_LabeledValueWindow):
    LABEL = "POS"
    DEFAULT_SIZE = (110, 90)

    def _value_text(self) -> str:
        pos = self._snap.get("view_position")
        return f"P{pos}" if pos else "--"


class FuelPctWindow(_LabeledValueWindow):
    MODULE_ID = "fuel_pct"
    LABEL = "FUEL"
    DEFAULT_SIZE = (140, 80)

    def _value_text(self) -> str:
        f = self._snap.get("view_fuel_pct")
        return f"{f:5.1f}%" if f is not None else "--"

    def _value_color(self) -> QColor:
        f = self._snap.get("view_fuel_pct")
        if f is None:
            return QColor(220, 220, 230)
        if f < 5:
            return QColor(255, 80, 80)
        if f < 15:
            return QColor(255, 220, 60)
        return QColor(220, 220, 230)


class FuelLapsRemainingWindow(_LabeledValueWindow):
    MODULE_ID = "fuel_laps"
    LABEL = "LAPS LEFT"
    DEFAULT_SIZE = (170, 80)

    def _value_text(self) -> str:
        n = self._snap.get("fuel_laps_remaining")
        return f"{n:4.1f}" if n is not None else "--"

    def _value_color(self) -> QColor:
        n = self._snap.get("fuel_laps_remaining")
        if n is None:
            return QColor(220, 220, 230)
        if n < 1.0:
            return QColor(255, 80, 80)
        if n < 2.0:
            return QColor(255, 220, 60)
        return QColor(220, 220, 230)


class SpeedWindow(_LabeledValueWindow):
    MODULE_ID = "speed"
    LABEL = "SPEED"
    DEFAULT_SIZE = (160, 80)

    def _value_text(self) -> str:
        v = self._snap.get("view_speed_kmh")
        return f"{v:5.1f}" if v is not None else "--"


class GearWindow(_LiveModuleWindow):
    MODULE_ID = "gear"
    """Big gear digit (no label)."""

    def __init__(
        self, source: LiveDataSource, *, opacity: float = 0.85,
    ) -> None:
        super().__init__(
            source, size=(110, 130),
            title="LFS Live - gear", opacity=opacity,
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)
        gear = self._snap.get("view_gear")
        if gear is None:
            text = "-"
        elif gear == 0:
            text = "R"
        elif gear == 1:
            text = "N"
        else:
            text = str(gear - 1)
        p.setPen(QPen(QColor(235, 235, 245)))
        p.setFont(QFont(
            "Segoe UI", self._scale_pt(72), QFont.Weight.Black,
        ))
        p.drawText(
            self.rect(), int(Qt.AlignmentFlag.AlignCenter), text,
        )


class RpmWindow(_LiveModuleWindow):
    MODULE_ID = "rpm"
    """Horizontal RPM bar + numeric readout."""

    def __init__(
        self, source: LiveDataSource, *,
        opacity: float = 0.85, redline: float = 8000.0,
    ) -> None:
        super().__init__(
            source, size=(220, 80),
            title="LFS Live - rpm", opacity=opacity,
        )
        self._redline = max(2000.0, float(redline))

    def set_rpm_redline(self, rpm: float) -> None:
        self._redline = max(2000.0, float(rpm))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)
        rpm = self._snap.get("view_rpm")
        m = 10
        bar = QRectF(m, m, self.width() - 2 * m, self.height() * 0.45)
        p.setPen(QPen(QColor(80, 80, 90), 1))
        p.setBrush(QColor(28, 28, 34, 220))
        p.drawRoundedRect(bar, 4, 4)
        if rpm:
            frac = max(0.0, min(1.0, float(rpm) / self._redline))
            fill = QRectF(bar.left() + 1, bar.top() + 1,
                          (bar.width() - 2) * frac, bar.height() - 2)
            grad = QLinearGradient(fill.left(), 0, fill.right(), 0)
            grad.setColorAt(0.0, QColor(120, 230, 140))
            grad.setColorAt(0.7, QColor(255, 220, 60))
            grad.setColorAt(1.0, QColor(255, 60, 60))
            p.setBrush(QBrush(grad))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(fill, 3, 3)
        p.setPen(QPen(QColor(220, 220, 230)))
        p.setFont(QFont(
            "Consolas", self._scale_pt(18), QFont.Weight.Bold,
        ))
        text = f"{int(rpm) if rpm else 0:>5d} RPM"
        p.drawText(
            QRectF(m, bar.bottom(), self.width() - 2 * m,
                   self.height() - bar.bottom() - m),
            int(Qt.AlignmentFlag.AlignCenter), text,
        )


# ---------------------------------------------------------------------------
# Pedal bars
# ---------------------------------------------------------------------------


class _PedalWindow(_LiveModuleWindow):
    LABEL = ""
    KEY = ""
    BAR_COLOR = QColor(220, 220, 220)

    def __init__(
        self, source: LiveDataSource, *, opacity: float = 0.85,
    ) -> None:
        super().__init__(
            source, size=(70, 160),
            title=f"LFS Live - {self.LABEL}", opacity=opacity,
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)
        m = 8
        label_h = max(14, int(self.height() * 0.16))
        p.setPen(QPen(QColor(170, 170, 180)))
        p.setFont(self._font(11, QFont.Weight.Normal))
        p.drawText(
            QRectF(m, 2, self.width() - 2 * m, label_h),
            int(Qt.AlignmentFlag.AlignCenter), self.LABEL,
        )
        track = QRectF(
            m, label_h + 4, self.width() - 2 * m,
            self.height() - label_h - m - 6,
        )
        p.setPen(QPen(QColor(80, 80, 90), 1))
        p.setBrush(QColor(28, 28, 34, 220))
        p.drawRoundedRect(track, 4, 4)
        v = self._snap.get(self.KEY)
        frac = max(0.0, min(1.0, float(v))) if v is not None else 0.0
        if frac > 0:
            fh = (track.height() - 2) * frac
            fill = QRectF(
                track.left() + 1, track.bottom() - 1 - fh,
                track.width() - 2, fh,
            )
            p.setBrush(self.BAR_COLOR)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(fill, 3, 3)
        pct = int(round(frac * 100.0))
        p.setPen(QPen(QColor(235, 235, 245)))
        p.setFont(QFont(
            "Consolas", self._scale_pt(12), QFont.Weight.Bold,
        ))
        p.drawText(
            QRectF(m, track.bottom() - 18, track.width(), 16),
            int(Qt.AlignmentFlag.AlignCenter),
            f"{pct:3d}%",
        )


class ThrottleWindow(_PedalWindow):
    LABEL = "THR"
    KEY = "view_throttle"
    BAR_COLOR = QColor(80, 220, 120)


class BrakeWindow(_PedalWindow):
    LABEL = "BRK"
    KEY = "view_brake"
    BAR_COLOR = QColor(255, 80, 80)


class ClutchWindow(_PedalWindow):
    LABEL = "CLU"
    KEY = "view_clutch"
    BAR_COLOR = QColor(140, 180, 255)


# ---------------------------------------------------------------------------
# Gap modules
# ---------------------------------------------------------------------------


class _GapWindow(_LiveModuleWindow):
    DIRECTION = "ahead"
    LABEL = "AHEAD"

    def __init__(
        self, source: LiveDataSource, *, opacity: float = 0.85,
    ) -> None:
        super().__init__(
            source, size=(200, 90),
            title=f"LFS Live - gap {self.DIRECTION}", opacity=opacity,
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)
        traffic = self._snap.get("traffic") or {}
        gap_s = traffic.get(f"{self.DIRECTION}_gap_s")
        gap_m = traffic.get(f"{self.DIRECTION}_gap_m")
        pos = traffic.get(f"{self.DIRECTION}_pos")
        p.setPen(QPen(QColor(150, 150, 160)))
        p.setFont(self._font(11, QFont.Weight.Normal))
        head = self.LABEL + (f"  P{pos}" if pos else "")
        p.drawText(
            QRectF(8, 4, self.width() - 16, self.height() * 0.28),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            head,
        )
        p.setPen(QPen(QColor(235, 235, 245) if gap_s is not None
                      else QColor(180, 180, 190)))
        p.setFont(QFont(
            "Consolas", self._scale_pt(22), QFont.Weight.Bold,
        ))
        p.drawText(
            QRectF(8, self.height() * 0.25,
                   self.width() - 16, self.height() * 0.45),
            int(Qt.AlignmentFlag.AlignCenter),
            _fmt_gap(gap_s),
        )
        p.setPen(QPen(QColor(180, 180, 190)))
        p.setFont(QFont("Consolas", self._scale_pt(10)))
        m_txt = f"{gap_m:6.1f} m" if gap_m is not None else "--"
        p.drawText(
            QRectF(8, self.height() * 0.70,
                   self.width() - 16, self.height() * 0.30),
            int(Qt.AlignmentFlag.AlignCenter), m_txt,
        )


class GapAheadWindow(_GapWindow):
    MODULE_ID = "gap_ahead"
    DIRECTION = "ahead"
    LABEL = "AHEAD"


class GapBehindWindow(_GapWindow):
    MODULE_ID = "gap_behind"
    DIRECTION = "behind"
    LABEL = "BEHIND"


class TyreRiskWindow(_LiveModuleWindow):
    """Per-wheel grip panel for race/long-run management."""

    MODULE_ID = "grip"

    def __init__(
        self, source: LiveDataSource, *, opacity: float = 0.85,
    ) -> None:
        super().__init__(
            source,
            size=(280, 180),
            title="LFS Live - grip",
            opacity=opacity,
        )
        self._grip_smoothed: dict[str, float] = {
            "FL": 0.0,
            "FR": 0.0,
            "RL": 0.0,
            "RR": 0.0,
        }
        self._grip_short: dict[str, float] = {
            "FL": 1.0,
            "FR": 1.0,
            "RL": 1.0,
            "RR": 1.0,
        }
        self._grip_long: dict[str, float] = {
            "FL": 1.0,
            "FR": 1.0,
            "RL": 1.0,
            "RR": 1.0,
        }
        self._trend: dict[str, float] = {
            "FL": 0.0,
            "FR": 0.0,
            "RL": 0.0,
            "RR": 0.0,
        }
        self._state_hold: dict[str, dict[str, int]] = {
            "FL": {"S": 0, "L": 0},
            "FR": {"S": 0, "L": 0},
            "RL": {"S": 0, "L": 0},
            "RR": {"S": 0, "L": 0},
        }
        # 10 Hz snapshot updates -> alpha 0.25 is responsive
        # without flicker.
        self._ema_alpha = 0.25
        self._short_alpha = 0.35
        self._long_alpha = 0.06
        # Keep S/L LEDs lit briefly so fast events are readable.
        self._state_hold_ticks = 4

    @staticmethod
    def _wheel_states(
        row: dict[str, Any],
        *,
        speed_kmh: float,
        brake: float,
        handbrake: float,
    ) -> tuple[bool, bool]:
        """Return (sliding, locked) inferred from live tyre telemetry."""
        touching = bool(row.get("touching", False))
        load_n = float(row.get("load_n") or 0.0)
        if not touching or load_n < 50.0:
            return (False, False)

        slip_frac = abs(float(row.get("slip_frac") or 0.0))
        slip_ratio = abs(float(row.get("slip_ratio") or 0.0))
        tan_slip = abs(float(row.get("tan_slip") or 0.0))

        # S = lateral/combined sliding event.
        sliding = (
            tan_slip >= 0.10
            or (slip_frac >= 0.16 and tan_slip >= 0.06)
            or (slip_frac >= 0.22 and slip_ratio >= 0.10)
        )

        # L = wheel lock tendency while braking (heuristic, no direct
        # lock bit from this telemetry stream).
        braking = max(0.0, float(brake))
        hb = max(0.0, float(handbrake))
        locked = (
            speed_kmh >= 25.0
            and slip_ratio >= 0.22
            and tan_slip <= 0.11
            and (braking >= 0.18 or hb >= 0.20)
        )

        return (sliding, locked)

    def _on_snapshot(self, snap: dict[str, Any]) -> None:
        rows = snap.get("tyres")
        tyres = rows if isinstance(rows, list) else []
        by_corner = {
            str(r.get("corner") or "?"): r
            for r in tyres if isinstance(r, dict)
        }
        speed_kmh = float(snap.get("view_speed_kmh") or 0.0)
        brake = float(snap.get("view_brake") or 0.0)
        handbrake = float(snap.get("view_handbrake") or 0.0)
        for corner in ("FL", "FR", "RL", "RR"):
            hold = self._state_hold.setdefault(corner, {"S": 0, "L": 0})
            hold["S"] = max(0, int(hold.get("S", 0)) - 1)
            hold["L"] = max(0, int(hold.get("L", 0)) - 1)

            row = by_corner.get(corner)
            raw_risk = (
                self._risk_level(row)
                if isinstance(row, dict) else 0.0
            )
            raw_grip = max(0.0, min(1.0, 1.0 - raw_risk))

            # Demand weighting: trend is only meaningful when tyres are
            # actually loaded (cornering/braking/traction).
            demand = 0.0
            if isinstance(row, dict):
                load_n = float(row.get("load_n") or 0.0)
                fx_n = float(row.get("fx_n") or 0.0)
                fy_n = float(row.get("fy_n") or 0.0)
                if load_n > 50.0:
                    demand_ratio = math.hypot(fx_n, fy_n) / load_n
                    demand = max(0.0, min(1.0, (demand_ratio - 1.5) / 7.5))

                sliding_now, locked_now = self._wheel_states(
                    row,
                    speed_kmh=speed_kmh,
                    brake=brake,
                    handbrake=handbrake,
                )
                if sliding_now:
                    hold["S"] = self._state_hold_ticks
                if locked_now:
                    hold["L"] = self._state_hold_ticks

            prev_s = self._grip_short.get(corner, raw_grip)
            prev_l = self._grip_long.get(corner, raw_grip)
            self._grip_short[corner] = (
                prev_s + self._short_alpha * (raw_grip - prev_s)
            )
            self._grip_long[corner] = (
                prev_l + self._long_alpha * (raw_grip - prev_l)
            )
            self._trend[corner] = (
                (self._grip_short[corner] - self._grip_long[corner]) * demand
            )

            prev = self._grip_smoothed.get(corner, raw_grip)
            a = self._ema_alpha
            blended = max(
                0.0,
                min(
                    1.0,
                    0.70 * self._grip_short[corner]
                    + 0.30 * self._grip_long[corner],
                ),
            )
            self._grip_smoothed[corner] = prev + a * (blended - prev)
        self._snap = snap
        self.update()

    @staticmethod
    def _risk_level(row: dict[str, Any]) -> float:
        touching = bool(row.get("touching", False))
        load_n = float(row.get("load_n") or 0.0)
        if not touching or load_n < 50.0:
            return 0.0
        slip_frac = abs(float(row.get("slip_frac") or 0.0))
        slip_ratio = abs(float(row.get("slip_ratio") or 0.0))
        tan_slip = abs(float(row.get("tan_slip") or 0.0))
        fx_n = float(row.get("fx_n") or 0.0)
        fy_n = float(row.get("fy_n") or 0.0)
        temp_c = float(row.get("temp_c") or 0.0)

        # Core degradation proxies.
        slip_frac_term = min(1.0, slip_frac / 0.25)
        slip_ratio_term = min(1.0, slip_ratio / 0.18)
        slip_angle_term = min(1.0, tan_slip / 0.14)

        # Demand/saturation proxy from tyre force magnitude over load.
        # Values around ~1 g should not imply degradation by themselves.
        force_ratio = math.hypot(fx_n, fy_n) / max(1.0, load_n)
        sat_term = min(1.0, max(0.0, (force_ratio - 4.0) / 8.0))

        # Temperature contributes but with lower weight (slow/indirect).
        temp_term = min(1.0, max(0.0, (temp_c - 102.0) / 30.0))

        score = (
            0.34 * slip_frac_term
            + 0.26 * slip_ratio_term
            + 0.18 * slip_angle_term
            + 0.14 * sat_term
            + 0.08 * temp_term
        )
        return score

    @staticmethod
    def _grip_color(grip: float, *, touching: bool) -> QColor:
        if not touching:
            return QColor(120, 120, 130)
        if grip >= 0.65:
            return QColor(120, 230, 140)
        if grip >= 0.35:
            return QColor(255, 220, 60)
        return QColor(255, 90, 90)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)

        rows = self._snap.get("tyres")
        tyres = rows if isinstance(rows, list) else []
        by_corner = {
            str(r.get("corner") or "?"): r
            for r in tyres if isinstance(r, dict)
        }
        order = ("FL", "FR", "RL", "RR")

        p.setPen(QPen(QColor(150, 150, 160)))
        p.setFont(self._font(10, QFont.Weight.Normal))
        p.drawText(
            QRectF(10, 6, self.width() - 20, 18),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            "GRIP",
        )

        if not by_corner:
            p.setPen(QPen(QColor(180, 180, 190)))
            p.setFont(
                QFont("Consolas", self._scale_pt(10), QFont.Weight.Normal)
            )
            p.drawText(
                QRectF(10, 30, self.width() - 20, self.height() - 40),
                int(Qt.AlignmentFlag.AlignCenter),
                "no wheel telemetry",
            )
            return

        card_w = (self.width() - 30) / 2.0
        card_h = (self.height() - 44) / 2.0
        lowest_corner = "--"
        lowest_grip = 1e9

        for idx, corner in enumerate(order):
            row = by_corner.get(corner) or {}
            grip = self._grip_smoothed.get(corner, 0.0)
            col = self._grip_color(
                grip,
                touching=bool(row.get("touching", False)),
            )
            if grip < lowest_grip:
                lowest_grip = grip
                lowest_corner = corner

            r = idx // 2
            c = idx % 2
            x = 10 + c * (card_w + 10)
            y = 26 + r * (card_h + 8)
            rect = QRectF(x, y, card_w, card_h)

            p.setPen(QPen(QColor(60, 60, 70), 1))
            p.setBrush(QColor(24, 24, 30, 220))
            p.drawRoundedRect(rect, 6, 6)

            side = min(rect.width(), rect.height()) - 22
            cx = rect.left() + rect.width() / 2.0
            cy = rect.top() + rect.height() * 0.47
            circle = QRectF(cx - side / 2.0, cy - side / 2.0, side, side)
            p.setPen(QPen(QColor(30, 30, 36), 2))
            p.setBrush(QColor(34, 34, 40))
            p.drawEllipse(circle)
            p.setPen(QPen(QColor(20, 20, 24), 1))
            p.setBrush(col)
            p.drawEllipse(circle.adjusted(2, 2, -2, -2))

            p.setPen(QPen(QColor(170, 170, 180)))
            p.setFont(self._font(8, QFont.Weight.Normal))
            p.drawText(
                QRectF(
                    rect.left() + 6,
                    rect.top() + 2,
                    rect.width() - 12,
                    12,
                ),
                int(
                    Qt.AlignmentFlag.AlignLeft
                    | Qt.AlignmentFlag.AlignVCenter
                ),
                corner,
            )

            # S/L realtime state badges (Sliding / Locked).
            hold = self._state_hold.get(corner, {})
            led_h = max(10.0, min(14.0, rect.height() * 0.24))
            led_w = led_h + 2.0
            y_led = rect.top() + 3.0
            x_l = rect.right() - 6.0 - led_w
            x_s = x_l - 3.0 - led_w

            for ch, x_led, active_col, key in (
                ("S", x_s, QColor(255, 180, 70), "S"),
                ("L", x_l, QColor(255, 95, 95), "L"),
            ):
                active = int(hold.get(key, 0)) > 0
                led_rect = QRectF(x_led, y_led, led_w, led_h)
                p.setPen(QPen(QColor(28, 28, 32), 1))
                p.setBrush(active_col if active else QColor(48, 48, 56))
                p.drawRoundedRect(led_rect, 3, 3)
                p.setPen(QPen(QColor(20, 20, 24) if active
                              else QColor(155, 155, 165)))
                p.setFont(QFont("Consolas", self._scale_pt(7),
                                QFont.Weight.Bold))
                p.drawText(
                    led_rect,
                    int(Qt.AlignmentFlag.AlignCenter),
                    ch,
                )

            pct = int(round(grip * 100.0))
            p.setPen(QPen(QColor(15, 15, 18)))
            p.setFont(QFont("Consolas", self._scale_pt(11), QFont.Weight.Bold))
            p.drawText(
                circle,
                int(Qt.AlignmentFlag.AlignCenter),
                f"{pct}%",
            )

            temp_c = row.get("temp_c")
            temp_txt = f"{float(temp_c):.0f}°C" if temp_c is not None else "--"
            tr = self._trend.get(corner, 0.0)
            if tr > 0.01:
                trend_txt = "↗"
            elif tr < -0.01:
                trend_txt = "↘"
            else:
                trend_txt = "→"

            p.setPen(QPen(QColor(190, 190, 205)))
            p.setFont(
                QFont("Consolas", self._scale_pt(9), QFont.Weight.Normal)
            )
            p.drawText(
                QRectF(
                    rect.left() + 6,
                    rect.bottom() - 16,
                    rect.width() - 12,
                    12,
                ),
                int(Qt.AlignmentFlag.AlignCenter),
                f"T {temp_txt}  {trend_txt}",
            )

        p.setPen(QPen(QColor(140, 140, 150)))
        p.setFont(self._font(8, QFont.Weight.Normal))
        p.drawText(
            QRectF(10, self.height() - 16, self.width() - 20, 12),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            f"low: {lowest_corner}",
        )


class SessionInfoWindow(_LiveModuleWindow):
    """Compact dynamic session summary for the floating overlay."""

    MODULE_ID = "session_info"

    def __init__(
        self, source: LiveDataSource, *, opacity: float = 0.85,
    ) -> None:
        super().__init__(
            source,
            size=(360, 200),
            title="LFS Live - session info",
            opacity=opacity,
        )
        self._compact = self._load_compact_mode(default=False)

    def _load_compact_mode(self, *, default: bool) -> bool:
        raw = self._settings().value(self._settings_key("compact"), None)
        if raw is None:
            return default
        if isinstance(raw, bool):
            return raw
        txt = str(raw).strip().lower()
        return txt in {"1", "true", "yes", "on"}

    def _save_compact_mode(self) -> None:
        self._settings().setValue(
            self._settings_key("compact"), bool(self._compact),
        )

    def set_compact_mode(self, on: bool) -> None:
        self._compact = bool(on)
        self._save_compact_mode()
        # Re-trigger sizing logic so switching to detailed mode grows
        # the window immediately (without waiting for the next 10 Hz
        # snapshot) and switching back to compact restores the small
        # default footprint.
        if self._compact:
            self.resize(self.width(), self._DETAILED_MIN_H)
        else:
            self._on_snapshot(self._snap)
        self.update()

    def compact_mode(self) -> bool:
        return bool(self._compact)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.set_compact_mode(not self._compact)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    # Layout constants for the detailed leaderboard: top reserved for
    # SESSION / POS / LAP / times / AHEAD-BEHIND / "LEADERBOARD" header.
    _LEADERBOARD_TOP_Y = 110
    _LEADERBOARD_ROW_PX = 14
    _LEADERBOARD_BOTTOM_PAD = 24  # shortcut hint
    _DETAILED_MIN_H = 200
    _DETAILED_MAX_H = 720

    def _on_snapshot(self, snap: dict[str, Any]) -> None:
        # Resize the window so every classified driver fits when in
        # detailed mode. Compact mode keeps its fixed size so users
        # who pin it as a small HUD don't see it grow unexpectedly.
        if not self._compact:
            standings = snap.get("standings")
            n = len(standings) if isinstance(standings, list) else 0
            needed = (
                self._LEADERBOARD_TOP_Y
                + max(1, n) * self._LEADERBOARD_ROW_PX
                + self._LEADERBOARD_BOTTOM_PAD
            )
            needed = max(self._DETAILED_MIN_H,
                         min(self._DETAILED_MAX_H, needed))
            if needed != self.height():
                self.resize(self.width(), needed)
        super()._on_snapshot(snap)

    def _mode_text(self) -> str:
        mode = str(self._snap.get("session_mode") or "practice")
        if mode == "race":
            return "RACE"
        if mode == "qualifying":
            return "QUALIFYING"
        return "PRACTICE"

    def _paint_shortcut_hint(self, p: QPainter) -> None:
        p.setPen(QPen(QColor(130, 130, 140)))
        p.setFont(QFont("Segoe UI", self._scale_pt(8), QFont.Weight.Normal))
        p.drawText(
            QRectF(10, self.height() - 18, self.width() - 20, 12),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            "double-click: compact / detailed",
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)

        mode_txt = self._mode_text()
        pos = self._snap.get("view_position")
        lap = self._snap.get("view_lap")
        cars = self._snap.get("cars") or []
        n_cars = len(cars)
        traffic = self._snap.get("traffic") or {}
        if isinstance(traffic, dict):
            n_from_traffic = traffic.get("num_cars")
            if isinstance(n_from_traffic, int) and n_from_traffic > 0:
                n_cars = n_from_traffic

        p.setPen(QPen(QColor(150, 150, 160)))
        p.setFont(self._font(10, QFont.Weight.Normal))
        p.drawText(
            QRectF(10, 6, self.width() - 20, 20),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"SESSION  {mode_txt}",
        )

        pos_txt = "--" if pos is None else (
            f"P{int(pos)}" if n_cars <= 0 else f"P{int(pos)}/{n_cars}"
        )
        lap_txt = "--" if lap is None else str(int(lap))
        p.setPen(QPen(QColor(235, 235, 245)))
        p.setFont(QFont("Consolas", self._scale_pt(17), QFont.Weight.Bold))
        p.drawText(
            QRectF(10, 24, self.width() - 20, 30),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"{pos_txt}   LAP {lap_txt}",
        )

        ahead = _fmt_gap(traffic.get("ahead_gap_s")) if traffic else "--"
        behind = _fmt_gap(traffic.get("behind_gap_s")) if traffic else "--"

        if self._compact:
            delta = _fmt_delta(self._snap.get("delta_vs_best_ms"))
            p.setPen(QPen(QColor(205, 205, 215)))
            p.setFont(
                QFont("Consolas", self._scale_pt(10), QFont.Weight.Normal)
            )
            p.drawText(
                QRectF(10, 56, self.width() - 20, 18),
                int(
                    Qt.AlignmentFlag.AlignLeft
                    | Qt.AlignmentFlag.AlignVCenter
                ),
                f"DELTA {delta}   A {ahead}   B {behind}",
            )
            top = None
            standings = self._snap.get("standings")
            if isinstance(standings, list) and standings:
                top = standings[0]
            if isinstance(top, dict):
                name = str(top.get("name") or "?")[:18]
                p.drawText(
                    QRectF(10, 74, self.width() - 20, 18),
                    int(
                        Qt.AlignmentFlag.AlignLeft
                        | Qt.AlignmentFlag.AlignVCenter
                    ),
                    f"LEAD {name}",
                )
            self._paint_shortcut_hint(p)
            return

        p.setPen(QPen(QColor(205, 205, 215)))
        p.setFont(QFont("Consolas", self._scale_pt(10), QFont.Weight.Normal))
        current = _fmt_clock(self._snap.get("current_lap_ms"))
        last = _fmt_clock(self._snap.get("last_lap_ms"))
        best = _fmt_clock(self._snap.get("best_lap_ms"))
        p.drawText(
            QRectF(10, 56, self.width() - 20, 18),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"CUR {current}   LAST {last}   BEST {best}",
        )

        p.drawText(
            QRectF(10, 74, self.width() - 20, 18),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"AHEAD {ahead}   BEHIND {behind}",
        )

        p.setPen(QPen(QColor(150, 150, 160)))
        p.setFont(self._font(9, QFont.Weight.Normal))
        p.drawText(
            QRectF(10, 94, self.width() - 20, 16),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            "LEADERBOARD",
        )

        standings = self._snap.get("standings")
        if not isinstance(standings, list) or not standings:
            p.setPen(QPen(QColor(180, 180, 190)))
            p.setFont(
                QFont("Consolas", self._scale_pt(10), QFont.Weight.Normal)
            )
            p.drawText(
                QRectF(10, 112, self.width() - 20, self.height() - 120),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
                "-- no classification yet --",
            )
            return

        lines: list[str] = []
        for row in standings:
            try:
                rpos = int(row.get("pos", 0))
                name = str(row.get("name") or "?")[:14]
                mark = ">" if bool(row.get("view")) else " "
                if str(self._snap.get("session_mode") or "practice") == "race":
                    val = _fmt_clock(row.get("last_lap_ms"))
                    tail = f"L{int(row.get('lap', 0)):>2}  {val}"
                else:
                    val = _fmt_clock(row.get("best_lap_ms"))
                    tail = f"BEST {val}"
                lines.append(f"{mark}{rpos:>2} {name:<14} {tail}")
            except (TypeError, ValueError):
                continue

        p.setPen(QPen(QColor(220, 220, 230)))
        p.setFont(QFont("Consolas", self._scale_pt(10), QFont.Weight.Normal))
        p.drawText(
            QRectF(10, 110, self.width() - 20, self.height() - 118),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
            "\n".join(lines) if lines else "--",
        )
        self._paint_shortcut_hint(p)


# ---------------------------------------------------------------------------
# Flags + TC/ABS LED
# ---------------------------------------------------------------------------


class FlagsWindow(_LiveModuleWindow):
    MODULE_ID = "flags"
    """Big BLUE/YELLOW flag indicator."""

    def __init__(
        self, source: LiveDataSource, *, opacity: float = 0.85,
    ) -> None:
        super().__init__(
            source, size=(160, 90),
            title="LFS Live - flags", opacity=opacity,
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)
        traffic = self._snap.get("traffic") or {}
        blue = bool(traffic.get("blue_flag"))
        yellow = bool(traffic.get("yellow_flag"))
        m = 8
        half_w = (self.width() - 3 * m) / 2
        rect_b = QRectF(m, m, half_w, self.height() - 2 * m)
        rect_y = QRectF(m + half_w + m, m, half_w, self.height() - 2 * m)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(60, 110, 220) if blue else QColor(40, 40, 50))
        p.drawRoundedRect(rect_b, 6, 6)
        p.setBrush(QColor(255, 220, 60) if yellow else QColor(40, 40, 50))
        p.drawRoundedRect(rect_y, 6, 6)
        p.setPen(QPen(QColor(20, 20, 28)))
        p.setFont(self._font(12, QFont.Weight.Bold))
        p.drawText(rect_b, int(Qt.AlignmentFlag.AlignCenter), "BLUE")
        p.drawText(rect_y, int(Qt.AlignmentFlag.AlignCenter), "YELLOW")


class PitLimiterWindow(_LiveModuleWindow):
    MODULE_ID = "pit_limiter"
    """Flashing band + speed-vs-limit delta while the pit limiter is on.

    Reads ``view_pit_limiter`` (OutGauge ``show_lights & DL_PITSPEED``)
    and ``view_speed_kmh`` from the live snapshot. The pit-lane speed
    limit defaults to 80 km/h (LFS standard) and is persisted per-user
    via ``QSettings`` so each driver can tune it for tracks that differ.
    """

    DEFAULT_LIMIT_KMH = 80.0

    def __init__(
        self, source: LiveDataSource, *, opacity: float = 0.85,
    ) -> None:
        super().__init__(
            source, size=(240, 90),
            title="LFS Live - pit limiter", opacity=opacity,
        )
        raw = self._settings().value(
            "overlay/pit_limiter/limit_kmh", self.DEFAULT_LIMIT_KMH,
        )
        try:
            self._limit_kmh = max(20.0, min(200.0, float(raw)))
        except (TypeError, ValueError):
            self._limit_kmh = self.DEFAULT_LIMIT_KMH

    def set_limit_kmh(self, value: float) -> None:
        self._limit_kmh = max(20.0, min(200.0, float(value)))
        self._settings().setValue(
            "overlay/pit_limiter/limit_kmh", self._limit_kmh,
        )
        self.update()

    def limit_kmh(self) -> float:
        return self._limit_kmh

    def paintEvent(self, event) -> None:  # noqa: N802
        import time as _time

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)

        active = bool(self._snap.get("view_pit_limiter"))
        speed = self._snap.get("view_speed_kmh")
        limit = float(self._limit_kmh)
        delta = (
            float(speed) - limit if isinstance(speed, (int, float)) else None
        )

        m = 6
        band = QRectF(m, m, self.width() - 2 * m, self.height() * 0.42)
        if active:
            # Blink ~2.5 Hz: phase toggles every 200 ms based on wall
            # clock so the flash keeps animating regardless of snapshot
            # cadence.
            phase = int(_time.monotonic() * 2.5) % 2
            if delta is not None and delta > 1.0:
                base = QColor(220, 40, 40) if phase else QColor(120, 20, 20)
            else:
                base = (
                    QColor(255, 200, 40) if phase else QColor(150, 110, 20)
                )
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(base)
            p.drawRoundedRect(band, 6, 6)
            p.setPen(QPen(QColor(20, 20, 24)))
            p.setFont(self._font(13, QFont.Weight.Black))
            p.drawText(
                band, int(Qt.AlignmentFlag.AlignCenter), "PIT LIMITER",
            )
        else:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(40, 40, 50))
            p.drawRoundedRect(band, 6, 6)
            p.setPen(QPen(QColor(120, 120, 130)))
            p.setFont(self._font(12, QFont.Weight.Bold))
            p.drawText(
                band, int(Qt.AlignmentFlag.AlignCenter), "PIT LIMITER OFF",
            )

        # ----- Bottom row: SPEED / LIMIT and DELTA --------------------
        row_top = self.height() * 0.50
        row_h = self.height() - row_top - 4
        left = QRectF(m, row_top, (self.width() - 2 * m) * 0.55, row_h)
        right = QRectF(
            left.right() + 4, row_top,
            self.width() - m - left.right() - 4, row_h,
        )

        if isinstance(speed, (int, float)):
            speed_txt = f"{speed:5.1f} / {limit:3.0f}"
        else:
            speed_txt = f"-- / {limit:3.0f}"
        p.setPen(QPen(QColor(220, 220, 230)))
        p.setFont(QFont(
            "Consolas", self._scale_pt(15), QFont.Weight.Bold,
        ))
        p.drawText(
            left,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            speed_txt,
        )

        if delta is None:
            delta_txt = "--"
            delta_col = QColor(180, 180, 190)
        else:
            sign = "+" if delta >= 0 else ""
            delta_txt = f"{sign}{delta:4.1f}"
            if delta > 1.0:
                delta_col = QColor(255, 90, 90)
            elif delta > -1.0:
                delta_col = QColor(255, 220, 60)
            else:
                delta_col = QColor(120, 230, 140)
        p.setPen(QPen(delta_col))
        p.setFont(QFont(
            "Consolas", self._scale_pt(18), QFont.Weight.Black,
        ))
        p.drawText(
            right,
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            delta_txt,
        )

    def _on_snapshot(self, snap: dict[str, Any]) -> None:
        # Force redraw at the snapshot cadence so the flash phase keeps
        # animating even when speed/limiter values are unchanged.
        super()._on_snapshot(snap)


class TcAbsWindow(_LiveModuleWindow):
    """LED that lights when wheel slip exceeds a threshold."""

    def __init__(
        self, source: LiveDataSource, *,
        opacity: float = 0.85, slip_threshold: float = 0.20,
    ) -> None:
        super().__init__(
            source, size=(120, 100),
            title="LFS Live - slip", opacity=opacity,
        )
        self._threshold = float(slip_threshold)

    def set_slip_threshold(self, value: float) -> None:
        self._threshold = float(value)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)
        slip = self._snap.get("view_max_slip")
        active = slip is not None and slip >= self._threshold
        side = min(self.width(), self.height()) * 0.45
        cx = self.width() / 2.0
        cy = self.height() * 0.45
        led = QRectF(cx - side / 2, cy - side / 2, side, side)
        p.setPen(QPen(QColor(40, 40, 50), 2))
        p.setBrush(QColor(255, 80, 80) if active else QColor(50, 30, 30))
        p.drawEllipse(led)
        p.setPen(QPen(QColor(220, 220, 230)))
        p.setFont(self._font(11, QFont.Weight.Bold))
        p.drawText(
            QRectF(0, self.height() * 0.78, self.width(),
                   self.height() * 0.22),
            int(Qt.AlignmentFlag.AlignCenter),
            f"SLIP {slip * 100:4.1f}%" if slip is not None else "SLIP --",
        )


# ---------------------------------------------------------------------------
# G-meter
# ---------------------------------------------------------------------------


class GMeterWindow(_LiveModuleWindow):
    MODULE_ID = "gmeter"
    """Lateral/longitudinal G dot inside a friction circle."""

    def __init__(
        self, source: LiveDataSource, *,
        opacity: float = 0.85, full_scale_g: float = 2.0,
    ) -> None:
        super().__init__(
            source, size=(180, 180),
            title="LFS Live - g-meter", opacity=opacity,
        )
        self._full_scale_g = max(0.5, float(full_scale_g))

    def set_full_scale_g(self, g: float) -> None:
        self._full_scale_g = max(0.5, float(g))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)
        m = 8
        side = min(self.width(), self.height()) - 2 * m
        rect = QRectF((self.width() - side) / 2,
                      (self.height() - side) / 2, side, side)
        cx = rect.center().x()
        cy = rect.center().y()
        radius = side / 2.0 - 4
        p.setPen(QPen(QColor(80, 80, 90), 1))
        p.setBrush(QColor(28, 28, 34, 220))
        p.drawEllipse(rect)
        for ring_g in (0.5, 1.0, 1.5):
            r = radius * (ring_g / self._full_scale_g)
            if r <= 1:
                continue
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(70, 70, 80), 1, Qt.PenStyle.DashLine))
            p.drawEllipse(QPointF(cx, cy), r, r)
        p.setPen(QPen(QColor(70, 70, 80), 1))
        p.drawLine(int(cx - radius), int(cy), int(cx + radius), int(cy))
        p.drawLine(int(cx), int(cy - radius), int(cx), int(cy + radius))
        ax = self._snap.get("view_accel_lat_ms2")
        ay = self._snap.get("view_accel_lon_ms2")
        if ax is not None and ay is not None:
            gx = ax / GRAVITY
            gy = ay / GRAVITY
            scale = radius / self._full_scale_g
            px = cx + max(-radius, min(radius, gx * scale))
            py = cy - max(-radius, min(radius, gy * scale))
            mag = math.hypot(gx, gy)
            if mag < 0.5:
                col = QColor(120, 230, 140)
            elif mag < 1.0:
                col = QColor(255, 220, 60)
            else:
                col = QColor(255, 80, 80)
            p.setBrush(col)
            p.setPen(QPen(QColor(20, 20, 24), 1))
            p.drawEllipse(QPointF(px, py), 6, 6)
            p.setPen(QPen(QColor(220, 220, 230)))
            p.setFont(QFont(
                "Consolas", self._scale_pt(11), QFont.Weight.Bold,
            ))
            p.drawText(
                QRectF(0, rect.bottom() - 18, self.width(), 16),
                int(Qt.AlignmentFlag.AlignCenter),
                f"{mag:4.2f} g",
            )


# ---------------------------------------------------------------------------
# Gap compass
# ---------------------------------------------------------------------------


class GapCompassWindow(_LiveModuleWindow):
    """Arrow pointing to the nearest rival."""

    def __init__(
        self, source: LiveDataSource, *, opacity: float = 0.85,
    ) -> None:
        super().__init__(
            source, size=(160, 160),
            title="LFS Live - compass", opacity=opacity,
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)
        cars = self._snap.get("cars") or []
        nearest = None
        nearest_d = None
        for c in cars:
            if c.get("view"):
                continue
            d = float(c.get("d", 0.0))
            if d <= 0:
                continue
            if nearest_d is None or d < nearest_d:
                nearest = c
                nearest_d = d
        m = 8
        side = min(self.width(), self.height()) - 2 * m
        cx = self.width() / 2.0
        cy = self.height() / 2.0 - 8
        r = side / 2.0 - 6
        p.setPen(QPen(QColor(80, 80, 90), 1))
        p.setBrush(QColor(28, 28, 34, 220))
        p.drawEllipse(QPointF(cx, cy), r, r)
        if nearest is not None:
            x_l = float(nearest["x"])
            y_l = float(nearest["y"])
            ang = math.atan2(x_l, y_l)
            ax = cx + r * 0.85 * math.sin(ang)
            ay = cy - r * 0.85 * math.cos(ang)
            p.setPen(QPen(QColor(255, 220, 60), 3))
            p.drawLine(int(cx), int(cy), int(ax), int(ay))
            head = QPolygonF([
                QPointF(ax, ay),
                QPointF(
                    ax - 8 * math.sin(ang) - 4 * math.cos(ang),
                    ay + 8 * math.cos(ang) - 4 * math.sin(ang),
                ),
                QPointF(
                    ax - 8 * math.sin(ang) + 4 * math.cos(ang),
                    ay + 8 * math.cos(ang) + 4 * math.sin(ang),
                ),
            ])
            p.setBrush(QColor(255, 220, 60))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(head)
            p.setPen(QPen(QColor(220, 220, 230)))
            p.setFont(self._font(12, QFont.Weight.Bold))
            p.drawText(
                QRectF(0, self.height() - 22, self.width(), 18),
                int(Qt.AlignmentFlag.AlignCenter),
                f"{nearest_d:5.1f} m",
            )
        else:
            p.setPen(QPen(QColor(150, 150, 160)))
            p.setFont(self._font(11, QFont.Weight.Normal))
            p.drawText(
                QRectF(0, self.height() - 22, self.width(), 18),
                int(Qt.AlignmentFlag.AlignCenter), "no rival",
            )


# ---------------------------------------------------------------------------
# Mini-map
# ---------------------------------------------------------------------------


class MiniMapWindow(_LiveModuleWindow):
    """Top-down track map with cars + optional PB ghost."""

    def __init__(
        self, source: LiveDataSource, *,
        opacity: float = 0.85, show_ghost: bool = True,
    ) -> None:
        super().__init__(
            source, size=(280, 280),
            title="LFS Live - minimap", opacity=opacity,
        )
        self._line = RacingLine.empty()
        self._show_ghost = bool(show_ghost)

    def set_racing_line(self, line: RacingLine) -> None:
        self._line = line
        self.update()

    def set_show_ghost(self, on: bool) -> None:
        self._show_ghost = bool(on)
        self.update()

    def _world_to_widget(
        self, x: float, y: float, *, m: float = 10.0,
    ) -> tuple[float, float]:
        bbox = self._line.bbox
        xmin, ymin, xmax, ymax = bbox
        w = max(1e-3, xmax - xmin)
        h = max(1e-3, ymax - ymin)
        avail_w = self.width() - 2 * m
        avail_h = self.height() - 2 * m
        scale = min(avail_w / w, avail_h / h)
        ox = m + (avail_w - w * scale) / 2
        oy = m + (avail_h - h * scale) / 2
        return (
            ox + (x - xmin) * scale,
            self.height() - (oy + (y - ymin) * scale),
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)
        if self._line.is_empty:
            p.setPen(QPen(QColor(180, 180, 190)))
            p.setFont(self._font(10, QFont.Weight.Normal))
            p.drawText(
                self.rect(), int(Qt.AlignmentFlag.AlignCenter),
                "no racing line\nfor this track",
            )
            return
        p.setPen(QPen(QColor(120, 120, 140), 2))
        prev = None
        for x, y in self._line.points:
            cur = self._world_to_widget(x, y)
            if prev is not None:
                p.drawLine(int(prev[0]), int(prev[1]),
                           int(cur[0]), int(cur[1]))
            prev = cur
        if self._show_ghost:
            gnode = self._snap.get("ghost_node")
            if (
                gnode is not None
                and 0 <= int(gnode) < len(self._line.points)
            ):
                gx, gy = self._line.points[int(gnode)]
                ux, uy = self._world_to_widget(gx, gy)
                p.setBrush(QColor(180, 200, 255, 200))
                p.setPen(QPen(QColor(220, 230, 255), 1))
                p.drawEllipse(QPointF(ux, uy), 5, 5)
        cars = self._snap.get("cars_world") or []
        traffic = self._snap.get("traffic") or {}
        ahead_plid = traffic.get("ahead_plid")
        behind_plid = traffic.get("behind_plid")
        for c in cars:
            ux, uy = self._world_to_widget(
                float(c.get("x", 0.0)), float(c.get("y", 0.0))
            )
            is_view = bool(c.get("view"))
            plid = c.get("plid")
            if is_view:
                col = QColor(120, 200, 255)
                radius = 6.0
            elif plid == ahead_plid:
                col = QColor(255, 100, 100)
                radius = 5.0
            elif plid == behind_plid:
                col = QColor(255, 200, 80)
                radius = 5.0
            else:
                col = QColor(220, 220, 220)
                radius = 4.0
            p.setBrush(col)
            p.setPen(QPen(QColor(20, 20, 24), 1))
            p.drawEllipse(QPointF(ux, uy), radius, radius)

        # Compact legend for line + marker colors.
        box_h = max(46.0, min(66.0, self.height() * 0.24))
        box_w = min(self.width() - 16.0, max(160.0, self.width() * 0.68))
        box = QRectF(8.0, self.height() - box_h - 8.0, box_w, box_h)
        p.setPen(QPen(QColor(95, 95, 105, 170), 1))
        p.setBrush(QColor(10, 10, 14, 175))
        p.drawRoundedRect(box, 6, 6)

        p.setPen(QPen(QColor(210, 210, 220)))
        p.setFont(QFont("Segoe UI", self._scale_pt(7), QFont.Weight.Normal))
        txt = (
            "linea pista · ghost · ego · rival delante · "
            "rival detrás · otros"
        )
        p.drawText(
            QRectF(box.left() + 6, box.top() + 2,
                   box.width() - 12, box.height() * 0.45),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            txt,
        )

        y = box.bottom() - max(12.0, box.height() * 0.28)
        x = box.left() + 8.0
        sw = max(8.0, self.width() * 0.022)

        def _dot(px: float, color: QColor, radius: float) -> float:
            p.setBrush(color)
            p.setPen(QPen(QColor(20, 20, 24), 1))
            p.drawEllipse(QPointF(px, y), radius, radius)
            return px + radius * 2.0 + 9.0

        # Track centerline swatch.
        p.setPen(QPen(QColor(120, 120, 140), 2))
        p.drawLine(int(x), int(y), int(x + sw + 6), int(y))
        x += sw + 12.0

        # Ghost / ego / rivals.
        x = _dot(x, QColor(180, 200, 255, 220), 3.6)
        x = _dot(x, QColor(120, 200, 255), 4.0)
        x = _dot(x, QColor(255, 100, 100), 3.8)
        x = _dot(x, QColor(255, 200, 80), 3.8)
        _dot(x, QColor(220, 220, 220), 3.5)


# ---------------------------------------------------------------------------
# Radar
# ---------------------------------------------------------------------------


class RadarWindow(_LiveModuleWindow):
    MODULE_ID = "radar"
    """Top-down proximity radar (helicorsa visual + D&M detection)."""

    def __init__(
        self,
        source: LiveDataSource,
        *,
        radar_scale_m: float = 30.0,
        red_m: float = 2.0,
        yellow_m: float = 5.0,
        white_m: float = 12.0,
        opacity: float = 0.85,
    ) -> None:
        super().__init__(
            source, size=(220, 220),
            title="LFS Live - radar", opacity=opacity,
        )
        self._radar_scale_m = float(radar_scale_m)
        self._red_m = float(red_m)
        self._yellow_m = float(yellow_m)
        self._white_m = float(white_m)

    def set_radar_scale(self, meters: float) -> None:
        self._radar_scale_m = max(5.0, float(meters))
        self.update()

    def set_thresholds(
        self, *, red_m: float, yellow_m: float, white_m: float,
    ) -> None:
        self._red_m = float(red_m)
        self._yellow_m = float(yellow_m)
        self._white_m = float(white_m)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)
        m = 10
        side = min(self.width(), self.height()) - 2 * m
        rect = QRectF((self.width() - side) / 2,
                      (self.height() - side) / 2, side, side)
        cx, cy = rect.center().x(), rect.center().y()
        scale_px = (rect.width() / 2.0 - 8.0) / self._radar_scale_m
        p.setPen(QPen(QColor(80, 80, 90), 1))
        p.setBrush(QColor(28, 28, 34, 220))
        p.drawEllipse(rect)
        p.setPen(QPen(QColor(70, 70, 80), 1, Qt.PenStyle.DashLine))
        p.drawLine(int(cx), int(rect.top() + 8),
                   int(cx), int(rect.bottom() - 8))
        p.drawLine(int(rect.left() + 8), int(cy),
                   int(rect.right() - 8), int(cy))
        for radius_m, col in (
            (self._red_m, QColor(255, 60, 60, 180)),
            (self._yellow_m, QColor(255, 220, 60, 160)),
            (self._white_m, QColor(220, 220, 220, 140)),
        ):
            r = radius_m * scale_px
            if r <= 1:
                continue
            p.setPen(QPen(col, 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(rect.center(), r, r)
        p.setBrush(QColor(120, 200, 255))
        p.setPen(QPen(QColor(40, 40, 50), 1))
        size = 7
        triangle = QPolygonF([
            QPointF(cx, cy - size),
            QPointF(cx - size * 0.7, cy + size * 0.7),
            QPointF(cx + size * 0.7, cy + size * 0.7),
        ])
        p.drawPolygon(triangle)
        cars = self._snap.get("cars") or []
        traffic = self._snap.get("traffic") or {}
        ahead_plid = traffic.get("ahead_plid")
        behind_plid = traffic.get("behind_plid")
        warn_left = warn_right = False
        any_drawn = False
        for car in cars:
            if car.get("view"):
                continue
            x_local = float(car.get("x", 0.0))
            y_local = float(car.get("y", 0.0))
            d = float(car.get("d", 0.0))
            if d <= 0:
                continue
            # Side-warning bars: trigger when an opponent is roughly
            # alongside (small longitudinal offset) AND laterally
            # inside the yellow proximity ring. Using ``d`` (total
            # distance) here used to miss cars that were 5 m ahead
            # and 3 m to the side because their hypotenuse exceeds
            # ``yellow_m``. Checking ``abs(x_local)`` directly fixes
            # that and matches helicorsa's intent.
            if abs(y_local) <= max(self._yellow_m, 5.0) \
                    and abs(x_local) <= self._yellow_m:
                if x_local < -0.3:
                    warn_left = True
                elif x_local > 0.3:
                    warn_right = True
            if d > self._radar_scale_m:
                continue
            any_drawn = True
            px = cx + x_local * scale_px
            py = cy - y_local * scale_px
            col = proximity_color(
                d, red_m=self._red_m, yellow_m=self._yellow_m,
                white_m=self._white_m,
            )
            plid = car.get("plid")
            outline = QColor(0, 0, 0, 200)
            radius = 4.0
            if plid == ahead_plid:
                outline = QColor(80, 200, 255)
                radius = 5.0
            elif plid == behind_plid:
                outline = QColor(255, 180, 80)
                radius = 5.0
            p.setBrush(col)
            p.setPen(QPen(outline, 1.5))
            p.drawEllipse(QRectF(px - radius, py - radius,
                                 radius * 2, radius * 2))
        if not any_drawn:
            ahead_d = traffic.get("ahead_gap_m")
            behind_d = traffic.get("behind_gap_m")
            if ahead_d is not None and ahead_d > 0:
                d = float(ahead_d)
                py = cy - min(d, self._radar_scale_m) * scale_px
                p.setBrush(proximity_color(
                    d, red_m=self._red_m, yellow_m=self._yellow_m,
                    white_m=self._white_m))
                p.setPen(QPen(QColor(80, 200, 255), 1.5))
                p.drawEllipse(QRectF(cx - 4, py - 4, 8, 8))
            if behind_d is not None and behind_d > 0:
                d = float(behind_d)
                py = cy + min(d, self._radar_scale_m) * scale_px
                p.setBrush(proximity_color(
                    d, red_m=self._red_m, yellow_m=self._yellow_m,
                    white_m=self._white_m))
                p.setPen(QPen(QColor(255, 180, 80), 1.5))
                p.drawEllipse(QRectF(cx - 4, py - 4, 8, 8))
        if warn_left:
            p.setBrush(QColor(255, 80, 80))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(QRectF(rect.left() + 2,
                              rect.center().y() - 18, 6, 36))
        if warn_right:
            p.setBrush(QColor(255, 80, 80))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(QRectF(rect.right() - 8,
                              rect.center().y() - 18, 6, 36))


# ---------------------------------------------------------------------------
# Delta bar
# ---------------------------------------------------------------------------


class DeltaBarWindow(_LiveModuleWindow):
    MODULE_ID = "delta"
    """Horizontal oscillating bar: green (gaining) <-> red (losing).

    The raw ``delta_vs_best_ms`` jitters in the millisecond range, which
    is distracting. We low-pass it with an EMA and display tenths only
    so the readout communicates *trend* (are you improving?) instead of
    fake millisecond precision.
    """

    # 10 Hz source → alpha 0.25 ≈ 400 ms time constant.
    _DELTA_ALPHA = 0.25

    def __init__(
        self,
        source: LiveDataSource,
        *,
        full_scale_ms: int = 2000,
        opacity: float = 0.85,
    ) -> None:
        super().__init__(
            source, size=(360, 70),
            title="LFS Live - delta", opacity=opacity,
        )
        self._full_scale_ms = max(100, int(full_scale_ms))
        self._delta_smoothed: float | None = None

    def set_full_scale_ms(self, ms: int) -> None:
        self._full_scale_ms = max(100, int(ms))
        self.update()

    def _on_snapshot(self, snap: dict[str, Any]) -> None:
        raw = snap.get("delta_vs_best_ms")
        if raw is None:
            # Reset so we don't carry a stale trend across laps/sessions.
            self._delta_smoothed = None
        else:
            x = float(raw)
            if self._delta_smoothed is None:
                self._delta_smoothed = x
            else:
                a = self._DELTA_ALPHA
                self._delta_smoothed = (
                    self._delta_smoothed + a * (x - self._delta_smoothed)
                )
        super()._on_snapshot(snap)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)
        margin = 12
        bar_h = max(14, int(self.height() * 0.32))
        rect = QRectF(margin, self.height() - margin - bar_h - 4,
                      self.width() - 2 * margin, bar_h)
        cx = rect.center().x()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(40, 40, 50))
        p.drawRoundedRect(rect, 4, 4)
        delta = self._delta_smoothed
        p.setFont(QFont(
            "Consolas", self._scale_pt(14), QFont.Weight.Bold,
        ))
        if delta is None:
            p.setPen(QPen(QColor(180, 180, 190)))
            value_text = "--.-"
        elif delta < 0:
            p.setPen(QPen(QColor(120, 230, 140)))
            value_text = f"{delta / 1000.0:+.1f}"
        else:
            p.setPen(QPen(QColor(255, 120, 120)))
            value_text = f"{delta / 1000.0:+.1f}"
        p.drawText(
            QRectF(margin, 6, self.width() - 2 * margin,
                   self.height() - bar_h - 16),
            int(Qt.AlignmentFlag.AlignCenter),
            f"DELTA  {value_text}",
        )
        if delta is not None:
            clamped = max(-self._full_scale_ms,
                          min(self._full_scale_ms, int(delta)))
            frac = clamped / self._full_scale_ms
            half_w = rect.width() / 2.0
            if frac < 0:
                fill = QRectF(cx + frac * half_w, rect.top(),
                              -frac * half_w, rect.height())
                grad = QLinearGradient(fill.right(), 0, fill.left(), 0)
                grad.setColorAt(0.0, QColor(120, 230, 140))
                grad.setColorAt(1.0, QColor(40, 160, 80))
                p.setBrush(QBrush(grad))
            else:
                fill = QRectF(cx, rect.top(),
                              frac * half_w, rect.height())
                grad = QLinearGradient(fill.left(), 0, fill.right(), 0)
                grad.setColorAt(0.0, QColor(255, 120, 120))
                grad.setColorAt(1.0, QColor(200, 40, 40))
                p.setBrush(QBrush(grad))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(fill, 4, 4)
        p.setPen(QPen(QColor(220, 220, 230), 2))
        p.drawLine(int(cx), int(rect.top() - 2),
                   int(cx), int(rect.bottom() + 2))


class SpeedDeltaBarWindow(_LiveModuleWindow):
    MODULE_ID = "speed_delta"
    """Speed delta vs PB at the same track node (km/h, bar).

    Companion to :class:`DeltaBarWindow`: instead of comparing time,
    it compares the **speed** you are carrying right now against the
    speed you carried through the same point of the circuit on your
    PB lap. Detect&Monitor exposes a similar gauge; it complements
    the time delta because you can be losing time (positive time
    delta) while still being faster locally (positive speed delta)
    -- typical signature of a different line / wider entry that pays
    off later.

    Positive = faster than PB here (green). Negative = slower (red).
    Reads ``speed_delta_kmh_vs_best`` from the live snapshot, low-pass
    filtered with an EMA (same time constant as the time bar) to
    suppress sample-level jitter.
    """

    # 10 Hz source -> alpha 0.25 ~ 400 ms time constant.
    _DELTA_ALPHA = 0.25

    def __init__(
        self,
        source: LiveDataSource,
        *,
        full_scale_kmh: float = 20.0,
        opacity: float = 0.85,
    ) -> None:
        super().__init__(
            source, size=(360, 70),
            title="LFS Live - speed delta", opacity=opacity,
        )
        self._full_scale_kmh = max(1.0, float(full_scale_kmh))
        self._delta_smoothed: float | None = None

    def set_full_scale_kmh(self, kmh: float) -> None:
        self._full_scale_kmh = max(1.0, float(kmh))
        self.update()

    def _on_snapshot(self, snap: dict[str, Any]) -> None:
        raw = snap.get("speed_delta_kmh_vs_best")
        if raw is None:
            self._delta_smoothed = None
        else:
            x = float(raw)
            if self._delta_smoothed is None:
                self._delta_smoothed = x
            else:
                a = self._DELTA_ALPHA
                self._delta_smoothed = (
                    self._delta_smoothed + a * (x - self._delta_smoothed)
                )
        super()._on_snapshot(snap)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)
        margin = 12
        bar_h = max(14, int(self.height() * 0.32))
        rect = QRectF(margin, self.height() - margin - bar_h - 4,
                      self.width() - 2 * margin, bar_h)
        cx = rect.center().x()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(40, 40, 50))
        p.drawRoundedRect(rect, 4, 4)
        delta = self._delta_smoothed
        p.setFont(QFont(
            "Consolas", self._scale_pt(14), QFont.Weight.Bold,
        ))
        if delta is None:
            p.setPen(QPen(QColor(180, 180, 190)))
            value_text = "--.-"
        elif delta >= 0:
            p.setPen(QPen(QColor(120, 230, 140)))
            value_text = f"{delta:+.1f}"
        else:
            p.setPen(QPen(QColor(255, 120, 120)))
            value_text = f"{delta:+.1f}"
        p.drawText(
            QRectF(margin, 6, self.width() - 2 * margin,
                   self.height() - bar_h - 16),
            int(Qt.AlignmentFlag.AlignCenter),
            f"\u0394V  {value_text} km/h",
        )
        if delta is not None:
            clamped = max(-self._full_scale_kmh,
                          min(self._full_scale_kmh, float(delta)))
            frac = clamped / self._full_scale_kmh
            half_w = rect.width() / 2.0
            if frac >= 0:
                # Faster than PB -> grow to the RIGHT, green.
                fill = QRectF(cx, rect.top(),
                              frac * half_w, rect.height())
                grad = QLinearGradient(fill.left(), 0, fill.right(), 0)
                grad.setColorAt(0.0, QColor(120, 230, 140))
                grad.setColorAt(1.0, QColor(40, 160, 80))
                p.setBrush(QBrush(grad))
            else:
                # Slower than PB -> grow to the LEFT, red.
                fill = QRectF(cx + frac * half_w, rect.top(),
                              -frac * half_w, rect.height())
                grad = QLinearGradient(fill.right(), 0, fill.left(), 0)
                grad.setColorAt(0.0, QColor(255, 120, 120))
                grad.setColorAt(1.0, QColor(200, 40, 40))
                p.setBrush(QBrush(grad))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(fill, 4, 4)
        p.setPen(QPen(QColor(220, 220, 230), 2))
        p.drawLine(int(cx), int(rect.top() - 2),
                   int(cx), int(rect.bottom() + 2))


__all__ = [
    "BrakeWindow",
    "ClutchWindow",
    "DeltaBarWindow",
    "FlagsWindow",
    "FuelLapsRemainingWindow",
    "FuelPctWindow",
    "GMeterWindow",
    "GapAheadWindow",
    "GapBehindWindow",
    "GapCompassWindow",
    "GearWindow",
    "MiniMapWindow",
    "PitLimiterWindow",
    "PositionWindow",
    "RadarWindow",
    "RpmWindow",
    "SessionInfoWindow",
    "SpeedDeltaBarWindow",
    "SpeedWindow",
    "TyreRiskWindow",
    "TcAbsWindow",
    "ThrottleWindow",
    "proximity_color",
]
