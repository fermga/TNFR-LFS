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

from ...lfs_paths import QSETTINGS_APP as APP, QSETTINGS_ORG as ORG
from .live_data_source import LiveDataSource
from .racing_line_loader import RacingLine
from ._format import (
    format_clock_ms,
    format_gap_seconds,
    format_signed_delta_ms,
)

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

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
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
            try:
                self.restoreGeometry(geo)
            except (TypeError, ValueError):
                pass

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
            gx = ax / 9.81
            gy = ay / 9.81
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
            if abs(y_local) <= max(self._yellow_m, 5.0) \
                    and d <= self._yellow_m:
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
    "PositionWindow",
    "RadarWindow",
    "RpmWindow",
    "SpeedWindow",
    "TcAbsWindow",
    "ThrottleWindow",
    "proximity_color",
]
