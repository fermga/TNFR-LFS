"""Auto-split from live_modules.py — MH1."""
from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPen,
)

from ....telemetry.constants import GRAVITY
from ..live_data_source import LiveDataSource
from ._base import (
    _LiveModuleWindow,
)


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

    def paintEvent(self, event) -> None:
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

    def paintEvent(self, event) -> None:
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

    def paintEvent(self, event) -> None:
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

    def paintEvent(self, event) -> None:
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


