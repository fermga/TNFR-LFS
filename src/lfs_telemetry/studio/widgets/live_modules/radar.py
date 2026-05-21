"""Auto-split from live_modules.py — MH1."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPen,
    QPolygonF,
)

from ..live_data_source import LiveDataSource
from ._base import (
    _LiveModuleWindow,
    proximity_color,
)


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

    def paintEvent(self, event) -> None:
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


