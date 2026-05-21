"""Auto-split from live_modules.py — MH1."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPen,
    QPolygonF,
)

from ..live_data_source import LiveDataSource
from ..racing_line_loader import RacingLine
from ._base import (
    _LiveModuleWindow,
)


class GapCompassWindow(_LiveModuleWindow):
    """Arrow pointing to the nearest rival."""

    def __init__(
        self, source: LiveDataSource, *, opacity: float = 0.85,
    ) -> None:
        super().__init__(
            source, size=(160, 160),
            title="LFS Live - compass", opacity=opacity,
        )

    def paintEvent(self, event) -> None:
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

    def paintEvent(self, event) -> None:
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


