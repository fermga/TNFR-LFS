"""Auto-split from live_modules.py — MH1."""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPen,
)

from ..live_data_source import LiveDataSource
from ._base import (
    _LiveModuleWindow,
)


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

    def paintEvent(self, event) -> None:
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

    def paintEvent(self, event) -> None:
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

    def paintEvent(self, event) -> None:
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
        pct = round(frac * 100.0)
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


