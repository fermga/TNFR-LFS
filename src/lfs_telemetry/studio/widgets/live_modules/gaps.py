"""Auto-split from live_modules.py — MH1."""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPen,
)

from ..live_data_source import LiveDataSource
from ._base import (
    _fmt_gap,
    _LiveModuleWindow,
)


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

    def paintEvent(self, event) -> None:
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


