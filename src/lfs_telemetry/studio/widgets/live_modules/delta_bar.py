"""Auto-split from live_modules.py — MH1."""
from __future__ import annotations

from typing import Any

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

    def paintEvent(self, event) -> None:
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

    def paintEvent(self, event) -> None:
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
