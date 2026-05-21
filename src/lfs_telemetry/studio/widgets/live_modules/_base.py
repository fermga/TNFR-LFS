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
from typing import Any

from PySide6.QtCore import QPoint, QRectF, QSettings, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import QWidget

from ....lfs_paths import QSETTINGS_APP as APP
from ....lfs_paths import QSETTINGS_ORG as ORG
from ...theme import (
    PROXIMITY_FAR,
    PROXIMITY_RED,
    PROXIMITY_WHITE,
    PROXIMITY_YELLOW,
)
from .._format import (
    format_clock_ms,
    format_gap_seconds,
    format_signed_delta_ms,
)
from ..live_data_source import LiveDataSource

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
        return QColor(PROXIMITY_RED)
    if distance_m <= yellow_m:
        return QColor(PROXIMITY_YELLOW)
    if distance_m <= white_m:
        return QColor(PROXIMITY_WHITE)
    return QColor(PROXIMITY_FAR)


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
        return round(self.windowOpacity() * 100)

    def reset_size(self) -> None:
        self.resize(*self._default_size)

    # ----- Drag + resize ----------------------------------------------

    def _in_resize_zone(self, pos: QPoint) -> bool:
        return (
            pos.x() >= self.width() - RESIZE_GRIP_PX
            and pos.y() >= self.height() - RESIZE_GRIP_PX
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
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

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
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

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None or self._resizing:
            self._save_geometry()
        self._drag_offset = None
        self._resizing = False
        self.unsetCursor()

    def closeEvent(self, event) -> None:
        # Persist the final spot so re-enabling brings the module back
        # to where the user last left it.
        self._save_geometry()
        super().closeEvent(event)

    def hideEvent(self, event) -> None:
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
        return max(6, round(base_pt * cur / ref_dim))

    def _font(self, base_pt: int, weight=QFont.Weight.Bold,
              family: str = "Segoe UI") -> QFont:
        return QFont(family, self._scale_pt(base_pt), weight)

    # ----- Off-screen render (for VR / capture sinks) ------------------

    def render_to_image(self) -> QImage:
        """Render the current widget contents to an off-screen ARGB image.

        The image has the same pixel size as the widget and a fully
        transparent background, so alternate sinks (e.g. an OpenVR/OpenXR
        overlay or a screenshot tool) can composite it without depending
        on the on-screen window being mapped.

        Safe to call whether or not the window is currently visible.
        """
        size = self.size()
        if size.width() <= 0 or size.height() <= 0:
            from PySide6.QtCore import QSize
            size = QSize(*self._default_size)
        img = QImage(size, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)  # fully transparent background
        self.render(img)
        return img


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

    def paintEvent(self, event) -> None:
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

