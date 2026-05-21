"""Dark MoTeC-like theme + pyqtgraph defaults.

Centralized so widgets stay declarative. The accent palette is chosen so
adjacent traces are unambiguously distinguishable on a near-black canvas
(important when 6-8 channels overlap during compare).
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# Qualitative palette tuned for dark backgrounds. The first 12 are the
# canonical MoTeC i2 / Tableau 10 hues with contrast-boosted variants.
TRACE_COLORS: tuple[str, ...] = (
    "#5dade2",  # cyan
    "#ec7063",  # red
    "#58d68d",  # green
    "#f5b041",  # amber
    "#bb8fce",  # violet
    "#f4d03f",  # yellow
    "#48c9b0",  # teal
    "#e59866",  # orange
    "#85c1e9",  # sky
    "#dc7633",  # rust
    "#a9dfbf",  # mint
    "#d2b4de",  # lavender
)

# Reference vs. candidate in the Compare view.
REF_COLOR = "#ffffff"
CAND_COLOR = "#ec7063"

# Background / grid / axis pen.
BG_COLOR = "#101418"
PANEL_COLOR = "#161b21"
GRID_COLOR = "#262d36"
TEXT_COLOR = "#d8dde3"
MUTED_COLOR = "#5b6470"
CURSOR_COLOR = "#ffd166"

# Semantic status / overlay colours. Keep these here so widgets never
# hard-code raw hex/RGB. Shades match the dark palette above.
STATUS_ERROR_COLOR = "#c0392b"          # validation/error text
LED_IDLE_COLOR = "#5a5f66"              # grey LED (idle)
LED_ERROR_COLOR = "#d04848"             # red LED (failure)
LED_OK_COLOR = "#3fbf5a"                # green LED (connected)
PROXIMITY_RED = QColor(255, 60, 60)     # imminent collision
PROXIMITY_YELLOW = QColor(255, 220, 60) # within yellow band
PROXIMITY_WHITE = QColor(230, 230, 230) # within white band
PROXIMITY_FAR = QColor(140, 140, 140)   # outside detection ring
COMPARE_OUTLINE_COLOR = "#ffffff"       # damper-histogram compare lap


def trace_color(index: int) -> str:
    """Return the trace color at ``index`` modulo the palette."""
    return TRACE_COLORS[index % len(TRACE_COLORS)]


# Canonical UI wheel order (driver's perspective, front row then rear row).
# Note: the LFS binary wire order is RL,RR,FL,FR — see
# ``telemetry.protocol.packets.WHEEL_ORDER``. Always remap between the two
# explicitly; do not assume they are the same.
WHEEL_ORDER_UI: tuple[str, ...] = ("FL", "FR", "RL", "RR")

# Stable per-wheel colours for the multi-line / histogram plots.
WHEEL_COLORS: dict[str, str] = {
    "FL": "#4ea3ff",   # blue
    "FR": "#ffa040",   # orange
    "RL": "#7ed957",   # green
    "RR": "#ff5d6c",   # red
}

# (corner, row, col) layout for 2×2 per-wheel grids (driver's view).
WHEEL_GRID_LAYOUT: tuple[tuple[str, int, int], ...] = (
    ("FL", 0, 0),
    ("FR", 0, 1),
    ("RL", 1, 0),
    ("RR", 1, 1),
)


def configure_pyqtgraph() -> None:
    """Apply global pyqtgraph defaults. Idempotent."""
    pg.setConfigOptions(
        background=BG_COLOR,
        foreground=TEXT_COLOR,
        antialias=True,
        useOpenGL=False,        # Qt's 2D backend is plenty fast with LTTB
        enableExperimental=False,
        crashWarning=True,
    )


def apply_dark_palette(app: QApplication) -> None:
    """Apply a dark Fusion palette consistent with the chart canvas."""
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG_COLOR))
    palette.setColor(QPalette.WindowText, QColor(TEXT_COLOR))
    palette.setColor(QPalette.Base, QColor(PANEL_COLOR))
    palette.setColor(QPalette.AlternateBase, QColor("#1a2028"))
    palette.setColor(QPalette.ToolTipBase, QColor(PANEL_COLOR))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT_COLOR))
    palette.setColor(QPalette.Text, QColor(TEXT_COLOR))
    palette.setColor(QPalette.Button, QColor("#1c232c"))
    palette.setColor(QPalette.ButtonText, QColor(TEXT_COLOR))
    palette.setColor(QPalette.BrightText, QColor("#ff5252"))
    palette.setColor(QPalette.Highlight, QColor("#3a86ff"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.Link, QColor("#5dade2"))
    palette.setColor(
        QPalette.Disabled, QPalette.WindowText, QColor(MUTED_COLOR)
    )
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(MUTED_COLOR))
    palette.setColor(
        QPalette.Disabled, QPalette.ButtonText, QColor(MUTED_COLOR)
    )
    app.setPalette(palette)
    app.setStyleSheet(_QSS)


# Touch-up specific widgets (dock title bars, splitter handles, scroll
# bars) so the whole window feels like one product, not "Qt with a dark
# palette". Kept short on purpose — every selector that grows here makes
# theming harder when we add new widgets later.
_QSS = """
QMainWindow, QDialog { background-color: #101418; }
QDockWidget {
    color: #d8dde3;
    titlebar-close-icon: url(none);
    titlebar-normal-icon: url(none);
}
QDockWidget::title {
    background: #1c232c;
    padding: 6px 10px;
    border-bottom: 1px solid #262d36;
    text-align: left;
    font-weight: 600;
}
QSplitter::handle { background: #262d36; }
QSplitter::handle:horizontal { width: 4px; }
QSplitter::handle:vertical { height: 4px; }
QHeaderView::section {
    background: #1c232c;
    color: #d8dde3;
    padding: 4px 8px;
    border: none;
    border-right: 1px solid #262d36;
}
QTableView, QTreeView, QListView {
    background: #161b21;
    alternate-background-color: #1a2028;
    selection-background-color: #2a4365;
    selection-color: #ffffff;
    gridline-color: #262d36;
}
QStatusBar { background: #1c232c; color: #d8dde3; }
QToolBar { background: #1c232c; border: none; spacing: 4px; padding: 4px; }
QPushButton, QToolButton {
    background: #232b35; color: #d8dde3;
    border: 1px solid #2e3744; border-radius: 3px;
    padding: 4px 10px;
}
QPushButton:hover, QToolButton:hover { background: #2c3542; }
QPushButton:pressed, QToolButton:pressed { background: #1a2028; }
QLineEdit, QComboBox {
    background: #161b21; color: #d8dde3;
    border: 1px solid #2e3744; border-radius: 3px;
    padding: 3px 6px;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: #161b21; border: none;
}
QScrollBar:vertical { width: 10px; }
QScrollBar:horizontal { height: 10px; }
QScrollBar::handle {
    background: #2e3744; border-radius: 4px; min-height: 24px; min-width: 24px;
}
QScrollBar::handle:hover { background: #3a4453; }
QScrollBar::add-line, QScrollBar::sub-line {
    background: none; height: 0; width: 0;
}
"""


__all__ = [
    "BG_COLOR", "CAND_COLOR", "COMPARE_OUTLINE_COLOR", "CURSOR_COLOR",
    "GRID_COLOR", "LED_ERROR_COLOR", "LED_IDLE_COLOR", "LED_OK_COLOR",
    "MUTED_COLOR", "PANEL_COLOR", "PROXIMITY_FAR", "PROXIMITY_RED",
    "PROXIMITY_WHITE", "PROXIMITY_YELLOW", "REF_COLOR", "STATUS_ERROR_COLOR",
    "TEXT_COLOR", "TRACE_COLORS",
    "apply_dark_palette", "configure_pyqtgraph", "trace_color",
]
