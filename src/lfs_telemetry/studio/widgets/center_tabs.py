"""Central tab widget hosting Channels / Stint / Capture views.

Replaces the previous flat ChartsDock central widget so the Studio
matches the Dash app's tabbed layout. Each tab is a self-contained
``QWidget`` that subscribes to the shared :class:`SignalBus`; switching
tabs has no side effects on lap loading or signal routing.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QTabWidget, QWidget

from ..i18n import tr
from ..models import LapLoader
from ..signals import SignalBus
from .capture_tab import CaptureTab
from .charts_dock import ChartsDock
from .dampers_tab import DampersTab
from .live_tab import LiveTab
from .sectors_tab import SectorsTab
from .stint_tab import StintTab


class CenterTabs(QTabWidget):
    """Channels / Stint / Capture tabs in a single central widget."""

    def __init__(
        self,
        loader: LapLoader,
        signals: SignalBus,
        workspace: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.charts = ChartsDock(loader, signals, self)
        self.stint = StintTab(loader, signals, self)
        self.sectors = SectorsTab(loader, signals, self)
        self.dampers = DampersTab(loader, signals, self)
        self.capture = CaptureTab(workspace, signals, self)
        self.live = LiveTab(self.capture.runner, signals, self)
        self.addTab(self.charts, tr("Telemetry"))
        self.addTab(self.dampers, tr("Dampers"))
        self.addTab(self.sectors, tr("Sectors"))
        self.addTab(self.stint, tr("Stint"))
        self.addTab(self.capture, tr("Capture"))
        self.addTab(self.live, tr("Overlay"))
        self.setDocumentMode(True)
        self.setMovable(False)


__all__ = ["CenterTabs"]
