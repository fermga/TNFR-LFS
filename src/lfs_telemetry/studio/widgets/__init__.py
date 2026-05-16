"""Dockable panels for the Studio main window."""

from __future__ import annotations

from .capture_tab import CaptureTab
from .captures_dock import CapturesDock
from .center_tabs import CenterTabs
from .channels_dock import ChannelsDock
from .charts_dock import ChartsDock
from .dampers_tab import DampersTab
from .race_dashboard_dock import RaceDashboardDock
from .sectors_tab import SectorsTab
from .stint_tab import StintTab
from .track_elevation_dock import TrackElevationDock
from .track_map_dock import TrackMapDock

__all__ = [
    "CaptureTab",
    "CapturesDock",
    "CenterTabs",
    "ChannelsDock",
    "ChartsDock",
    "DampersTab",
    "RaceDashboardDock",
    "SectorsTab",
    "StintTab",
    "TrackElevationDock",
    "TrackMapDock",
]
