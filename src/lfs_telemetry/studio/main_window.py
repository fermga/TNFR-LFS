"""Studio main window: dockable layout (Captures | Charts | Channels).

Layout philosophy mirrors MoTeC i2 / AIM RaceStudio:

* Captures (workspace browser) docks left.
* Channels (signal selector) docks right.
* Charts (the multi-channel stack) is the central widget — never
  closable, always present.
* Status bar shows the live cursor position in the current x-axis unit
  plus transient messages from the signal bus.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
)

from .models import LapLoader
from .signals import SignalBus
from .widgets import (
    CapturesDock,
    CenterTabs,
    ChannelsDock,
    RaceDashboardDock,
    TrackElevationDock,
    TrackMapDock,
)
from .workspace_state import WorkspaceState
from ..lfs_paths import QSETTINGS_APP as APP, QSETTINGS_ORG as ORG
from .i18n import (
    LANG_ENGLISH,
    LANG_SPANISH,
    current_language,
    set_language,
    tr,
)

# Bump when the dock layout changes so users don't restore a stale
# (and possibly broken) geometry from a previous build.
_LAYOUT_VERSION = 5


class MainWindow(QMainWindow):
    """The Studio shell."""

    def __init__(self, workspace_root: Path) -> None:
        super().__init__()
        self.setWindowTitle(APP)
        self.resize(1500, 900)

        # ----- Domain objects --------------------------------------
        self._workspace = WorkspaceState(workspace_root)
        self._signals = SignalBus(self)
        self._loader = LapLoader(self._workspace, max_workers=2, parent=self)

        # ----- Docks -----------------------------------------------
        self._captures_dock = QDockWidget(tr("Captures"), self)
        self._captures_dock.setObjectName("CapturesDock")
        self._captures = CapturesDock(self._workspace, self._signals,
                                      self._captures_dock)
        self._captures_dock.setWidget(self._captures)
        self._captures_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,
                           self._captures_dock)

        self._channels_dock = QDockWidget(tr("Telemetry"), self)
        self._channels_dock.setObjectName("ChannelsDock")
        self._channels = ChannelsDock(self._signals, self._channels_dock)
        self._channels_dock.setWidget(self._channels)
        self._channels_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self._channels_dock)

        # Track-map dock (bottom; consumes cursor_moved).
        self._track_dock = QDockWidget(tr("Track map"), self)
        self._track_dock.setObjectName("TrackMapDock")
        self._track_map = TrackMapDock(self._loader, self._signals,
                                       self._track_dock)
        self._track_dock.setWidget(self._track_map)
        self._track_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        # Stack under Captures (bottom-left corner) via vertical split.
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,
                           self._track_dock)
        self.splitDockWidget(self._captures_dock, self._track_dock,
                             Qt.Orientation.Vertical)

        # Track-elevation dock (tabbed with the top-down map so the two
        # 3D-related views share the same screen real-estate).
        self._elev_dock = QDockWidget(tr("Elevation"), self)
        self._elev_dock.setObjectName("TrackElevationDock")
        self._track_elev = TrackElevationDock(self._loader, self._signals,
                                              self._elev_dock)
        self._elev_dock.setWidget(self._track_elev)
        self._elev_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,
                           self._elev_dock)
        self.tabifyDockWidget(self._track_dock, self._elev_dock)
        self._track_dock.raise_()

        # Central area: tabbed (Channels / Stint / Capture).
        self._center = CenterTabs(
            self._loader, self._signals, self._workspace.workspace, self,
        )
        self._center.setMinimumWidth(480)
        self._charts = self._center.charts  # backwards-compat alias
        self.setCentralWidget(self._center)

        # Live Race Dashboard dock (right side, tabbed with Channels)
        # — reads live.json via the same CaptureRunner the Live tab uses.
        self._dash_dock = QDockWidget(tr("Race dashboard"), self)
        self._dash_dock.setObjectName("RaceDashboardDock")
        self._dash = RaceDashboardDock(
            self._center.capture.runner, self._signals, self._dash_dock,
        )
        self._dash_dock.setWidget(self._dash)
        self._dash_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self._dash_dock)
        self.tabifyDockWidget(self._channels_dock, self._dash_dock)
        self._channels_dock.raise_()

        # Constrain dock widths so the central chart area never gets
        # squeezed to nothing on first show — Qt's default dock layout
        # gives docks their sizeHint, which can starve a wide tree view.
        self._captures_dock.setMinimumWidth(220)
        self._channels_dock.setMinimumWidth(240)
        self.resizeDocks(
            [self._captures_dock, self._channels_dock],
            [320, 320],
            Qt.Orientation.Horizontal,
        )
        # Vertical split under captures: give track-map a square-ish slot.
        self._track_dock.setMinimumHeight(220)
        self._track_dock.setMinimumWidth(220)
        self.resizeDocks(
            [self._captures_dock, self._track_dock],
            [360, 360],
            Qt.Orientation.Vertical,
        )

        # ----- Menus + actions ------------------------------------
        self._build_actions()

        # ----- Status bar -----------------------------------------
        self._status = QStatusBar(self)
        self.setStatusBar(self._status)
        self._cursor_label = QLabel("", self)
        self._cursor_label.setMinimumWidth(220)
        self._status.addPermanentWidget(self._cursor_label)
        self._status.showMessage(f"Workspace: {workspace_root}")
        self._x_axis_kind = "distance"
        self._signals.cursor_moved.connect(self._on_cursor_moved)
        self._signals.cursor_left.connect(lambda: self._cursor_label.setText(""))
        self._signals.x_axis_changed.connect(self._on_axis_changed)
        self._signals.status_message.connect(self._status.showMessage)

        # Restore window geometry / dock state if available.
        self._restore_state()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_state()
        self._loader.shutdown()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Menus
    # ------------------------------------------------------------------

    def _build_actions(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu(tr("&File"))

        open_act = QAction(tr("Open Workspace\u2026"), self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.triggered.connect(self._action_open_workspace)
        file_menu.addAction(open_act)

        refresh_act = QAction(tr("Refresh Captures"), self)
        refresh_act.setShortcut("F5")
        refresh_act.triggered.connect(self._captures.refresh)
        file_menu.addAction(refresh_act)

        clear_cache_act = QAction(tr("Clear Lap Cache"), self)
        clear_cache_act.triggered.connect(self._action_clear_cache)
        file_menu.addAction(clear_cache_act)

        file_menu.addSeparator()
        quit_act = QAction(tr("&Quit"), self)
        quit_act.setShortcut(QKeySequence.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        view_menu = menubar.addMenu(tr("&View"))
        view_menu.addAction(self._captures_dock.toggleViewAction())
        view_menu.addAction(self._channels_dock.toggleViewAction())
        view_menu.addAction(self._track_dock.toggleViewAction())
        view_menu.addAction(self._elev_dock.toggleViewAction())
        view_menu.addSeparator()
        reset_act = QAction(tr("Reset Layout"), self)
        reset_act.triggered.connect(self._action_reset_layout)
        view_menu.addAction(reset_act)
        view_menu.addSeparator()
        lang_menu = view_menu.addMenu(tr("&Language"))
        self._lang_actions: dict[str, QAction] = {}
        current = current_language()
        for code, label in (
            (LANG_ENGLISH, tr("English")),
            (LANG_SPANISH, tr("Spanish")),
        ):
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(code == current)
            act.triggered.connect(
                lambda _checked, c=code: self._action_set_language(c),
            )
            lang_menu.addAction(act)
            self._lang_actions[code] = act

        tools_menu = menubar.addMenu(tr("&Tools"))
        configure_lfs_act = QAction(tr("Configure LFS\u2026"), self)
        configure_lfs_act.setStatusTip(
            tr(
                "Patch LFS cfg.txt with the OutSim/OutGauge/InSim settings "
                "required by lfs-telemetry.",
            ),
        )
        configure_lfs_act.triggered.connect(self._action_configure_lfs)
        tools_menu.addAction(configure_lfs_act)

        help_menu = menubar.addMenu(tr("&Help"))
        guide_act = QAction(tr("Channel guide\u2026"), self)
        guide_act.setShortcut("F1")
        guide_act.setStatusTip(
            tr(
                "Open the telemetry guide: what each channel measures "
                "and how to read it, in plain language.",
            ),
        )
        guide_act.triggered.connect(self._action_channel_guide)
        help_menu.addAction(guide_act)
        about_act = QAction(tr("About"), self)
        about_act.triggered.connect(self._action_about)
        help_menu.addAction(about_act)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _action_open_workspace(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            tr("Choose workspace folder"),
            str(self._workspace.workspace),
        )
        if not path:
            return
        self.set_workspace(Path(path))

    def _action_clear_cache(self) -> None:
        self._workspace.clear_cache()
        self._status.showMessage(tr("Lap cache cleared."), 4000)

    def _action_reset_layout(self) -> None:
        self.removeDockWidget(self._captures_dock)
        self.removeDockWidget(self._channels_dock)
        self.removeDockWidget(self._track_dock)
        self.removeDockWidget(self._elev_dock)
        self.removeDockWidget(self._dash_dock)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,
                           self._captures_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self._channels_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self._dash_dock)
        self.tabifyDockWidget(self._channels_dock, self._dash_dock)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,
                           self._track_dock)
        self.splitDockWidget(self._captures_dock, self._track_dock,
                             Qt.Orientation.Vertical)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,
                           self._elev_dock)
        self.tabifyDockWidget(self._track_dock, self._elev_dock)
        self._track_dock.raise_()
        self._channels_dock.raise_()
        self._captures_dock.show()
        self._channels_dock.show()
        self._dash_dock.show()
        self._track_dock.show()
        self._elev_dock.show()
        self.resizeDocks(
            [self._captures_dock, self._channels_dock],
            [320, 320],
            Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self._captures_dock, self._track_dock],
            [360, 360],
            Qt.Orientation.Vertical,
        )

    def _action_about(self) -> None:
        from . import __version__
        QMessageBox.information(
            self, tr("About"),
            tr(
                "LFS Telemetry Studio {version}\n"
                "Native dockable analyser built on PySide6 + pyqtgraph.\n\n"
                "To stream telemetry, LFS needs OutSim/OutGauge/InSim entries "
                "in cfg.txt.\n"
                "Use \u201cTools \u2192 Configure LFS\u2026\u201d to patch "
                "them automatically or copy the snippet manually.",
            ).format(version=__version__),
        )

    def _action_configure_lfs(self) -> None:
        from .widgets.lfs_config_dialog import LfsConfigDialog
        dlg = LfsConfigDialog(self)
        dlg.exec()

    def _action_set_language(self, code: str) -> None:
        set_language(code)
        for c, act in self._lang_actions.items():
            act.setChecked(c == code)
        QMessageBox.information(
            self, tr("Restart required"),
            tr(
                "Language will change the next time you start "
                "the application.",
            ),
        )

    def _action_channel_guide(self) -> None:
        from .widgets.help_dialog import HelpDialog
        dlg = HelpDialog(self)
        dlg.exec()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_workspace(self, path: Path) -> None:
        self._workspace.workspace = path
        self._captures.refresh()
        self._signals.workspace_changed.emit(path)
        self.setWindowTitle(f"{APP} — {path}")
        self._status.showMessage(f"Workspace: {path}", 6000)

    # ------------------------------------------------------------------
    # Cursor / axis
    # ------------------------------------------------------------------

    def _on_cursor_moved(self, x: float) -> None:
        unit = "m" if self._x_axis_kind == "distance" else "s"
        self._cursor_label.setText(f"x = {x:,.2f} {unit}")

    def _on_axis_changed(self, kind: str) -> None:
        self._x_axis_kind = kind

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _settings(self) -> QSettings:
        return QSettings(ORG, APP)

    def _save_state(self) -> None:
        s = self._settings()
        s.setValue("layoutVersion", _LAYOUT_VERSION)
        s.setValue("geometry", self.saveGeometry())
        s.setValue("windowState", self.saveState())

    def _restore_state(self) -> None:
        s = self._settings()
        version = s.value("layoutVersion", 0, int)
        if version != _LAYOUT_VERSION:
            # Layout schema changed; ignore stale geometry/state.
            return
        geom = s.value("geometry")
        state = s.value("windowState")
        if geom is not None:
            self.restoreGeometry(geom)
        if state is not None:
            self.restoreState(state)


__all__ = ["MainWindow"]
