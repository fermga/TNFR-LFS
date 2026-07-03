"""Live tab: enable, configure, and place every overlay module.

Every datum in the live snapshot is exposed as its own toggleable,
draggable, **resizable** frameless top-most window. Tick a module's
checkbox to show it; right-click any window to reset its size; drag
the bottom-right corner to resize.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...app.capture_runner import CaptureRunner
from ...lfs_config import read_lfs_vr_mode
from ...lfs_paths import QSETTINGS_APP as APP
from ...lfs_paths import QSETTINGS_ORG as ORG
from ...lfs_paths import autodetect_lfs_dir, get_lfs_dir
from ..i18n import tr
from ..signals import SignalBus
from ..vr import VrMirror
from .live_data_source import LiveDataSource
from .live_modules import (
    DeltaBarWindow,
    FlagsWindow,
    FuelLapsRemainingWindow,
    FuelPctWindow,
    GapAheadWindow,
    GapBehindWindow,
    GearWindow,
    GMeterWindow,
    PitLimiterWindow,
    RadarWindow,
    RpmWindow,
    SessionInfoWindow,
    SpeedDeltaBarWindow,
    SpeedWindow,
    TyreRiskWindow,
)

# Registry: (id, label, factory(source, opacity) -> widget).
# Order = display order in the scroll list.
_ModuleFactory = Callable[[LiveDataSource, float], QWidget]


def _factory(cls: type, **kw: Any) -> _ModuleFactory:
    def make(src: LiveDataSource, op: float) -> QWidget:
        return cls(src, opacity=op, **kw)
    return make


_MODULES: list[tuple[str, str, _ModuleFactory]] = [
    # Visual aids
    ("radar", "Radar",
     _factory(RadarWindow)),
    ("gmeter", "G-meter (friction circle)",
     _factory(GMeterWindow)),
    # Headline timing
    ("delta", "Delta bar vs personal best",
     _factory(DeltaBarWindow)),
    ("speed_delta", "Speed delta vs PB (same track point)",
     _factory(SpeedDeltaBarWindow)),
    ("session_info", "Session info (dynamic)",
     _factory(SessionInfoWindow)),
    ("grip", "Grip (per wheel)",
     _factory(TyreRiskWindow)),
    # Gaps
    ("gap_ahead", "Gap to driver ahead",
     _factory(GapAheadWindow)),
    ("gap_behind", "Gap to driver behind",
     _factory(GapBehindWindow)),
    # Drivetrain / dash
    ("gear", "Gear (big digit)",
     _factory(GearWindow)),
    ("rpm", "RPM bar",
     _factory(RpmWindow)),
    ("speed", "Speed (km/h)",
     _factory(SpeedWindow)),
    # Fuel
    ("fuel_pct", "Fuel %",
     _factory(FuelPctWindow)),
    ("fuel_laps", "Fuel laps remaining",
     _factory(FuelLapsRemainingWindow)),
    # Flags
    ("flags", "Flags (BLUE / YELLOW)",
     _factory(FlagsWindow)),
    # Pit lane
    ("pit_limiter", "Pit limiter (flashing + speed delta)",
     _factory(PitLimiterWindow)),
]


class LiveTab(QWidget):
    """Per-module enable/disable + light configuration."""

    def __init__(
        self,
        runner: CaptureRunner,
        signals: SignalBus,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._signals = signals

        self._source = LiveDataSource(parent=self)
        self._source.start()

        self._widgets: dict[str, QWidget | None] = {
            mid: None for mid, _label, _f in _MODULES
        }
        self._checkboxes: dict[str, QCheckBox] = {}
        self._opacity_spins: dict[str, QSpinBox] = {}

        # VR mirror is created lazily on first enable so absence of
        # SteamVR / openvr is silent when the user never opts in.
        self._vr_mirror: VrMirror | None = None
        # While the mirror is on, re-poll the SteamVR runtime so the
        # status label reflects LFS entering/leaving the VR scene live
        # (e.g. the moment LFS starts presenting to the headset).
        self._vr_status_timer = QTimer(self)
        self._vr_status_timer.setInterval(1500)
        self._vr_status_timer.timeout.connect(self._refresh_vr_status)

        # ----- Scrollable module toggles ------------------------------
        modules_box = QGroupBox(
            tr(
                "Overlay modules \u2014 drag body to move, "
                "drag bottom-right corner to resize, right-click to "
                "reset. Position and opacity persist per module.",
            ),
            self,
        )
        modules_layout = QGridLayout(modules_box)
        modules_layout.setColumnStretch(0, 1)
        settings = QSettings(ORG, APP)

        # Bulk actions: turn every overlay module off in one click,
        # useful when the screen is cluttered or when switching
        # between configurations (race vs hot-lap vs setup work).
        self._deselect_all_btn = QPushButton(tr("Deselect all"), self)
        self._deselect_all_btn.setToolTip(
            tr("Hide every overlay module.")
        )
        self._deselect_all_btn.clicked.connect(self._deselect_all_modules)
        modules_layout.addWidget(self._deselect_all_btn, 0, 0, 1, 2)
        row_offset = 1

        for row, (mid, label, _f) in enumerate(_MODULES):
            cb = QCheckBox(tr(label), self)
            cb.toggled.connect(
                lambda on, m=mid: self._toggle_module(m, on)
            )
            self._checkboxes[mid] = cb
            modules_layout.addWidget(cb, row + row_offset, 0)

            spin = QSpinBox(self)
            spin.setRange(20, 100)
            spin.setSuffix(" %")
            spin.setToolTip(
                tr(
                    "Opacity for this overlay module \u2014 persisted "
                    "between sessions.",
                ),
            )
            stored = settings.value(
                f"overlay/{mid}/opacity", None,
            )
            try:
                pct = (
                    round(float(stored) * 100)
                    if stored is not None else 85
                )
            except (TypeError, ValueError):
                pct = 85
            spin.setValue(max(20, min(100, pct)))
            spin.valueChanged.connect(
                lambda v, m=mid: self._apply_module_opacity(m, v)
            )
            self._opacity_spins[mid] = spin
            modules_layout.addWidget(spin, row + row_offset, 1)
        modules_layout.setRowStretch(len(_MODULES) + row_offset, 1)

        scroll = QScrollArea(self)
        scroll.setWidget(modules_box)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(260)

        # ----- Radar config -------------------------------------------
        self._radar_scale = QDoubleSpinBox(self)
        self._radar_scale.setRange(5.0, 200.0)
        self._radar_scale.setSingleStep(5.0)
        self._radar_scale.setSuffix(" m")
        self._radar_scale.setValue(30.0)

        self._red_m = QDoubleSpinBox(self)
        self._red_m.setRange(0.5, 50.0)
        self._red_m.setSingleStep(0.5)
        self._red_m.setSuffix(" m")
        self._red_m.setValue(2.0)

        self._yellow_m = QDoubleSpinBox(self)
        self._yellow_m.setRange(1.0, 80.0)
        self._yellow_m.setSingleStep(0.5)
        self._yellow_m.setSuffix(" m")
        self._yellow_m.setValue(5.0)

        self._white_m = QDoubleSpinBox(self)
        self._white_m.setRange(2.0, 200.0)
        self._white_m.setSingleStep(1.0)
        self._white_m.setSuffix(" m")
        self._white_m.setValue(12.0)

        radar_form = QFormLayout()
        radar_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        radar_form.addRow(tr("Scale:"), self._radar_scale)
        radar_form.addRow(tr("Red:"), self._red_m)
        radar_form.addRow(tr("Yellow:"), self._yellow_m)
        radar_form.addRow(tr("White:"), self._white_m)
        radar_box = QGroupBox(tr("Radar"), self)
        radar_box.setLayout(radar_form)

        # ----- Delta config -------------------------------------------
        self._delta_scale = QSpinBox(self)
        self._delta_scale.setRange(200, 10_000)
        self._delta_scale.setSingleStep(100)
        self._delta_scale.setSuffix(" ms")
        self._delta_scale.setValue(2000)
        delta_form = QFormLayout()
        delta_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        delta_form.addRow(tr("Full scale (\u00b1):"), self._delta_scale)
        delta_box = QGroupBox(tr("Delta bar"), self)
        delta_box.setLayout(delta_form)

        # ----- RPM redline --------------------------------------------
        self._rpm_redline = QSpinBox(self)
        self._rpm_redline.setRange(2000, 20_000)
        self._rpm_redline.setSingleStep(500)
        self._rpm_redline.setSuffix(" rpm")
        self._rpm_redline.setValue(8000)
        rpm_form = QFormLayout()
        rpm_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        rpm_form.addRow(tr("Redline:"), self._rpm_redline)
        rpm_box = QGroupBox(tr("RPM"), self)
        rpm_box.setLayout(rpm_form)

        # ----- G-meter ------------------------------------------------
        self._g_full_scale = QDoubleSpinBox(self)
        self._g_full_scale.setRange(0.5, 4.0)
        self._g_full_scale.setSingleStep(0.25)
        self._g_full_scale.setSuffix(" g")
        self._g_full_scale.setValue(2.0)
        gmeter_form = QFormLayout()
        gmeter_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        gmeter_form.addRow(tr("Full scale:"), self._g_full_scale)
        gmeter_box = QGroupBox(tr("G-meter"), self)
        gmeter_box.setLayout(gmeter_form)

        # ----- Pit limiter --------------------------------------------
        self._pit_limit_kmh = QDoubleSpinBox(self)
        self._pit_limit_kmh.setRange(20.0, 200.0)
        self._pit_limit_kmh.setSingleStep(1.0)
        self._pit_limit_kmh.setDecimals(1)
        self._pit_limit_kmh.setSuffix(" km/h")
        pit_raw = settings.value("overlay/pit_limiter/limit_kmh", 80.0)
        try:
            pit_default = max(20.0, min(200.0, float(pit_raw)))
        except (TypeError, ValueError):
            pit_default = 80.0
        self._pit_limit_kmh.setValue(pit_default)
        self._pit_limit_kmh.setToolTip(
            tr(
                "Pit-lane speed limit used by the pit-limiter overlay "
                "to compute the speed vs limit delta. LFS default: 80 km/h.",
            ),
        )
        pit_form = QFormLayout()
        pit_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        pit_form.addRow(tr("Speed limit:"), self._pit_limit_kmh)
        pit_box = QGroupBox(tr("Pit limiter"), self)
        pit_box.setLayout(pit_form)

        # ----- Session info -------------------------------------------
        self._session_compact = QCheckBox(tr("Compact layout"), self)
        self._session_compact.setToolTip(
            tr("Show condensed session info in the session overlay module."),
        )
        compact_raw = settings.value("overlay/session_info/compact", False)
        compact_on = str(compact_raw).strip().lower() in {
            "1", "true", "yes", "on"
        }
        self._session_compact.setChecked(compact_on)
        session_form = QFormLayout()
        session_form.addRow("", self._session_compact)
        session_box = QGroupBox(tr("Session info"), self)
        session_box.setLayout(session_form)

        # ----- Display compatibility ----------------------------------
        self._fullscreen_compat = QCheckBox(
            tr("Borderless / windowed-fullscreen compat"), self,
        )
        self._fullscreen_compat.setToolTip(
            tr(
                "Use regular top-most windows for overlay modules. "
                "Helps visibility when LFS runs in windowed or "
                "borderless (Full screen window) mode.\n\n"
                "NOTE: Windows cannot draw any overlay over a true "
                "DirectX exclusive-fullscreen game. If overlays are "
                "invisible in LFS fullscreen, set 'Full screen window 1' "
                "in LFS\\cfg.txt or use windowed mode.",
            ),
        )
        fs_raw = settings.value("overlay/fullscreen_compat", True)
        fs_on = str(fs_raw).strip().lower() in {
            "1", "true", "yes", "on"
        }
        self._fullscreen_compat.setChecked(fs_on)

        self._fullscreen_hint = QLabel(
            tr(
                "Tip: for overlays over LFS fullscreen, set "
                "'Full screen window 1' in LFS\\cfg.txt (exclusive "
                "fullscreen blocks all overlays system-wide).",
            ),
            self,
        )
        self._fullscreen_hint.setWordWrap(True)
        self._fullscreen_hint.setStyleSheet("color: #888; font-size: 10px;")

        display_layout = QVBoxLayout()
        display_layout.addWidget(self._fullscreen_compat)
        display_layout.addWidget(self._fullscreen_hint)
        display_box = QGroupBox(tr("Display compatibility"), self)
        display_box.setLayout(display_layout)

        # ----- VR mirror ----------------------------------------------
        # Same content layer as the desktop overlay — the mirror reads
        # render_to_image() from each visible module and uploads it to
        # SteamVR as an IVROverlay. Toggle is a no-op if SteamVR isn't
        # running or the optional ``openvr`` dep isn't installed.
        self._vr_enable = QCheckBox(
            tr("Mirror overlays to VR (SteamVR / OpenVR)"), self,
        )
        self._vr_enable.setToolTip(
            tr(
                "Show the same overlay modules inside your VR headset "
                "via SteamVR. Requires SteamVR running and the optional "
                "'openvr' Python package. Layout, opacity and content "
                "are identical to the desktop overlay.",
            ),
        )
        self._vr_enable.toggled.connect(self._toggle_vr_mirror)
        self._vr_status = QLabel("", self)
        self._vr_status.setWordWrap(True)
        vr_form = QFormLayout()
        vr_form.addRow("", self._vr_enable)
        vr_form.addRow("", self._vr_status)
        vr_box = QGroupBox(tr("VR"), self)
        vr_box.setLayout(vr_form)

        # ----- Status label -------------------------------------------
        self._status = QLabel(
            tr(
                "Start a capture, then tick the modules you want. "
                "Each window is frameless, stays on top, and remembers "
                "its last position and opacity.",
            ),
            self,
        )
        self._status.setWordWrap(True)

        # ----- Layout -------------------------------------------------
        layout = QVBoxLayout(self)
        layout.addWidget(scroll, 1)
        grid = QGridLayout()
        # Row 0: visual aids / timing
        grid.addWidget(radar_box,   0, 0)
        grid.addWidget(delta_box,   0, 1)
        # Row 1: drivetrain / forces
        grid.addWidget(rpm_box,     1, 0)
        grid.addWidget(gmeter_box,  1, 1)
        # Row 2: situational / pit
        grid.addWidget(session_box, 2, 0)
        grid.addWidget(pit_box,     2, 1)
        # Row 3: window-mode plumbing (full-width, applies to every module)
        grid.addWidget(display_box, 3, 0, 1, 2)
        # Row 4: VR mirror (full-width, applies to every module)
        grid.addWidget(vr_box,      4, 0, 1, 2)
        layout.addLayout(grid)
        layout.addWidget(self._status)

        # ----- Wiring -------------------------------------------------
        self._radar_scale.valueChanged.connect(self._apply_radar_config)
        self._red_m.valueChanged.connect(self._apply_radar_config)
        self._yellow_m.valueChanged.connect(self._apply_radar_config)
        self._white_m.valueChanged.connect(self._apply_radar_config)
        self._delta_scale.valueChanged.connect(self._apply_delta_config)
        self._rpm_redline.valueChanged.connect(self._apply_rpm_config)
        self._g_full_scale.valueChanged.connect(self._apply_g_config)
        self._pit_limit_kmh.valueChanged.connect(self._apply_pit_limiter_config)
        self._session_compact.toggled.connect(self._apply_session_overlay_mode)
        self._fullscreen_compat.toggled.connect(
            self._apply_fullscreen_compat_mode,
        )

        # Poll for capture lifecycle / live.json path.
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._poll_runner)
        self._timer.start()
        self._poll_runner()

    # ------------------------------------------------------------------
    # Toggle dispatch
    # ------------------------------------------------------------------

    def _deselect_all_modules(self) -> None:
        """Untick every module checkbox; the toggle handler closes
        each open window and persists the off state.
        """
        for cb in self._checkboxes.values():
            if cb.isChecked():
                cb.setChecked(False)

    def _toggle_vr_mirror(self, on: bool) -> None:
        """Enable/disable the SteamVR mirror of every visible module.

        Same widgets, same paint pipeline — the mirror just polls
        ``render_to_image()`` and uploads the bytes to OpenVR.
        """
        if on:
            if self._vr_mirror is None:
                # Pass a callable so the mirror always sees the *current*
                # widget map, including modules toggled on after enable.
                self._vr_mirror = VrMirror(
                    provider=lambda: dict(self._widgets),
                    parent=self,
                )
            ok = self._vr_mirror.enable()
            if ok:
                self._vr_status.setText(self._compose_vr_status_text())
                self._vr_status_timer.start()
            else:
                self._vr_status_timer.stop()
                err = (
                    self._vr_mirror._sink.init_error
                    if self._vr_mirror._sink is not None
                    else "unavailable"
                )
                self._vr_status.setText(
                    tr("VR mirror unavailable: ") + str(err or "")
                )
                # Bounce the checkbox back so the UI matches reality.
                self._vr_enable.blockSignals(True)
                self._vr_enable.setChecked(False)
                self._vr_enable.blockSignals(False)
        else:
            self._vr_status_timer.stop()
            if self._vr_mirror is not None:
                self._vr_mirror.disable()
            self._vr_status.setText("")

    def _refresh_vr_status(self) -> None:
        """Live-update the VR status label while the mirror is enabled.

        Stops itself if the mirror was torn down, so a stale timer can
        never poke a dead sink.
        """
        mirror = self._vr_mirror
        if mirror is None or not mirror.is_enabled:
            self._vr_status_timer.stop()
            return
        self._vr_status.setText(self._compose_vr_status_text())

    def _compose_vr_status_text(self) -> str:
        """Build the multi-line status shown after a successful enable.

        Surfaces three pieces of evidence so the user can confirm the
        whole VR pipeline is wired up *before* putting on the headset:

        * HMD model from SteamVR (proves we're talking to the runtime),
        * the active scene-app, with a special ``LFS scene detected``
          callout when ``LFS.exe`` owns the compositor,
        * the LFS ``cfg.txt`` VR-mode setting when LFS is configured
          for OpenVR/Oculus but isn't focused yet.
        """
        lines: list[str] = [tr("VR mirror active (SteamVR overlay).")]

        sink = (
            self._vr_mirror._sink if self._vr_mirror else None
        )
        status = sink.runtime_status() if sink is not None else None

        if status is not None:
            if status.hmd_connected:
                if status.hmd_model:
                    lines.append(
                        tr("HMD: ") + str(status.hmd_model),
                    )
                else:
                    lines.append(tr("HMD connected."))
            else:
                lines.append(tr("HMD not connected."))

            if status.scene_app_is_lfs:
                lines.append(
                    tr("LFS scene detected — overlays will composite "
                       "over your VR view."),
                )
            elif status.scene_app_name:
                lines.append(
                    tr("Scene app: ") + status.scene_app_name,
                )
            else:
                lines.append(
                    tr("No VR scene focused — start LFS in VR mode "
                       "to see overlays in your headset."),
                )

        # Read LFS cfg.txt for the selected display device (best-effort).
        # This is the pre-flight complement to the runtime scene check:
        # it tells the user whether LFS will even try to enter VR.
        lfs_dir = get_lfs_dir() or autodetect_lfs_dir()
        if lfs_dir is not None:
            try:
                vr_mode = read_lfs_vr_mode(lfs_dir)
            except Exception:
                vr_mode = None
            if vr_mode is not None:
                backend, _system = vr_mode
                lines.append(
                    tr("LFS display device: VR headset (")
                    + backend + ").",
                )
            else:
                lines.append(
                    tr("LFS is not set to a VR headset (Options \u2192 "
                       "Display \u2192 3D/VR). It renders to the flat "
                       "monitor, so SteamVR shows the desktop, not your "
                       "overlays."),
                )

        return "\n".join(lines)

    def _toggle_module(self, mid: str, on: bool) -> None:
        w = self._widgets.get(mid)
        if on and w is None:
            for k, _label, factory in _MODULES:
                if k != mid:
                    continue
                # Pass the spinbox value as fallback; the module will
                # prefer its persisted per-module opacity if present.
                op = self._opacity_spins[mid].value() / 100.0
                w = factory(self._source, op)
                self._widgets[mid] = w
                self._configure_freshly_created(mid, w)
                # Sync the spinner with whatever the module actually
                # ended up with (restored from disk on first creation).
                try:
                    pct = int(
                        w.current_opacity_pct()  # type: ignore[attr-defined]
                    )
                    spin = self._opacity_spins[mid]
                    spin.blockSignals(True)
                    spin.setValue(max(20, min(100, pct)))
                    spin.blockSignals(False)
                except (AttributeError, TypeError, ValueError):
                    pass
                break
        if w is None:
            return
        if on:
            w.show()
        else:
            w.hide()

    def _configure_freshly_created(self, mid: str, w: QWidget) -> None:
        """Push current config values into a newly-instantiated widget."""
        if mid == "radar" and isinstance(w, RadarWindow):
            w.set_radar_scale(self._radar_scale.value())
            w.set_thresholds(
                red_m=self._red_m.value(),
                yellow_m=self._yellow_m.value(),
                white_m=self._white_m.value(),
            )
        elif mid == "delta" and isinstance(w, DeltaBarWindow):
            w.set_full_scale_ms(self._delta_scale.value())
        elif mid == "rpm" and isinstance(w, RpmWindow):
            w.set_rpm_redline(self._rpm_redline.value())
        elif mid == "gmeter" and isinstance(w, GMeterWindow):
            w.set_full_scale_g(self._g_full_scale.value())
        elif mid == "pit_limiter" and isinstance(w, PitLimiterWindow):
            w.set_limit_kmh(self._pit_limit_kmh.value())
        elif mid == "session_info" and isinstance(w, SessionInfoWindow):
            w.set_compact_mode(self._session_compact.isChecked())

    # ------------------------------------------------------------------
    # Config propagation
    # ------------------------------------------------------------------

    def _apply_radar_config(self) -> None:
        w = self._widgets.get("radar")
        if isinstance(w, RadarWindow):
            w.set_radar_scale(self._radar_scale.value())
            w.set_thresholds(
                red_m=self._red_m.value(),
                yellow_m=self._yellow_m.value(),
                white_m=self._white_m.value(),
            )

    def _apply_delta_config(self) -> None:
        w = self._widgets.get("delta")
        if isinstance(w, DeltaBarWindow):
            w.set_full_scale_ms(self._delta_scale.value())

    def _apply_rpm_config(self) -> None:
        w = self._widgets.get("rpm")
        if isinstance(w, RpmWindow):
            w.set_rpm_redline(self._rpm_redline.value())

    def _apply_g_config(self) -> None:
        w = self._widgets.get("gmeter")
        if isinstance(w, GMeterWindow):
            w.set_full_scale_g(self._g_full_scale.value())

    def _apply_pit_limiter_config(self) -> None:
        w = self._widgets.get("pit_limiter")
        if isinstance(w, PitLimiterWindow):
            w.set_limit_kmh(self._pit_limit_kmh.value())
        else:
            # Module not instantiated yet — persist directly so the
            # widget picks the user's value up on creation.
            QSettings(ORG, APP).setValue(
                "overlay/pit_limiter/limit_kmh",
                float(self._pit_limit_kmh.value()),
            )

    def _apply_session_overlay_mode(self, on: bool) -> None:
        QSettings(ORG, APP).setValue("overlay/session_info/compact", bool(on))
        w = self._widgets.get("session_info")
        if isinstance(w, SessionInfoWindow):
            w.set_compact_mode(bool(on))

    def _apply_module_opacity(self, mid: str, pct: int) -> None:
        w = self._widgets.get(mid)
        if w is not None:
            w.set_opacity_pct(pct)  # type: ignore[attr-defined]
        else:
            # Module not instantiated yet — still persist so the next
            # toggle picks the user's intent up.
            QSettings(ORG, APP).setValue(
                f"overlay/{mid}/opacity", pct / 100.0,
            )

    def _apply_fullscreen_compat_mode(self, on: bool) -> None:
        QSettings(ORG, APP).setValue("overlay/fullscreen_compat", bool(on))
        # Recreate instantiated modules so new window flags take effect.
        for mid, w in list(self._widgets.items()):
            if w is None:
                continue
            was_visible = w.isVisible()
            w.close()
            self._widgets[mid] = None
            if was_visible:
                self._toggle_module(mid, True)

    # ------------------------------------------------------------------
    # Runner polling
    # ------------------------------------------------------------------

    def _poll_runner(self) -> None:
        st = self._runner.status()
        path_str = st.get("live_file") or ""
        path = Path(path_str) if path_str else None
        self._source.set_path(path)
        running = bool(st.get("running"))
        if running and path is not None:
            self._status.setText(f"Capture running. Reading: {path}")
        elif path is not None:
            self._status.setText(f"Capture stopped. Last file: {path}")
        else:
            self._status.setText(
                "Start a capture in the Capture tab, then enable the "
                "modules you want."
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._vr_status_timer.stop()
        if self._vr_mirror is not None:
            self._vr_mirror.shutdown()
            self._vr_mirror = None
        self._source.stop()
        for w in self._widgets.values():
            if w is not None:
                w.close()
        super().closeEvent(event)


__all__ = ["LiveTab"]
