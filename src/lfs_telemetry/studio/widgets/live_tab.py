"""Live tab: enable, configure, and place every overlay module.

Every datum in the live snapshot is exposed as its own toggleable,
draggable, **resizable** frameless top-most window. Tick a module's
checkbox to show it; right-click any window to reset its size; drag
the bottom-right corner to resize.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...app.capture_runner import CaptureRunner
from ...lfs_paths import QSETTINGS_APP as APP, QSETTINGS_ORG as ORG
from ..signals import SignalBus
from ..i18n import tr
from .live_data_source import LiveDataSource
from .live_modules import (
    DeltaBarWindow,
    FlagsWindow,
    FuelLapsRemainingWindow,
    FuelPctWindow,
    GMeterWindow,
    GapAheadWindow,
    GapBehindWindow,
    GearWindow,
    RadarWindow,
    RpmWindow,
    SessionInfoWindow,
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
        for row, (mid, label, _f) in enumerate(_MODULES):
            cb = QCheckBox(tr(label), self)
            cb.toggled.connect(
                lambda on, m=mid: self._toggle_module(m, on)
            )
            self._checkboxes[mid] = cb
            modules_layout.addWidget(cb, row, 0)

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
                    int(round(float(stored) * 100))
                    if stored is not None else 85
                )
            except (TypeError, ValueError):
                pct = 85
            spin.setValue(max(20, min(100, pct)))
            spin.valueChanged.connect(
                lambda v, m=mid: self._apply_module_opacity(m, v)
            )
            self._opacity_spins[mid] = spin
            modules_layout.addWidget(spin, row, 1)
        modules_layout.setRowStretch(len(_MODULES), 1)

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

        # ----- G full-scale -------------------------------------------
        self._g_full_scale = QDoubleSpinBox(self)
        self._g_full_scale.setRange(0.5, 4.0)
        self._g_full_scale.setSingleStep(0.25)
        self._g_full_scale.setSuffix(" g")
        self._g_full_scale.setValue(2.0)

        self._session_compact = QCheckBox(tr("Session overlay compact"), self)
        self._session_compact.setToolTip(
            tr("Show condensed session info in the session overlay module."),
        )
        compact_raw = settings.value("overlay/session_info/compact", False)
        compact_on = str(compact_raw).strip().lower() in {
            "1", "true", "yes", "on"
        }
        self._session_compact.setChecked(compact_on)

        misc_form = QFormLayout()
        misc_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        misc_form.addRow(tr("G-meter full scale:"), self._g_full_scale)
        misc_form.addRow("", self._session_compact)
        misc_box = QGroupBox(tr("G-meter"), self)
        misc_box.setLayout(misc_form)

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
        grid.addWidget(radar_box, 0, 0)
        grid.addWidget(delta_box, 0, 1)
        grid.addWidget(rpm_box, 1, 0)
        grid.addWidget(misc_box, 1, 1)
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
        self._session_compact.toggled.connect(self._apply_session_overlay_mode)

        # Poll for capture lifecycle / live.json path.
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._poll_runner)
        self._timer.start()
        self._poll_runner()

    # ------------------------------------------------------------------
    # Toggle dispatch
    # ------------------------------------------------------------------

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

    def closeEvent(self, event) -> None:  # noqa: N802
        for w in self._widgets.values():
            if w is not None:
                w.close()
        super().closeEvent(event)


__all__ = ["LiveTab"]
