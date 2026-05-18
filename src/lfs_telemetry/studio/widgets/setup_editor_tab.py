"""In-app garage editor — the LFS F11 garage as a Qt form.

This widget lets the user mirror in-app the setup actually loaded in
LFS. Workflow:

1. The user selects a lap on the left dock; we resolve the car and
   load ``<car>_CAR_info.bin`` via :func:`load_car_info_bin_for`. The
   bin is the *baseline* — whatever the user had in the LFS garage the
   last time they exported it.
2. We pre-fill every editable field with
   :func:`setup_overrides.from_baseline`, so the form opens on the
   exact numbers LFS shows.
3. The user tweaks any value — brake balance, springs, tyre pressures,
   gear ratios, etc. — to match what they actually drove. On every
   commit (``Apply`` button or focus-out) we build a patched
   :class:`CarInfoBin` with :func:`setup_overrides.apply` and broadcast
   it through ``signals.setup_overrides_changed``.

The form intentionally mirrors the LFS garage panels (Brakes,
Suspension, Drivetrain, Tyres, Chassis) and uses display units the
user sees in the game: degrees for camber/toe, psi for tyre pressures,
N/mm for springs/ARBs, percentages for brake balance and weight
distribution. The dataclass underneath keeps everything in SI.

Persistence: none. Per the project decision the editor lives only in
memory; the user re-enters changes whenever they relaunch. Adding
QSettings persistence later is a localised change to ``_collect`` /
``_apply``.
"""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...telemetry import LapTelemetry
from ...telemetry.car_info_bin import CarInfoBin
from ...telemetry.observables import load_car_info_bin_for
from ...telemetry.setup_overrides import (
    SetupOverrides,
    from_baseline,
)
from ...telemetry.setup_overrides import (
    apply as apply_overrides,
)
from ..i18n import tr
from ..models import LapLoader
from ..signals import SignalBus
from ..theme import MUTED_COLOR, TEXT_COLOR

# Unit conversions (LFS internal SI ⇄ garage display).
_KPA_PER_PSI = 6.894757293168361     # psi → kPa
_RAD_PER_DEG = math.pi / 180.0


def _spin(
    minimum: float,
    maximum: float,
    step: float,
    decimals: int,
    suffix: str = "",
) -> QDoubleSpinBox:
    """Compact helper for a configured :class:`QDoubleSpinBox`."""
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    if suffix:
        box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    box.setAlignment(Qt.AlignmentFlag.AlignRight)
    return box


class SetupEditorTab(QWidget):
    """Editable replica of the LFS garage for the active car.

    The widget is *passive* until a lap is selected and its
    ``<car>_CAR_info.bin`` is available on the search path. Until then
    it shows a hint that mirrors the Baseline tab's empty state.
    """

    def __init__(
        self,
        loader: LapLoader,
        signals: SignalBus,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._loader = loader
        self._signals = signals
        self._first_lap: LapTelemetry | None = None
        self._first_path: Path | None = None
        self._baseline: CarInfoBin | None = None
        self._car_key: str = ""
        # Gear-ratio spinboxes are recreated whenever forward_gears
        # changes (different cars have different gear counts).
        self._gear_boxes: list[QDoubleSpinBox] = []

        # ---------- Header / hint ------------------------------------
        self._hint = QLabel(
            tr("Select a lap on the left to load the car's garage."),
            self,
        )
        self._hint.setStyleSheet(f"color: {MUTED_COLOR};")
        self._hint.setWordWrap(True)

        # ---------- Build all spinboxes upfront ----------------------
        self._build_brakes_group()
        self._build_suspension_group()
        self._build_tyres_group()
        self._build_drivetrain_group()
        self._build_chassis_group()
        self._gears_group = self._build_gears_group()

        # ---------- Buttons -----------------------------------------
        self._apply_btn = QPushButton(tr("Apply overrides"))
        self._apply_btn.setToolTip(
            tr(
                "Broadcast the current values as the active setup"
                " override. Other tabs that consume CAR_info will see"
                " these numbers instead of the raw on-disk export."
            )
        )
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        self._reset_btn = QPushButton(tr("Reset to imported"))
        self._reset_btn.setToolTip(
            tr(
                "Re-read the values from the on-disk CAR_info.bin"
                " export and discard local edits."
            )
        )
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        self._status = QLabel("", self)
        self._status.setStyleSheet(f"color: {MUTED_COLOR};")

        toolbar = QHBoxLayout()
        toolbar.addWidget(self._status, 1)
        toolbar.addWidget(self._reset_btn)
        toolbar.addWidget(self._apply_btn)

        # ---------- Scroll container --------------------------------
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(8, 8, 8, 8)
        body_layout.addWidget(self._hint)
        body_layout.addWidget(self._brakes_group)
        body_layout.addWidget(self._suspension_group)
        body_layout.addWidget(self._tyres_group)
        body_layout.addWidget(self._drivetrain_group)
        body_layout.addWidget(self._gears_group)
        body_layout.addWidget(self._chassis_group)
        body_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ color: {TEXT_COLOR}; }}")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll, 1)
        root.addLayout(toolbar)

        self._set_enabled(False)
        signals.laps_selected.connect(self._on_laps_selected)
        loader.lap_loaded.connect(self._on_lap_loaded)

    # ------------------------------------------------------------------
    # Group builders
    # ------------------------------------------------------------------

    def _build_brakes_group(self) -> None:
        self._brake_strength = _spin(0.0, 20000.0, 25.0, 0, " Nm")
        self._brake_balance = _spin(0.0, 100.0, 0.5, 1, " % front")
        self._parallel_steer = _spin(0.0, 100.0, 1.0, 0, " %")

        g = QGroupBox(tr("Brakes & steering"))
        form = QFormLayout(g)
        form.addRow(tr("Max force"), self._brake_strength)
        form.addRow(tr("Balance"), self._brake_balance)
        form.addRow(tr("Parallel steer"), self._parallel_steer)
        self._brakes_group = g

    def _build_suspension_group(self) -> None:
        # Per-axle spring/damper/ARB/camber/toe — exactly how the LFS
        # garage groups them.
        self._fr_camber = _spin(-10.0, 10.0, 0.1, 2, " °")
        self._rr_camber = _spin(-10.0, 10.0, 0.1, 2, " °")
        self._fr_toe = _spin(-5.0, 5.0, 0.05, 2, " °")
        self._rr_toe = _spin(-5.0, 5.0, 0.05, 2, " °")
        self._fr_spring = _spin(0.0, 500.0, 1.0, 1, " N/mm")
        self._rr_spring = _spin(0.0, 500.0, 1.0, 1, " N/mm")
        self._fr_bump = _spin(0.0, 50000.0, 50.0, 0, " N·s/m")
        self._rr_bump = _spin(0.0, 50000.0, 50.0, 0, " N·s/m")
        self._fr_rebound = _spin(0.0, 50000.0, 50.0, 0, " N·s/m")
        self._rr_rebound = _spin(0.0, 50000.0, 50.0, 0, " N·s/m")
        self._fr_arb = _spin(0.0, 500.0, 1.0, 1, " N/mm")
        self._rr_arb = _spin(0.0, 500.0, 1.0, 1, " N/mm")

        g = QGroupBox(tr("Suspension (per axle)"))
        grid = QGridLayout(g)
        grid.addWidget(QLabel(""), 0, 0)
        grid.addWidget(QLabel(f"<b>{tr('Front')}</b>"), 0, 1)
        grid.addWidget(QLabel(f"<b>{tr('Rear')}</b>"), 0, 2)
        rows = [
            (tr("Camber"), self._fr_camber, self._rr_camber),
            (tr("Toe-in"), self._fr_toe, self._rr_toe),
            (tr("Spring rate"), self._fr_spring, self._rr_spring),
            (tr("Damper bump"), self._fr_bump, self._rr_bump),
            (tr("Damper rebound"), self._fr_rebound, self._rr_rebound),
            (tr("Anti-roll bar"), self._fr_arb, self._rr_arb),
        ]
        for i, (lbl, fr, rr) in enumerate(rows, start=1):
            grid.addWidget(QLabel(lbl), i, 0)
            grid.addWidget(fr, i, 1)
            grid.addWidget(rr, i, 2)
        self._suspension_group = g

    def _build_tyres_group(self) -> None:
        self._fr_press = _spin(5.0, 60.0, 0.1, 1, " psi")
        self._rr_press = _spin(5.0, 60.0, 0.1, 1, " psi")

        g = QGroupBox(tr("Tyres (per axle)"))
        grid = QGridLayout(g)
        grid.addWidget(QLabel(f"<b>{tr('Front')}</b>"), 0, 1)
        grid.addWidget(QLabel(f"<b>{tr('Rear')}</b>"), 0, 2)
        grid.addWidget(QLabel(tr("Pressure")), 1, 0)
        grid.addWidget(self._fr_press, 1, 1)
        grid.addWidget(self._rr_press, 1, 2)
        self._tyres_group = g

    def _build_drivetrain_group(self) -> None:
        self._final_drive = _spin(1.0, 12.0, 0.01, 3, "")
        self._drive_eff = _spin(0.0, 100.0, 0.5, 1, " %")
        self._torque_split = _spin(0.0, 100.0, 1.0, 1, " % front")
        g = QGroupBox(tr("Drivetrain"))
        form = QFormLayout(g)
        form.addRow(tr("Final drive"), self._final_drive)
        form.addRow(tr("Drivetrain efficiency"), self._drive_eff)
        form.addRow(tr("Torque split (AWD)"), self._torque_split)
        self._drivetrain_group = g

    def _build_gears_group(self) -> QGroupBox:
        # The contents are rebuilt every time a baseline with a
        # different ``forward_gears`` is loaded.
        g = QGroupBox(tr("Gear ratios"))
        self._gears_layout = QFormLayout(g)
        return g

    def _rebuild_gear_boxes(self, count: int) -> None:
        # Drop any previous spinboxes. ``removeRow`` detaches the
        # widget from the layout but does *not* delete the underlying
        # QObject; on a long session of car switches that would leak
        # parented widgets. Explicit ``deleteLater`` keeps memory flat.
        for box in self._gear_boxes:
            box.setParent(None)
            box.deleteLater()
        while self._gears_layout.rowCount():
            self._gears_layout.removeRow(0)
        self._gear_boxes = []
        for i in range(count):
            box = _spin(0.20, 10.0, 0.01, 3, "")
            self._gears_layout.addRow(tr("Gear {n}").format(n=i + 1), box)
            self._gear_boxes.append(box)

    def _build_chassis_group(self) -> None:
        self._passengers = QSpinBox()
        self._passengers.setRange(0, 4)
        self._weight_dist = _spin(0.0, 100.0, 0.5, 1, " % front")
        self._fuel_capacity = _spin(0.0, 200.0, 0.5, 1, " L")
        g = QGroupBox(tr("Chassis & fuel"))
        form = QFormLayout(g)
        form.addRow(tr("Passengers / ballast"), self._passengers)
        form.addRow(tr("Weight distribution"), self._weight_dist)
        form.addRow(tr("Fuel tank capacity"), self._fuel_capacity)
        self._chassis_group = g

    # ------------------------------------------------------------------
    # Lap selection lifecycle
    # ------------------------------------------------------------------

    def _on_laps_selected(self, paths: list[Path]) -> None:
        self._first_lap = None
        self._first_path = paths[0] if paths else None
        if not paths:
            self._car_key = ""
            self._baseline = None
            self._hint.setText(
                tr("Select a lap on the left to load the car's garage.")
            )
            self._set_enabled(False)
            return
        self._loader.request(paths[0])
        self._hint.setText(
            tr("Loading garage for {name}\u2026").format(name=paths[0].name)
        )

    def _on_lap_loaded(self, path: Path, lap: LapTelemetry) -> None:
        if path != self._first_path:
            return
        self._first_lap = lap
        car_key = ""
        if lap.summary:
            car_key = str(lap.summary.get("car") or "").upper().strip()
        if not car_key:
            self._hint.setText(
                tr("Lap has no car id in its summary \u2014 cannot load garage.")
            )
            self._baseline = None
            self._car_key = ""
            self._set_enabled(False)
            return
        baseline = load_car_info_bin_for(car_key)
        if baseline is None:
            self._hint.setText(
                tr(
                    "No <code>{key}_CAR_info.bin</code> found on the"
                    " search path. Use <b>Import from LFS folder\u2026</b>"
                    " on the Baseline tab first."
                ).format(key=car_key)
            )
            self._baseline = None
            self._car_key = car_key
            self._set_enabled(False)
            return
        self._car_key = car_key
        self._baseline = baseline
        self._hint.setText(
            tr(
                "<b>{key}</b> \u2014 loaded from"
                " <code>{key}_CAR_info.bin</code>. Edit any field and"
                " press <b>Apply overrides</b> to publish it as the"
                " active setup."
            ).format(key=car_key)
        )
        self._populate_from(from_baseline(baseline), baseline.forward_gears)
        self._set_enabled(True)

    # ------------------------------------------------------------------
    # Populate / collect
    # ------------------------------------------------------------------

    def _populate_from(
        self, ov: SetupOverrides, forward_gears: int,
    ) -> None:
        """Push the SI values from ``ov`` into the display spinboxes."""
        # Brakes / steering
        self._brake_strength.setValue(float(ov.brake_strength_nm or 0.0))
        self._brake_balance.setValue(
            float((ov.brake_balance_front or 0.0) * 100.0)
        )
        self._parallel_steer.setValue(
            float((ov.parallel_steer or 0.0) * 100.0)
        )
        # Suspension geometry — rad → deg.
        self._fr_camber.setValue(math.degrees(ov.front_camber_rad or 0.0))
        self._rr_camber.setValue(math.degrees(ov.rear_camber_rad or 0.0))
        self._fr_toe.setValue(math.degrees(ov.front_toe_in_rad or 0.0))
        self._rr_toe.setValue(math.degrees(ov.rear_toe_in_rad or 0.0))
        # Springs / ARB — N/m → N/mm.
        self._fr_spring.setValue((ov.front_spring_const or 0.0) / 1000.0)
        self._rr_spring.setValue((ov.rear_spring_const or 0.0) / 1000.0)
        self._fr_arb.setValue((ov.front_anti_roll or 0.0) / 1000.0)
        self._rr_arb.setValue((ov.rear_anti_roll or 0.0) / 1000.0)
        # Dampers stay in N·s/m to match LFS's "click" range.
        self._fr_bump.setValue(float(ov.front_damping_comp or 0.0))
        self._rr_bump.setValue(float(ov.rear_damping_comp or 0.0))
        self._fr_rebound.setValue(float(ov.front_damping_rebound or 0.0))
        self._rr_rebound.setValue(float(ov.rear_damping_rebound or 0.0))
        # Tyres — kPa → psi.
        self._fr_press.setValue(
            (ov.front_tyre_pressure_kpa or 0.0) / _KPA_PER_PSI
        )
        self._rr_press.setValue(
            (ov.rear_tyre_pressure_kpa or 0.0) / _KPA_PER_PSI
        )
        # Drivetrain
        self._final_drive.setValue(float(ov.final_drive or 0.0))
        self._drive_eff.setValue(
            float((ov.drivetrain_efficiency or 0.0) * 100.0)
        )
        self._torque_split.setValue(
            float((ov.torque_split or 0.0) * 100.0)
        )
        # Gears (rebuild + fill)
        self._rebuild_gear_boxes(forward_gears)
        gears = list(ov.gear_ratios or ())
        for i, box in enumerate(self._gear_boxes):
            box.setValue(float(gears[i]) if i < len(gears) else 1.0)
        # Chassis
        self._passengers.setValue(int(ov.passengers or 0))
        self._weight_dist.setValue(
            float((ov.weight_dist_front or 0.0) * 100.0)
        )
        self._fuel_capacity.setValue(float(ov.fuel_capacity_l or 0.0))

    def _collect(self) -> SetupOverrides:
        """Read all spinboxes back into SI-unit overrides."""
        return SetupOverrides(
            passengers=int(self._passengers.value()),
            weight_dist_front=self._weight_dist.value() / 100.0,
            fuel_capacity_l=self._fuel_capacity.value(),
            brake_strength_nm=self._brake_strength.value(),
            brake_balance_front=self._brake_balance.value() / 100.0,
            parallel_steer=self._parallel_steer.value() / 100.0,
            final_drive=self._final_drive.value(),
            gear_ratios=tuple(b.value() for b in self._gear_boxes),
            drivetrain_efficiency=self._drive_eff.value() / 100.0,
            torque_split=self._torque_split.value() / 100.0,
            front_camber_rad=self._fr_camber.value() * _RAD_PER_DEG,
            rear_camber_rad=self._rr_camber.value() * _RAD_PER_DEG,
            front_toe_in_rad=self._fr_toe.value() * _RAD_PER_DEG,
            rear_toe_in_rad=self._rr_toe.value() * _RAD_PER_DEG,
            front_spring_const=self._fr_spring.value() * 1000.0,
            rear_spring_const=self._rr_spring.value() * 1000.0,
            front_damping_comp=self._fr_bump.value(),
            rear_damping_comp=self._rr_bump.value(),
            front_damping_rebound=self._fr_rebound.value(),
            rear_damping_rebound=self._rr_rebound.value(),
            front_anti_roll=self._fr_arb.value() * 1000.0,
            rear_anti_roll=self._rr_arb.value() * 1000.0,
            front_tyre_pressure_kpa=self._fr_press.value() * _KPA_PER_PSI,
            rear_tyre_pressure_kpa=self._rr_press.value() * _KPA_PER_PSI,
        )

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def _on_apply_clicked(self) -> None:
        if self._baseline is None or not self._car_key:
            return
        try:
            patched = apply_overrides(self._baseline, self._collect())
        except ValueError as exc:
            self._status.setText(tr("Invalid setup: {error}").format(error=exc))
            return
        self._status.setText(
            tr("Overrides applied \u2014 other tabs will use the new values.")
        )
        self._signals.setup_overrides_changed.emit(self._car_key, patched)

    def _on_reset_clicked(self) -> None:
        if self._baseline is None:
            return
        self._populate_from(
            from_baseline(self._baseline), self._baseline.forward_gears,
        )
        self._status.setText(tr("Reset to imported CAR_info.bin values."))
        # Broadcast a None so listeners fall back to the raw bin.
        self._signals.setup_overrides_changed.emit(self._car_key, None)

    # ------------------------------------------------------------------
    # Enable / disable the whole form
    # ------------------------------------------------------------------

    def _set_enabled(self, enabled: bool) -> None:
        for g in (
            self._brakes_group, self._suspension_group, self._tyres_group,
            self._drivetrain_group, self._gears_group, self._chassis_group,
        ):
            g.setEnabled(enabled)
        self._apply_btn.setEnabled(enabled)
        self._reset_btn.setEnabled(enabled)


__all__ = ["SetupEditorTab"]
