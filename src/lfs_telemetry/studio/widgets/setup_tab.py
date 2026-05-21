"""Setup tab — full car setup for the first selected lap.

The public widget exported here is :class:`SetupTab`, a thin container
that mounts a :class:`QTabWidget` with two sub-tabs:

* **Baseline** — :class:`SetupBaselineTab`, the historical HTML report
  parsed from ``<car>_CAR_info.bin`` (unchanged behaviour).
* **Garage editor** — :class:`SetupEditorTab`, the in-app editor for
  setup overrides.

``SetupBaselineTab`` subscribes to ``laps_selected`` on the shared
:class:`SignalBus`, takes the first selected lap's ``summary["car"]``
short-name, locates the matching ``<car>_CAR_info.bin`` export via
:func:`telemetry.observables.load_car_info_bin_for`, and renders the
parsed :class:`telemetry.car_info_bin.CarInfoBin` as a structured
HTML report (read-only ``QTextEdit``). The report groups data by
domain (Chassis & weight distribution, Engine, Drivetrain, Brakes,
Tyres + suspension per wheel, Fuel tank) so a race engineer can scan
the snapshot the same way as in the LFS in-game F11 setup screen.
"""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...telemetry import LapTelemetry
from ...telemetry.car_info_bin import CarInfoBin
from ...telemetry.observables import (
    import_car_info_bin,
    load_car_info_bin_for,
    user_car_info_bin_dir,
)
from ..models import LapLoader
from ..signals import SignalBus
from ..theme import MUTED_COLOR, PANEL_COLOR, TEXT_COLOR, WHEEL_ORDER_UI
from ._format import format_finite, format_signed_finite

_WHEEL_ORDER = WHEEL_ORDER_UI  # canonical UI order, re-exported locally

# Unit conversions used by the LFS in-game garage display.
_M_TO_INCH = 1.0 / 0.0254     # metres → inches
_KPA_TO_PSI = 0.145037738      # kPa → psi (exact ISO factor)


def _fmt(v: float, digits: int = 2, suffix: str = "") -> str:
    return format_finite(v, digits, suffix)


def _signed(v: float, digits: int = 1, suffix: str = "") -> str:
    """LFS-style signed value (camber, toe, etc.): +1.5° / -2.3°."""
    return format_signed_finite(v, digits, suffix)


def _deg(rad: float) -> float:
    return math.degrees(rad) if math.isfinite(rad) else float("nan")


def _rim_inches(radius_m: float) -> float:
    """Rim diameter in inches — the figure LFS shows next to the wheel."""
    if not math.isfinite(radius_m):
        return float("nan")
    return radius_m * 2.0 * _M_TO_INCH


class SetupBaselineTab(QWidget):
    """HTML report of the full setup parsed from CAR_info.bin."""

    def __init__(
        self,
        loader: LapLoader,
        signals: SignalBus,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._loader = loader
        self._signals = signals
        self._requested: list[Path] = []
        self._first_lap: LapTelemetry | None = None
        self._first_path: Path | None = None

        self._view = QTextEdit(self)
        self._view.setReadOnly(True)
        self._view.setStyleSheet(
            f"QTextEdit {{ background:{PANEL_COLOR};"
            f" color:{TEXT_COLOR}; border:0; padding:8px; }}"
        )
        self._view.setHtml(self._empty_html())

        # Toolbar: "Import CAR_info.bin\u2026" file picker. This is the
        # one-click escape hatch for the most common support request
        # ("setup tab errors out for FBM / stock car X"): the user
        # exports the .bin from LFS once via the Programmer mode and
        # drops it in here; we copy it into the writable search dir,
        # clear the per-key cache, and re-render immediately.
        self._import_btn = QPushButton("Import CAR_info.bin\u2026", self)
        self._import_btn.setToolTip(
            "Pick a *_CAR_info.bin export and copy it into the\n"
            f"app's search path ({user_car_info_bin_dir()})."
        )
        self._import_btn.clicked.connect(self._on_import_clicked)
        self._import_lfs_btn = QPushButton("Import from LFS folder…", self)
        self._import_lfs_btn.setToolTip(
            "Scan your LFS install's data/ folder for every\n"
            "<car>_CAR_info.bin export and copy them all at once."
        )
        self._import_lfs_btn.clicked.connect(self._on_import_lfs_clicked)
        self._gen_lfs_btn = QPushButton("Generate CAR_info.bin (LFS)…", self)
        self._gen_lfs_btn.setToolTip(
            "Launch LFS.exe in Programmer Mode so you can save\n"
            "<car>_CAR_info.bin from the in-game menu."
        )
        self._gen_lfs_btn.clicked.connect(self._on_gen_lfs_clicked)
        toolbar = QHBoxLayout()
        toolbar.addStretch(1)
        toolbar.addWidget(self._gen_lfs_btn)
        toolbar.addWidget(self._import_lfs_btn)
        toolbar.addWidget(self._import_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(toolbar)
        layout.addWidget(self._view)

        signals.laps_selected.connect(self._on_laps_selected)
        loader.lap_loaded.connect(self._on_lap_loaded)

    # ------------------------------------------------------------------
    # Import workflow
    # ------------------------------------------------------------------

    def _on_import_clicked(self) -> None:
        car_hint = self._current_car_key()
        title = (
            f"Locate {car_hint}_CAR_info.bin\u2026" if car_hint
            else "Locate *_CAR_info.bin\u2026"
        )
        chosen, _ = QFileDialog.getOpenFileName(
            self, title, str(Path.home()),
            "LFS CAR info (*_CAR_info.bin *.bin);;All files (*.*)",
        )
        if not chosen:
            return
        try:
            dst, _info = import_car_info_bin(
                Path(chosen),
                target_key=car_hint or None,
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Import failed",
                f"Could not import {chosen}:\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            return
        QMessageBox.information(
            self, "Imported",
            f"Saved to:\n{dst}\n\n"
            "The setup will now refresh.",
        )
        self._refresh()

    def _on_import_lfs_clicked(self) -> None:
        from ._lfs_bin_import import import_bins_from_lfs_folder
        if import_bins_from_lfs_folder(self) > 0:
            self._refresh()

    def _on_gen_lfs_clicked(self) -> None:
        from ._lfs_bin_import import launch_lfs_programmer_mode
        launch_lfs_programmer_mode(self)

    def _current_car_key(self) -> str:
        lap = self._first_lap
        if lap is None or not lap.summary:
            return ""
        return str(lap.summary.get("car") or "").upper().strip()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_laps_selected(self, paths: list[Path]) -> None:
        self._requested = list(paths)
        self._first_lap = None
        self._first_path = paths[0] if paths else None
        if not paths:
            self._view.setHtml(self._empty_html())
            return
        self._loader.request(paths[0])
        # While the lap loads, hint the user.
        self._view.setHtml(
            f"<p style='color:{MUTED_COLOR};'>Loading setup for "
            f"<b>{paths[0].name}</b>…</p>"
        )

    def _on_lap_loaded(self, path: Path, lap: LapTelemetry) -> None:
        if path != self._first_path:
            return
        self._first_lap = lap
        self._refresh()

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _empty_html(self) -> str:
        return (
            f"<p style='color:{MUTED_COLOR};'>"
            f"Select a lap on the left to view its setup."
            f"</p>"
        )

    def _refresh(self) -> None:
        lap = self._first_lap
        if lap is None:
            self._view.setHtml(self._empty_html())
            return
        car_key = ""
        if lap.summary:
            car_key = str(lap.summary.get("car") or "").upper()
        if not car_key:
            self._view.setHtml(
                f"<p style='color:{MUTED_COLOR};'>"
                f"Lap has no car id in its summary — cannot resolve setup."
                f"</p>"
            )
            return
        info = load_car_info_bin_for(car_key)
        if info is None:
            search_dir = user_car_info_bin_dir()
            self._view.setHtml(
                f"<h3 style='margin:0 0 6px 0;'>Setup data not"
                f" available for {car_key}</h3>"
                f"<p>The full setup view needs the"
                f" <code>{car_key}_CAR_info.bin</code> file that LFS"
                f" exports from its garage. Two ways to provide it:</p>"
                f"<ol>"
                f"<li>Press <b>Import CAR_info.bin\u2026</b> above and"
                f" pick the file you exported from LFS.</li>"
                f"<li>Or copy it manually to"
                f" <code>{search_dir}</code> and reselect the lap.</li>"
                f"</ol>"
                f"<p style='color:{MUTED_COLOR};margin-top:10px;'>"
                f"To export from LFS: open the car in the garage, run"
                f" LFS in Programmer mode (<code>LFS.exe /prog</code>)"
                f" and use <b>Save CAR_info.bin</b>; the file appears"
                f" under <code>LFS/data/</code>. Advanced users can also"
                f" set <code>$LFS_TELEMETRY_CAR_INFO_DIR</code> to a"
                f" shared folder."
                f"</p>"
            )
            return
        self._view.setHtml(self._render_html(car_key, info))

    def _render_html(self, car_key: str, info: CarInfoBin) -> str:
        parts: list[str] = []
        parts.append(
            f"<h2 style='margin:0 0 6px 0;'>{car_key} setup</h2>"
            f"<p style='color:{MUTED_COLOR};margin:0 0 10px 0;'>"
            f"Source: <code>{info.short_name}_CAR_info.bin</code>"
            f" (file v{info.file_version}) — fields and units mapped per"
            f" the official LFS specs at"
            f" <code>lfs.net/programmer/carinfo</code> and"
            f" <code>lfs.net/programmer/raf</code>."
            f"</p>"
        )
        parts.append(self._chassis_section(info))
        parts.append(self._engine_section(info))
        parts.append(self._drivetrain_section(info))
        parts.append(self._brakes_section(info))
        parts.append(self._tyres_section(info))
        parts.append(self._suspension_section(info))
        parts.append(self._tuning_metrics_section(info))
        parts.append(self._fuel_section(info))
        parts.append(self._balance_reference_section())
        parts.append(
            f"<p style='color:{MUTED_COLOR};margin:14px 0 0 0;"
            f"font-size:11px;'>"
            f"Tuning conventions and adjustability flags follow the"
            f" official LFS guides:"
            f" <code>en.lfsmanual.net/wiki/Basic_Setup_Guide</code>,"
            f" <code>en.lfsmanual.net/wiki/Advanced_Setup_Guide</code>"
            f" and <code>en.lfsmanual.net/wiki/Technical_Reference</code>."
            f" Differential type / slip limits / preload, wing angles"
            f" and ride height are setup-screen only and are not"
            f" exported in <code>CAR_info.bin</code>."
            f"</p>"
        )
        return "<div>" + "".join(parts) + "</div>"

    # ----- sections ---------------------------------------------------

    @staticmethod
    def _section(title: str, rows: list[tuple[str, str]]) -> str:
        body = "".join(
            f"<tr><td style='padding:1px 14px 1px 0;color:#9aa;'>"
            f"{label}</td><td>{value}</td></tr>"
            for label, value in rows
        )
        return (
            f"<h3 style='margin:10px 0 4px 0;'>{title}</h3>"
            f"<table style='border-collapse:collapse;'>{body}</table>"
        )

    def _chassis_section(self, info: CarInfoBin) -> str:
        # LFS displays wheelbase / track in mm in the chassis info panel.
        return self._section("Chassis", [
            ("Mass", _fmt(info.mass_kg, 1, " kg")),
            ("Wheelbase (LFS mm)",
             _fmt(info.wheelbase_m * 1000.0, 0, " mm")),
            ("Track front (LFS mm)",
             _fmt(info.track_front_m * 1000.0, 0, " mm")),
            ("Track rear (LFS mm)",
             _fmt(info.track_rear_m * 1000.0, 0, " mm")),
            ("Weight dist. front (LFS %)",
             _fmt(info.weight_dist_front * 100.0, 1, " %")),
            ("CG height",
             _fmt(info.cg_height_m * 1000.0, 0, " mm")),
            ("Passengers", str(info.passengers)),
        ])

    def _engine_section(self, info: CarInfoBin) -> str:
        return self._section("Engine", [
            ("Max torque",
             f"{_fmt(info.max_torque_nm, 1, ' N·m')} @"
             f" {_fmt(info.max_torque_rpm, 0, ' rpm')}"),
            ("Max power",
             f"{_fmt(info.max_power_kw, 1, ' kW')} @"
             f" {_fmt(info.max_power_rpm, 0, ' rpm')}"),
        ])

    def _drivetrain_section(self, info: CarInfoBin) -> str:
        gears_html = []
        # gear_ratios[0] = reverse; the rest are forward gears.
        ratios = info.gear_ratios
        if ratios:
            gears_html.append(f"R: {_fmt(ratios[0], 3)}")
            for i, r in enumerate(ratios[1:], start=1):
                gears_html.append(f"{i}: {_fmt(r, 3)}")
        rows = [
            ("Drive", info.drive),
            ("Forward gears", str(info.forward_gears)),
            ("Final drive", _fmt(info.final_drive, 3)),
            ("Drivetrain efficiency",
             _fmt(info.drivetrain_efficiency * 100.0, 1, " %")),
            ("Gear ratios", " &nbsp; ".join(gears_html) or "—"),
        ]
        if info.drive == "AWD":
            rows.append(
                ("Torque split front",
                 _fmt(info.torque_split * 100.0, 1, " %")),
            )
        return self._section("Drivetrain", rows)

    def _brakes_section(self, info: CarInfoBin) -> str:
        # LFS shows brake balance as "rear–front bias" (% = front).
        # The in-game slider is clamped to 5–95 %. Basic Setup Guide
        # recommends a typical road/race baseline of ~60% F / 40% R
        # (front and rear should lock at the same time).
        bal_pct = info.brake_balance_front * 100.0
        delta = bal_pct - 60.0
        bias_tag = ("on benchmark" if abs(delta) < 2.0
                    else (f"{delta:+.1f}% vs 60% F benchmark"))
        bal_lfs = (f"F {bal_pct:.1f}% / R {100.0 - bal_pct:.1f}%"
                   f" · rear–front bias = {bal_pct:.1f}%"
                   f" <span style='color:{MUTED_COLOR};'>({bias_tag})</span>")
        return self._section("Brakes & steering", [
            ("Max per wheel",
             _fmt(info.brake_strength_nm, 0, " N·m")),
            ("Brake balance (LFS, 5–95%)", bal_lfs),
            ("Parallel steer (LFS %)",
             f"{_fmt(info.parallel_steer * 100.0, 1, ' %')}"
             f" — 100% parallel / 0% full Ackermann"),
        ])

    def _wheels_by_label(self, info: CarInfoBin) -> dict:
        return {w.name: w for w in info.wheels}

    def _tyres_section(self, info: CarInfoBin) -> str:
        wheels = self._wheels_by_label(info)
        head = (
            "<tr><th></th>"
            + "".join(
                f"<th style='padding:1px 10px;color:#9aa;'>{w}</th>"
                for w in _WHEEL_ORDER
            )
            + "</tr>"
        )

        def _row(label: str, fmt) -> str:
            cells = "".join(
                f"<td style='padding:1px 10px;'>"
                f"{fmt(wheels[w])}</td>"
                for w in _WHEEL_ORDER
            )
            return (
                f"<tr><td style='padding:1px 14px 1px 0;color:#9aa;'>"
                f"{label}</td>{cells}</tr>"
            )

        body = (
            head
            + _row("Compound id", lambda w: str(w.tyre_type))
            + _row("Width (LFS mm)",
                   lambda w: _fmt(w.tyre_width_m * 1000.0, 0, " mm"))
            + _row("Profile (LFS %)",
                   lambda w: _fmt(w.sidewall_height_prop * 100.0, 0, " %"))
            + _row("Rim (LFS in)",
                   lambda w: _fmt(_rim_inches(w.rim_radius_m), 1, '"'))
            + _row("Rim width",
                   lambda w: _fmt(w.rim_width_m * _M_TO_INCH, 1, '"'))
            # Setup-screen pressure (kPa default, psi alt unit).
            + _row("Pressure (LFS kPa)",
                   lambda w: f"{_fmt(w.tyre_pressure_kpa, 1, ' kPa')}"
                   f" ({_fmt(w.tyre_pressure_kpa * _KPA_TO_PSI, 1, ' psi')})")
            + _row("Air temp",
                   lambda w: _fmt(w.air_temp_c, 1, " °C"))
            + _row("Vert. spring",
                   lambda w: _fmt(w.tyre_vert_spring / 1000.0, 1, " N/mm"))
        )
        return (
            "<h3 style='margin:10px 0 4px 0;'>Tyres"
            " <span style='color:#9aa;font-weight:normal;'>"
            "(LFS in-game units)</span></h3>"
            "<table style='border-collapse:collapse;'>" + body + "</table>"
        )

    def _suspension_section(self, info: CarInfoBin) -> str:
        wheels = self._wheels_by_label(info)
        head = (
            "<tr><th></th>"
            + "".join(
                f"<th style='padding:1px 10px;color:#9aa;'>{w}</th>"
                for w in _WHEEL_ORDER
            )
            + "</tr>"
        )

        def _row(label: str, fmt) -> str:
            cells = "".join(
                f"<td style='padding:1px 10px;'>"
                f"{fmt(wheels[w])}</td>"
                for w in _WHEEL_ORDER
            )
            return (
                f"<tr><td style='padding:1px 14px 1px 0;color:#9aa;'>"
                f"{label}</td>{cells}</tr>"
            )

        body = (
            head
            # LFS in-game F11 setup screen shows springs in N/mm
            # and dampers in N/mm/s (= N·s/m / 1000).
            + _row("Spring (LFS N/mm)",
                   lambda w: _fmt(w.spring_const / 1000.0, 1, " N/mm"))
            + _row("Damp bump (LFS N/mm/s)",
                   lambda w: _fmt(w.damping_comp / 1000.0, 1, " N/mm/s"))
            + _row("Damp rebound (LFS N/mm/s)",
                   lambda w: _fmt(w.damping_rebound / 1000.0, 1, " N/mm/s"))
            # Official RAF spec lists antiroll as N/m (linear at the wheel),
            # not N·m/rad. Display matches the LFS in-game ARB N/mm scale.
            + _row("ARB (LFS N/mm)",
                   lambda w: _fmt(w.anti_roll / 1000.0, 2, " N/mm"))
            + _row("Camber (LFS deg)",
                   lambda w: _signed(_deg(w.camber_rad), 1, "°"))
            + _row("Caster (LFS deg, race-cars only)",
                   lambda w: _fmt(_deg(w.caster_rad), 1, "°"))
            # Toe is in radians in the bin; LFS in-game shows it in degrees.
            + _row("Toe-in (LFS deg)",
                   lambda w: _signed(_deg(w.toe_in_rad), 2, "°"))
            + _row("Inclination (view-only)",
                   lambda w: _signed(_deg(w.inclination_rad), 2, "°"))
            + _row("Scrub radius (view-only)",
                   lambda w: _fmt(w.scrub_radius_m * 1000.0, 1, " mm"))
            + _row("Max susp. travel",
                   lambda w: _fmt(w.max_susp_deflection_m * 1000.0,
                                  0, " mm"))
            + _row("Unsprung mass",
                   lambda w: _fmt(w.unsprung_mass_kg, 1, " kg"))
        )
        return (
            "<h3 style='margin:10px 0 4px 0;'>Suspension (per wheel)"
            " <span style='color:#9aa;font-weight:normal;'>"
            "(LFS in-game units)</span></h3>"
            "<table style='border-collapse:collapse;'>" + body + "</table>"
        )

    def _tuning_metrics_section(self, info: CarInfoBin) -> str:
        # Mirror what the Advanced Setup Guide actually optimises:
        # spring frequency (Hz), damping as % of critical, ARB-to-spring
        # roll-stiffness ratio per axle.
        wheels = self._wheels_by_label(info)
        unsprung_total = sum(w.unsprung_mass_kg for w in info.wheels)
        sprung = max(info.mass_kg - unsprung_total, 1.0)
        wd_f = info.weight_dist_front
        # Sprung mass per corner (front split equally L–R, same for rear).
        sprung_corner = {
            "FL": sprung * wd_f / 2.0,
            "FR": sprung * wd_f / 2.0,
            "RL": sprung * (1.0 - wd_f) / 2.0,
            "RR": sprung * (1.0 - wd_f) / 2.0,
        }

        head = (
            "<tr><th></th>"
            + "".join(
                f"<th style='padding:1px 10px;color:#9aa;'>{w}</th>"
                for w in _WHEEL_ORDER
            )
            + "</tr>"
        )

        def _row(label: str, fmt) -> str:
            cells = "".join(
                f"<td style='padding:1px 10px;'>"
                f"{fmt(corner, wheels[corner])}</td>"
                for corner in _WHEEL_ORDER
            )
            return (
                f"<tr><td style='padding:1px 14px 1px 0;color:#9aa;'>"
                f"{label}</td>{cells}</tr>"
            )

        def _freq(corner: str, w) -> str:
            m = sprung_corner[corner]
            if w.spring_const <= 0 or m <= 0:
                return "—"
            f = math.sqrt(w.spring_const / m) / (2.0 * math.pi)
            return _fmt(f, 2, " Hz")

        def _crit_pct(value: float, corner: str, w) -> str:
            m = sprung_corner[corner]
            if w.spring_const <= 0 or m <= 0 or value <= 0:
                return "—"
            c_crit = 2.0 * math.sqrt(w.spring_const * m)
            return _fmt(value / c_crit * 100.0, 0, " %")

        body = (
            head
            + _row("Sprung mass / corner",
                   lambda c, _w: _fmt(sprung_corner[c], 1, " kg"))
            + _row("Spring frequency", _freq)
            + _row("Bump  (% of critical)",
                   lambda c, w: _crit_pct(w.damping_comp, c, w))
            + _row("Rebound (% crit, target ≈80%)",
                   lambda c, w: _crit_pct(w.damping_rebound, c, w))
        )

        def _arb_ratio(front: bool) -> str:
            if front:
                k_s = (wheels["FL"].spring_const
                       + wheels["FR"].spring_const) / 2.0
                k_a = (wheels["FL"].anti_roll
                       + wheels["FR"].anti_roll) / 2.0
            else:
                k_s = (wheels["RL"].spring_const
                       + wheels["RR"].spring_const) / 2.0
                k_a = (wheels["RL"].anti_roll
                       + wheels["RR"].anti_roll) / 2.0
            if k_s <= 0:
                return "—"
            return _fmt(k_a / k_s, 2)

        axle_rows = [
            ("ARB / spring roll ratio — front",
             f"{_arb_ratio(True)} (target ≤ 1.0)"),
            ("ARB / spring roll ratio — rear",
             f"{_arb_ratio(False)} (target ≤ 1.0)"),
        ]
        axle_html = "".join(
            f"<tr><td style='padding:1px 14px 1px 0;color:#9aa;'>"
            f"{label}</td><td>{value}</td></tr>"
            for label, value in axle_rows
        )

        return (
            "<h3 style='margin:10px 0 4px 0;'>Tuning metrics"
            " <span style='color:#9aa;font-weight:normal;'>"
            "(derived per Advanced Setup Guide)</span></h3>"
            "<table style='border-collapse:collapse;'>" + body + "</table>"
            "<table style='border-collapse:collapse;margin-top:4px;'>"
            + axle_html + "</table>"
        )

    def _fuel_section(self, info: CarInfoBin) -> str:
        # LFS lets you pick fuel load as % of tank capacity in the
        # garage — expose both the raw capacity (L) and the litres-per-1%
        # conversion factor so the user can cross-check stints.
        cap = info.fuel_capacity_l
        per_pct = (cap / 100.0) if math.isfinite(cap) else float("nan")
        return self._section("Fuel tank", [
            ("Capacity", _fmt(cap, 1, " L")),
            ("Per 1% (LFS slider)", _fmt(per_pct, 3, " L")),
            ("Tank X (body)",
             _fmt(info.fuel_tank_x_m * 1000.0, 0, " mm")),
            ("Tank Y (body)",
             _fmt(info.fuel_tank_y_m * 1000.0, 0, " mm")),
            ("Tank Z (body)",
             _fmt(info.fuel_tank_z_m * 1000.0, 0, " mm")),
        ])

    def _balance_reference_section(self) -> str:
        # Static cheat-sheets condensed from the Basic Setup Guide
        # (Moby / Cyber Racing) — no per-car data, but the most useful
        # references to keep next to the live setup snapshot.
        muted = MUTED_COLOR
        th = ("padding:2px 8px;background:#222;color:#cfd;"
              "text-align:left;font-weight:normal;")
        td = "padding:2px 8px;border-top:1px solid #2a2a2a;"

        def _balance_table(title: str, rows: list[tuple[str, str]]) -> str:
            head = (
                f"<tr><th style='{th}'>Understeer fix</th>"
                f"<th style='{th}'>Oversteer fix</th></tr>"
            )
            body = "".join(
                f"<tr><td style='{td}'>{u}</td>"
                f"<td style='{td}'>{o}</td></tr>"
                for u, o in rows
            )
            return (
                f"<h4 style='margin:8px 0 2px 0;color:#cfd;'>{title}</h4>"
                f"<table style='border-collapse:collapse;"
                f"min-width:520px;'>{head}{body}</table>"
            )

        entry = _balance_table("Corner entry", [
            ("Soften front comp. damp", "Harden front comp. damp"),
            ("Soften rear rebound damp", "Harden rear rebound damp"),
            ("More caster", "Less caster"),
            ("Softer front ARB", "Harder front ARB"),
            ("Harder rear ARB", "Softer rear ARB"),
        ])
        mid = _balance_table("Mid corner", [
            ("Soften front ARB", "Harden front ARB"),
            ("Harder rear ARB", "Softer rear ARB"),
            ("More camber", "Less camber"),
            ("More front downforce", "Less front downforce"),
            ("Less rear downforce", "More rear downforce"),
        ])
        exit_ = _balance_table("Corner exit", [
            ("Harder front rebound damp", "Softer front rebound damp"),
            ("Harder rear comp. damp", "Softer rear comp. damp"),
            ("Less caster", "More caster"),
            ("Softer front ARB", "Harder front ARB"),
            ("Harder rear ARB", "Softer rear ARB"),
        ])

        # Camber heuristic from tyre temps (outer / middle / inner).
        camber_rows = [
            ("Inner hot, outer cool (e.g. 110 / 81 / 70)",
             "Too much negative camber"),
            ("Outer hot, inner cool (e.g. 90 / 80 / 55)",
             "Not enough negative camber"),
            ("Even across the tread", "Camber is on target"),
        ]
        camber_html = "".join(
            f"<tr><td style='{td}'>{a}</td>"
            f"<td style='{td}'>{b}</td></tr>"
            for a, b in camber_rows
        )
        camber_block = (
            f"<h4 style='margin:8px 0 2px 0;color:#cfd;'>"
            f"Camber from tyre temps "
            f"<span style='color:{muted};font-weight:normal;'>"
            f"(outer / middle / inner)</span></h4>"
            f"<table style='border-collapse:collapse;min-width:520px;'>"
            f"<tr><th style='{th}'>Reading</th>"
            f"<th style='{th}'>Action</th></tr>"
            f"{camber_html}</table>"
        )

        # Recommended ordered tuning checklist (Basic Setup Guide §2.1.2).
        order = [
            "Brakes",
            "Springs",
            "Anti-roll bars",
            "Aerodynamics / gearbox",
            "Dampers",
            "Steering, caster, parallel steer",
            "Tyre pressure",
        ]
        order_html = "".join(f"<li>{name}</li>" for name in order)
        order_block = (
            f"<h4 style='margin:8px 0 2px 0;color:#cfd;'>"
            f"Tuning order (one change at a time)</h4>"
            f"<ol style='margin:2px 0 0 18px;color:{TEXT_COLOR};'>"
            f"{order_html}</ol>"
        )

        # Aero rule of thumb — not in CAR_info.bin, info-only.
        aero_note = (
            f"<p style='color:{muted};margin:6px 0 0 0;font-size:11px;'>"
            f"Aero rule of thumb (Basic Setup Guide): for every notch of"
            f" rear-wing change, adjust the front wing by ~2 notches to"
            f" keep balance. Wing angles are not exported in"
            f" <code>CAR_info.bin</code>.</p>"
        )

        return (
            "<h3 style='margin:14px 0 4px 0;'>Setup balance reference"
            " <span style='color:#9aa;font-weight:normal;'>"
            "(LFS Basic Setup Guide cheat-sheet)</span></h3>"
            + entry + mid + exit_ + camber_block + order_block + aero_note
        )


class SetupTab(QWidget):
    """Container tab: hosts ``Baseline`` and ``Garage editor`` sub-tabs.

    The class deliberately holds *no* business logic — the heavy lifting
    lives in :class:`SetupBaselineTab` (parsed ``CAR_info.bin`` view)
    and the garage editor. This keeps the refactor surgical: existing
    call-sites that import :class:`SetupTab` keep working bit-for-bit.
    """

    def __init__(
        self,
        loader: LapLoader,
        signals: SignalBus,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # Lazy import: the editor pulls in the setup-overrides apply
        # path which is not needed for the baseline view alone.
        from .setup_editor_tab import SetupEditorTab

        tabs = QTabWidget(self)
        self._baseline = SetupBaselineTab(loader, signals)
        self._editor = SetupEditorTab(loader, signals)
        tabs.addTab(self._baseline, "Baseline")
        tabs.addTab(self._editor, "Garage editor")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tabs)


__all__ = ["SetupBaselineTab", "SetupTab"]
