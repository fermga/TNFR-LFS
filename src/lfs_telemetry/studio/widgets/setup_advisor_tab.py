"""TNFR Setup Advisor sub-tab.

Lives inside :class:`SetupTab` (see ``setup_tab.py``). Subscribes to the
shared :class:`SignalBus` ``laps_selected`` signal, waits until every
requested lap is available from the :class:`LapLoader`, resolves the
matching ``<car>_CAR_info.bin`` baseline, and runs
:class:`lfs_telemetry.tnfr_racing.advisor.SetupAdvisor` to produce a
``ProposedSetup`` (or a structured refusal).

Render layout (docs/TNFR_SETUP_ADVISOR.md §9.2):

* Top — selector status (laps count, car, track, refusal hints).
* Middle — diagnostics in physical language only (no TNFR jargon).
* Bottom — proposed actions table + ΔC pill + grammar pill + export.

All textual output goes through advisor :class:`Diagnostic`s (already
jargon-filtered by the Phase 5 test suite) and through
:func:`_humanize_action` (a fixed translation table). No raw operator
class names, ``EPI``/``νf``/``ΔNFR``/``Φ_s`` or U1–U6 references ever
reach the UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
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
from ...tnfr_racing.advisor import (
    AdvisorResult,
    ProposedSetup,
    SetupAdvisor,
)
from ...tnfr_racing.serialize import (
    format_refusal as _format_refusal_text,
    humanize_action as _humanize_action,
    result_to_json,
    result_to_markdown,
    synthesis_to_html as _synthesis_to_html,
)
from ..models import LapLoader
from ..signals import SignalBus
from ..theme import MUTED_COLOR, PANEL_COLOR, TEXT_COLOR

# ---------------------------------------------------------------------------
# Translation tables (TNFR-free physical language only)
# ---------------------------------------------------------------------------

# Widget
# ---------------------------------------------------------------------------


class SetupAdvisorTab(QWidget):
    """TNFR Setup Advisor view (Phase 6 UI)."""

    DEFAULT_SEED = 20260516

    def __init__(
        self,
        loader: LapLoader,
        signals: SignalBus,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._loader = loader
        self._signals = signals
        self._requested_paths: List[Path] = []
        self._loaded_laps: Dict[Path, LapTelemetry] = {}
        self._last_result: AdvisorResult | None = None
        # Per-car overrides emitted by SetupEditorTab. If a key is
        # present here, the advisor uses the patched CarInfoBin as its
        # baseline instead of re-reading the on-disk bin. Cleared (set
        # to None) when the user presses "Reset to imported".
        self._overrides: Dict[str, CarInfoBin] = {}

        # ---- top bar: status + seed + recalc -----------------------
        self._status = QLabel(self)
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color:{MUTED_COLOR}; padding:6px;")

        self._seed = QSpinBox(self)
        self._seed.setRange(0, 2_147_483_647)
        self._seed.setValue(self.DEFAULT_SEED)
        self._seed.setPrefix("seed: ")
        self._seed.setSingleStep(1)
        self._recalc = QPushButton("Recalculate", self)
        self._recalc.clicked.connect(self._recompute)
        self._export_json = QPushButton("Export JSON…", self)
        self._export_json.clicked.connect(self._on_export_json)
        self._export_md = QPushButton("Export Markdown…", self)
        self._export_md.clicked.connect(self._on_export_markdown)
        self._import_btn = QPushButton("Import CAR_info.bin…", self)
        self._import_btn.setToolTip(
            "Pick a <car>_CAR_info.bin export from LFS Programmer Mode."
            f" It will be copied to {user_car_info_bin_dir()}."
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
        for btn in (self._export_json, self._export_md):
            btn.setEnabled(False)

        top_row = QHBoxLayout()
        top_row.addWidget(self._seed)
        top_row.addWidget(self._recalc)
        top_row.addStretch(1)
        top_row.addWidget(self._gen_lfs_btn)
        top_row.addWidget(self._import_lfs_btn)
        top_row.addWidget(self._import_btn)
        top_row.addWidget(self._export_json)
        top_row.addWidget(self._export_md)

        # ---- main view ---------------------------------------------
        self._view = QTextEdit(self)
        self._view.setReadOnly(True)
        self._view.setStyleSheet(
            f"QTextEdit {{ background:{PANEL_COLOR};"
            f" color:{TEXT_COLOR}; border:0; padding:8px; }}"
        )
        self._view.setHtml(self._empty_html())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._status)
        layout.addLayout(top_row)
        layout.addWidget(self._view, 1)

        signals.laps_selected.connect(self._on_laps_selected)
        loader.lap_loaded.connect(self._on_lap_loaded)
        signals.setup_overrides_changed.connect(
            self._on_setup_overrides_changed
        )

    # ----- garage editor bridge ---------------------------------------

    def _on_setup_overrides_changed(
        self, car_key: str, patched: object,
    ) -> None:
        """Receive a patched ``CarInfoBin`` from :class:`SetupEditorTab`.

        ``patched`` is either a :class:`CarInfoBin` (apply this as the
        car's baseline) or ``None`` (the editor was reset — fall back
        to the on-disk bin). We never auto-recompute; the user clicks
        Recalculate when ready so they can batch edits.
        """
        key = (car_key or "").upper().strip()
        if not key:
            return
        if patched is None:
            self._overrides.pop(key, None)
        elif isinstance(patched, CarInfoBin):
            self._overrides[key] = patched

    # ----- slots --------------------------------------------------------

    def _on_laps_selected(self, paths: List[Path]) -> None:
        self._requested_paths = [Path(p) for p in paths]
        wanted = set(self._requested_paths)
        self._loaded_laps = {
            p: lap for p, lap in self._loaded_laps.items() if p in wanted
        }
        self._update_status()
        if not self._requested_paths:
            self._view.setHtml(self._empty_html())
            return
        for path in self._requested_paths:
            if path not in self._loaded_laps:
                self._loader.request(path)
        if all(p in self._loaded_laps for p in self._requested_paths):
            self._recompute()

    def _on_lap_loaded(self, path: Path, lap: LapTelemetry) -> None:
        path = Path(path)
        if path not in self._requested_paths:
            return
        self._loaded_laps[path] = lap
        self._update_status()
        if all(p in self._loaded_laps for p in self._requested_paths):
            self._recompute()

    # ----- pipeline ----------------------------------------------------

    def _recompute(self) -> None:
        laps = [
            self._loaded_laps[p]
            for p in self._requested_paths
            if p in self._loaded_laps
        ]
        if len(laps) < 5:
            self._set_refusal(
                f"Need ≥ 5 consecutive valid laps from the same stint."
                f" Currently selected: {len(laps)}."
            )
            return

        first = laps[0]
        car_key = ""
        track_code = ""
        if first.summary:
            car_key = str(first.summary.get("car") or "").upper()
            track_code = str(first.summary.get("track") or "").upper()
        if not car_key or not track_code:
            self._set_refusal(
                "First selected lap has no car/track id in its summary."
            )
            return

        baseline: CarInfoBin | None = self._overrides.get(car_key)
        if baseline is None:
            baseline = load_car_info_bin_for(car_key)
        if baseline is None:
            target_dir = user_car_info_bin_dir()
            self._set_refusal(
                f"No <code>{car_key}_CAR_info.bin</code> export found"
                f" on the search path.<br><br>"
                f"<b>Easiest path:</b> click"
                f" <b>Import from LFS folder…</b> above and point us"
                f" at your LFS install. We'll copy every"
                f" <code>*_CAR_info.bin</code> from"
                f" <code>LFS\\data\\</code> in one shot.<br><br>"
                f"<b>Other options:</b><ol>"
                f"<li>Click <b>Import CAR_info.bin…</b> and pick a"
                f" single file exported by LFS.</li>"
                f"<li>Copy <code>{car_key}_CAR_info.bin</code> manually"
                f" into <code>{target_dir}</code> (or set"
                f" <code>$LFS_TELEMETRY_CAR_INFO_DIR</code> to a folder"
                f" containing it).</li></ol>"
                f"<b>To generate the file in LFS:</b> launch"
                f" <code>LFS.exe /prog</code> (Programmer Mode), drive"
                f" the {car_key}, choose <i>Save CAR_info.bin</i>; the"
                f" export lands in <code>LFS\\data\\</code>."
            )
            return

        advisor = SetupAdvisor(seed=int(self._seed.value()))
        try:
            result = advisor.advise(laps, baseline, first.car, track_code)
        except Exception as exc:  # defensive: never crash the UI thread
            self._set_refusal(f"Advisor failed: {type(exc).__name__}: {exc}")
            self._last_result = None
            self._export_json.setEnabled(False)
            self._export_md.setEnabled(False)
            return

        self._last_result = result
        has_proposal = result.proposed is not None
        self._export_json.setEnabled(has_proposal)
        self._export_md.setEnabled(has_proposal)
        self._render_result(result, car_key, track_code, len(laps))

    # ----- rendering ---------------------------------------------------

    def _empty_html(self) -> str:
        return (
            f"<p style='color:{MUTED_COLOR};'>"
            f"Select ≥ 5 consecutive laps from the same stint on the"
            f" left, then press <b>Recalculate</b> to obtain a setup"
            f" recommendation.</p>"
        )

    def _set_refusal(self, message_html: str) -> None:
        self._view.setHtml(
            f"<h3 style='margin:0 0 6px 0;'>No safe recommendation</h3>"
            f"<p style='color:{TEXT_COLOR};'>{message_html}</p>"
        )

    def _update_status(self) -> None:
        n_sel = len(self._requested_paths)
        n_loaded = len(self._loaded_laps)
        if n_sel == 0:
            self._status.setText("No laps selected.")
            return
        first = self._requested_paths[0].name if n_sel else ""
        last = self._requested_paths[-1].name if n_sel else ""
        self._status.setText(
            f"Stint selection: {n_loaded}/{n_sel} laps loaded "
            f"({first} … {last})."
        )

    def _render_result(
        self,
        result: AdvisorResult,
        car_key: str,
        track_code: str,
        n_laps: int,
    ) -> None:
        parts: list[str] = []
        parts.append(
            f"<h2 style='margin:0 0 4px 0;'>{car_key} @ {track_code}"
            f" — {n_laps}-lap stint</h2>"
        )

        if result.refusal_reason is not None:
            parts.append(
                f"<h3 style='margin:8px 0 4px 0;'>No safe recommendation</h3>"
                f"<p>{self._format_refusal(result.refusal_reason)}</p>"
            )
            parts.append(self._diagnostics_block(result))
            self._view.setHtml("".join(parts))
            return

        assert result.proposed is not None
        proposed: ProposedSetup = result.proposed

        parts.append(self._diagnostics_block(result))
        parts.append(self._actions_table(proposed))
        parts.append(_synthesis_to_html(
            proposed.synthesis, muted_color=MUTED_COLOR,
        ))
        parts.append(self._footer_block(proposed))
        self._view.setHtml("".join(parts))

    def _format_refusal(self, reason: str) -> str:
        # Delegate to the shared, Qt-free translation table so UI and
        # CLI render the same physical sentence.
        return _format_refusal_text(reason)

    def _diagnostics_block(self, result: AdvisorResult) -> str:
        if not result.diagnostics:
            return ""
        rows: list[str] = []
        for d in result.diagnostics:
            val = ""
            if d.value is not None:
                val = f" <span style='color:{MUTED_COLOR};'>= {d.value:.3f}"
                if d.units:
                    val += f" {d.units}"
                val += "</span>"
            rows.append(f"<li>{d.message}{val}</li>")
        return (
            "<h3 style='margin:10px 0 4px 0;'>Diagnostics</h3>"
            f"<ul style='margin:0 0 6px 18px;color:{TEXT_COLOR};'>"
            + "".join(rows)
            + "</ul>"
        )

    def _actions_table(self, p: ProposedSetup) -> str:
        if not p.actions:
            return ""
        th = ("padding:2px 10px;background:#222;color:#cfd;"
              "text-align:left;font-weight:normal;")
        td = "padding:2px 10px;border-top:1px solid #2a2a2a;"
        head = (
            f"<tr><th style='{th}'>#</th>"
            f"<th style='{th}'>Subsystem</th>"
            f"<th style='{th}'>Change</th>"
            f"<th style='{th}'>Why</th></tr>"
        )
        rows: list[str] = []
        for i, act in enumerate(p.actions, start=1):
            sub, change, rationale = _humanize_action(act)
            rows.append(
                f"<tr><td style='{td}'>{i}</td>"
                f"<td style='{td}'>{sub}</td>"
                f"<td style='{td}'>{change}</td>"
                f"<td style='{td}'>{rationale}</td></tr>"
            )
        return (
            "<h3 style='margin:10px 0 4px 0;'>Proposed changes</h3>"
            "<table style='border-collapse:collapse;min-width:560px;'>"
            f"{head}{''.join(rows)}</table>"
        )

    def _footer_block(self, p: ProposedSetup) -> str:
        delta = p.expected_coherence_delta
        grammar = "✓ passes stability constraints" if p.grammar_passed \
            else "✗ fails stability constraints"
        return (
            "<p style='margin:10px 0 0 0;'>"
            f"Projected coherence improvement: <b>+{delta:.3f}</b>"
            f" ({p.coherence_before:.3f} → {p.coherence_after:.3f})"
            f" &nbsp;|&nbsp; lap-time impact: <b>n/a (v1)</b>"
            f" &nbsp;|&nbsp; {grammar}"
            "</p>"
            f"<p style='color:{MUTED_COLOR};margin:2px 0 0 0;font-size:11px;'>"
            f"seed={p.seed} · baseline={p.baseline_hash} ·"
            f" stint={p.stint_signature}</p>"
        )

    # ----- import ------------------------------------------------------

    def _current_car_key(self) -> str:
        for path in self._requested_paths:
            lap = self._loaded_laps.get(path)
            if lap is None or not lap.summary:
                continue
            key = str(lap.summary.get("car") or "").strip().upper()
            if key:
                return key
        return ""

    def _on_import_clicked(self) -> None:
        car_key = self._current_car_key()
        prompt = (
            f"Select {car_key}_CAR_info.bin"
            if car_key
            else "Select <car>_CAR_info.bin"
        )
        src, _ = QFileDialog.getOpenFileName(
            self,
            prompt,
            "",
            "LFS CAR_info.bin (*_CAR_info.bin *.bin);;All files (*)",
        )
        if not src:
            return
        try:
            dst, info = import_car_info_bin(
                Path(src),
                target_key=car_key or None,
            )
        except Exception as exc:  # surface failures to the user
            QMessageBox.critical(
                self,
                "Import failed",
                f"Could not import CAR_info.bin:\n{type(exc).__name__}: {exc}",
            )
            return
        QMessageBox.information(
            self,
            "Imported",
            f"Imported {info.short_name or '<car>'}\nto {dst}\n\n"
            "Press Recalculate to use the new baseline.",
        )
        self._recompute()

    def _on_import_lfs_clicked(self) -> None:
        from ._lfs_bin_import import import_bins_from_lfs_folder
        if import_bins_from_lfs_folder(self) > 0:
            self._recompute()

    def _on_gen_lfs_clicked(self) -> None:
        from ._lfs_bin_import import launch_lfs_programmer_mode
        launch_lfs_programmer_mode(self)

    # ----- export ------------------------------------------------------

    def _on_export_json(self) -> None:
        if not self._last_result or not self._last_result.proposed:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export advisor result (JSON)",
            "advisor_result.json", "JSON (*.json)",
        )
        if not path:
            return
        Path(path).write_text(self._serialize_json(), encoding="utf-8")

    def _on_export_markdown(self) -> None:
        if not self._last_result or not self._last_result.proposed:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export advisor result (Markdown)",
            "advisor_result.md", "Markdown (*.md)",
        )
        if not path:
            return
        Path(path).write_text(self._serialize_markdown(), encoding="utf-8")

    def _serialize_json(self) -> str:
        assert self._last_result and self._last_result.proposed
        return result_to_json(self._last_result)

    def _serialize_markdown(self) -> str:
        assert self._last_result and self._last_result.proposed
        return result_to_markdown(self._last_result)


__all__ = ["SetupAdvisorTab"]
