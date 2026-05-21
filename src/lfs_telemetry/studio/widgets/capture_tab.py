"""Capture tab — start / stop the bundled ``lfs-telemetry capture``
subprocess from inside Studio.

Records every completed lap to a separate CSV (including the out-lap
from pit/grid exit to the first start/finish crossing as ``_lap00.csv``)
and stops only when the user clicks Stop.  No warm-up trickery, no lap
count cap: press Start, drive, press Stop — every full lap is on disk.

The lap-in-progress when Stop is pressed is intentionally discarded
(it has no end line crossing, so no canonical lap-time can be derived).
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...app.capture_runner import CaptureRunner
from ...telemetry.constants import (
    INSIM_DEFAULT_PORT,
    OUTGAUGE_DEFAULT_PORT,
    OUTSIM_DEFAULT_PORT,
)
from ..i18n import tr
from ..signals import SignalBus
from ..theme import MUTED_COLOR

_LAP_DONE_RE = re.compile(r"flying lap (\d+)")
_OUT_LAP_RE = re.compile(r"out-lap complete")
_ARMED_RE = re.compile(r"\[capture\] armed:")
_WAITING_RE = re.compile(r"Waiting for car to start moving")
_INSIM_WAIT_RE = re.compile(r"\[insim\] waiting for LFS")
_INSIM_READY_RE = re.compile(r"InSim ready\.|\[capture\] tracking PLID")
# Streaming per-lap mode: the capture CLI emits one "wrote N rows to
# /path/to/foo_lapNN.csv (streaming)" line per lap as it lands on disk.
# We watch for these so the dock can rescan the workspace mid-session.
_LAP_STREAMED_RE = re.compile(
    r"wrote\s+\d+\s+rows to .*?_lap\d{2}\.csv\s*\(streaming\)"
)


def _led_qss(color: str) -> str:
    """Stylesheet for a 14 px circular LED of ``color``."""
    return (
        "QLabel { "
        f"background-color: {color}; "
        "border-radius: 7px; "
        "min-width: 14px; max-width: 14px; "
        "min-height: 14px; max-height: 14px; "
        "border: 1px solid #1c1f24; "
        "}"
    )


class CaptureTab(QWidget):
    """Form + log + start/stop for a managed UDP capture subprocess."""

    def __init__(
        self,
        workspace: Path,
        signals: SignalBus,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._workspace = Path(workspace)
        self._signals = signals
        self._runner = CaptureRunner()
        # Public alias so the Live tab can locate live.json without
        # poking at private state.
        self.runner = self._runner
        self._was_running = False
        self._completed_laps = 0
        self._out_lap_done = False
        # How many "(streaming)" lap-write lines we've already
        # signalled, so each new one fires exactly one refresh.
        self._streamed_seen = 0

        # ----- Form ----------------------------------------------------
        self._stem = QLineEdit("stint", self)

        self._insim_host = QLineEdit("127.0.0.1", self)
        self._insim_port = QSpinBox(self)
        self._insim_port.setRange(1, 65535)
        self._insim_port.setValue(INSIM_DEFAULT_PORT)
        self._insim_port.setToolTip(
            tr(
                "TCP port LFS uses for InSim. Enable it inside LFS at "
                "runtime with  /insim 29999  in the console (or launch "
                "LFS.exe with /insim=29999). InSim has no cfg.txt entry.",
            )
        )

        self._outsim_port = QSpinBox(self)
        self._outsim_port.setRange(1, 65535)
        self._outsim_port.setValue(OUTSIM_DEFAULT_PORT)

        self._outgauge_port = QSpinBox(self)
        self._outgauge_port.setRange(1, 65535)
        self._outgauge_port.setValue(OUTGAUGE_DEFAULT_PORT)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.addRow(tr("Filename stem:"), self._stem)
        form.addRow(tr("InSim host:"), self._insim_host)
        form.addRow(tr("InSim port:"), self._insim_port)
        form.addRow(tr("OutSim port:"), self._outsim_port)
        form.addRow(tr("OutGauge port:"), self._outgauge_port)

        # Overlay-only mode: keep the live snapshot updating for the
        # Overlay tab but skip buffering samples and writing CSVs. The
        # session catalog is left untouched, no per-lap files are
        # produced — ideal for users who only want the in-game HUD.
        self._overlay_only = QCheckBox(
            tr("Overlay only (no CSV recording)"), self,
        )
        self._overlay_only.setToolTip(
            tr(
                "When enabled, the connection to LFS still drives the "
                "Overlay tab in real time, but no telemetry is buffered "
                "in memory and no per-lap or aggregate CSV is written "
                "to the workspace. Uncheck to record stints as usual.",
            )
        )
        form.addRow("", self._overlay_only)

        form_box = QGroupBox(tr("Capture"), self)
        form_box.setLayout(form)

        # ----- Buttons + status ---------------------------------------
        self._btn_start = QPushButton(tr("Start"), self)
        self._btn_stop = QPushButton(tr("Stop"), self)
        self._btn_stop.setEnabled(False)
        self._btn_start.clicked.connect(self._on_start)
        self._btn_stop.clicked.connect(self._on_stop)

        # LFS connection LED.
        self._led = QLabel(self)
        self._led.setStyleSheet(_led_qss("#5a5f66"))  # grey = idle
        self._led.setToolTip(tr("LFS InSim status: idle"))
        self._led_label = QLabel(tr("LFS"), self)
        self._led_label.setStyleSheet(f"color: {MUTED_COLOR};")

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        btn_row.addStretch(1)
        btn_row.addWidget(self._led)
        btn_row.addWidget(self._led_label)

        self._status = QLabel(tr("Idle."), self)
        self._status.setStyleSheet(f"color: {MUTED_COLOR};")

        self._lap_counter = QLabel(tr("Laps recorded: 0"), self)

        self._workspace_label = QLabel(
            tr("Workspace: {path}").format(path=self._workspace), self,
        )
        self._workspace_label.setStyleSheet(f"color: {MUTED_COLOR};")
        self._workspace_label.setWordWrap(True)

        # ----- Log ----------------------------------------------------
        self._log = QPlainTextEdit(self)
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.setStyleSheet(
            "QPlainTextEdit { background-color: #0f1418; "
            "color: #cfd6dc; font-family: Consolas, monospace; }"
        )

        info = QLabel(
            tr(
                "Records LFS UDP telemetry. You can press Start at any "
                "time (menu, pre-race countdown, pit, or already on "
                "track): the capture waits for LFS InSim to come up and "
                "only begins recording when the car actually starts "
                "moving. Every completed lap (out-lap included) is saved "
                "when you press Stop. Enable InSim in LFS first: "
                "<code>/insim 29999</code>.",
            ),
            self,
        )
        info.setStyleSheet(f"color: {MUTED_COLOR};")
        info.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(info)
        layout.addWidget(self._workspace_label)
        layout.addWidget(form_box)
        layout.addLayout(btn_row)
        layout.addWidget(self._status)
        layout.addWidget(self._lap_counter)
        layout.addWidget(QLabel(tr("Log:"), self))
        layout.addWidget(self._log, 1)

        # Workspace updates from the rest of the app.
        signals.workspace_changed.connect(self._on_workspace_changed)

        # Polling timer.
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    # ------------------------------------------------------------------

    def _on_workspace_changed(self, ws: Path) -> None:
        self._workspace = Path(ws)
        self._workspace_label.setText(
            tr("Workspace: {path}").format(path=self._workspace),
        )

    def _on_start(self) -> None:
        if self._runner.running:
            self._status.setText(tr("Already running."))
            return
        try:
            msg = self._runner.start(
                workspace=self._workspace,
                stem=self._stem.text() or "stint",
                seconds=0.0,
                laps=0,
                warmup_laps=0,
                per_lap=True,
                # Capture everything by default: out-lap as lap00 + every
                # flying lap. The user decides afterwards which laps are
                # useful instead of the recorder silently dropping data.
                include_out_lap=True,
                insim_host=self._insim_host.text() or "127.0.0.1",
                insim_port=int(self._insim_port.value()),
                outsim_port=int(self._outsim_port.value()),
                outgauge_port=int(self._outgauge_port.value()),
                write_csv=not self._overlay_only.isChecked(),
            )
        except Exception as exc:  # noqa: BLE001
            self._status.setText(
                tr("Start failed: {error}").format(error=exc),
            )
            return
        self._status.setText(msg)
        self._log.clear()
        self._completed_laps = 0
        self._out_lap_done = False
        self._streamed_seen = 0
        self._lap_counter.setText(tr("Laps recorded: 0"))
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._was_running = True

    def _on_stop(self) -> None:
        msg = self._runner.stop()
        self._status.setText(msg)

    def _update_lap_counter_from_log(self, lines: list[str]) -> None:
        """Parse the runner log to keep a live count of saved laps."""
        max_flying = 0
        out_done = False
        for ln in lines:
            m = _LAP_DONE_RE.search(ln)
            if m:
                try:
                    n = int(m.group(1))
                    if n > max_flying:
                        max_flying = n
                except ValueError:
                    pass
            elif _OUT_LAP_RE.search(ln):
                out_done = True
        self._completed_laps = max_flying
        self._out_lap_done = out_done
        out_tag = tr(" (+ out-lap)") if out_done else ""
        self._lap_counter.setText(
            tr("Laps recorded: {n}{out_tag}").format(
                n=max_flying, out_tag=out_tag,
            ),
        )

    def _poll(self) -> None:
        st = self._runner.status()
        log_lines = list(st["log"])
        text = "\n".join(log_lines)
        if text != self._log.toPlainText():
            self._log.setPlainText(text)
            sb = self._log.verticalScrollBar()
            sb.setValue(sb.maximum())
        self._update_lap_counter_from_log(log_lines)
        # Detect new "(streaming)" per-lap writes and refresh the
        # captures dock once per new file.
        streamed_now = sum(
            1 for ln in log_lines if _LAP_STREAMED_RE.search(ln)
        )
        if streamed_now > self._streamed_seen:
            self._streamed_seen = streamed_now
            self._signals.capture_lap_streamed.emit()
        running = bool(st["running"])
        out = st.get("output") or ""
        out_name = Path(out).name if out else ""
        # Determine sub-state from log tail.
        sub_state = ""
        armed = any(_ARMED_RE.search(ln) for ln in log_lines)
        insim_ready = any(_INSIM_READY_RE.search(ln) for ln in log_lines)
        tail = log_lines[-30:]
        insim_waiting = any(_INSIM_WAIT_RE.search(ln) for ln in tail)
        if running and not armed:
            if insim_waiting:
                sub_state = tr(" \u2014 waiting for LFS InSim")
            elif any(_WAITING_RE.search(ln) for ln in tail):
                sub_state = tr(" \u2014 waiting for car to move")
        # LED color logic:
        #   not running                -> grey
        #   running, InSim not yet up  -> red
        #   running, InSim up          -> green
        if not running:
            led_color = "#5a5f66"
            led_tip = tr("LFS InSim status: idle")
        elif insim_waiting and not insim_ready:
            led_color = "#d04848"
            led_tip = tr("LFS InSim status: waiting for connection")
        else:
            led_color = "#3fbf5a"
            led_tip = tr("LFS InSim status: connected")
        self._led.setStyleSheet(_led_qss(led_color))
        self._led.setToolTip(led_tip)
        self._led_label.setText(
            "LFS" if not running else ("LFS ●" if insim_ready else "LFS")
        )
        if running:
            self._status.setText(
                tr("\u25cf Recording \u2192 {file}{state}").format(
                    file=out_name, state=sub_state,
                ),
            )
            self._btn_start.setEnabled(False)
            self._btn_stop.setEnabled(True)
        else:
            code = st.get("exit_code")
            if code is None and not out:
                self._status.setText(tr("Idle."))
            else:
                self._status.setText(
                    tr(
                        "\u25a0 Finished (code={code}) \u2192 {file}",
                    ).format(code=code, file=out_name),
                )
            self._btn_start.setEnabled(True)
            self._btn_stop.setEnabled(False)
        if self._was_running and not running:
            self._was_running = False


__all__ = ["CaptureTab"]
