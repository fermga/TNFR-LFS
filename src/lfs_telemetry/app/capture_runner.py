"""Run ``lfs-telemetry capture`` as a managed background subprocess.

Used by the Capture tab in the viewer so the user can configure InSim
host/ports, warm-up laps, stint duration (laps OR seconds) and start /
stop a capture without leaving the browser. Output goes to the same
workspace folder the viewer is browsing, so the catalog picks it up
automatically on the next "Reload".

The subprocess runs the same packaged executable (``sys.executable``)
with the hidden ``--cli`` switch, which delegates to
``lfs_telemetry.cli.main``. That keeps the canonical capture pipeline as
the single source of truth (warm-up handling, per-lap slicing, IS_LAP
trimming…) instead of duplicating it inside the viewer.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

from ..telemetry.constants import (
    INSIM_DEFAULT_PORT,
    OUTGAUGE_DEFAULT_PORT,
    OUTSIM_DEFAULT_PORT,
)
from ..telemetry.session_naming import safe_token, timestamp_tag


class CaptureRunner:
    """Spawn / supervise one ``lfs-telemetry capture`` subprocess at a time."""

    def __init__(self, log_lines: int = 400) -> None:
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()
        self._log: deque[str] = deque(maxlen=log_lines)
        self._exit_code: int | None = None
        self._cmd_str: str = ""
        self._out_path: Path | None = None
        self._stop_file: Path | None = None
        self._live_file: Path | None = None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict:
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
            return {
                "running": running,
                "exit_code": self._exit_code,
                "log": list(self._log),
                "cmd": self._cmd_str,
                "output": str(self._out_path) if self._out_path else "",
                "live_file": str(getattr(self, "_live_file", "") or ""),
            }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        *,
        workspace: Path,
        stem: str = "stint",
        seconds: float = 0.0,
        laps: int = 0,
        warmup_laps: int = 0,
        per_lap: bool = True,
        include_out_lap: bool = True,
        insim_host: str = "127.0.0.1",
        insim_port: int = INSIM_DEFAULT_PORT,
        outsim_port: int = OUTSIM_DEFAULT_PORT,
        outgauge_port: int = OUTGAUGE_DEFAULT_PORT,
        write_csv: bool = True,
    ) -> str:
        if self.running:
            return "Already running"
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        ts = timestamp_tag()
        safe_stem = safe_token(stem, extra_allowed="-_") if stem else "stint"
        if safe_stem == "unknown":
            safe_stem = "stint"
        if write_csv:
            # Each capture lands in its own subfolder under the workspace, so
            # the aggregate stint CSV and the per-lap CSVs stay grouped and
            # don't pollute the workspace root. The catalog scans recursively
            # so the new files show up on the next reload.
            session_dir = workspace / f"{safe_stem}_{ts}"
            session_dir.mkdir(parents=True, exist_ok=True)
            out_path = session_dir / f"{safe_stem}_{ts}.csv"
        else:
            # Overlay-only mode: no per-session folder, no CSV. Reuse a
            # single hidden directory under the workspace for live.json
            # and the stop sentinel so we don't litter the catalog with
            # empty session folders.
            session_dir = workspace / "_overlay"
            session_dir.mkdir(parents=True, exist_ok=True)
            # The CLI still requires a positional output path; nothing
            # will be written to it because --no-csv is passed below.
            out_path = session_dir / f"{safe_stem}.csv"

        # When running under a regular Python interpreter we invoke the
        # CLI module directly (``python -m lfs_telemetry.cli``). Under
        # a PyInstaller-frozen exe ``-m`` is not available, so we fall
        # back to the launcher's hidden ``--cli`` switch.
        frozen = bool(getattr(sys, "frozen", False))
        if frozen:
            argv: list[str] = [
                "--cli", "capture", str(out_path),
            ]
        else:
            argv = [
                "-m", "lfs_telemetry.cli",
                "capture", str(out_path),
            ]
        argv += [
            "--insim-host", str(insim_host or "127.0.0.1"),
            "--insim-port", str(int(insim_port)),
            "--outsim-port", str(int(outsim_port)),
            "--outgauge-port", str(int(outgauge_port)),
            "--warmup-laps", str(max(0, int(warmup_laps))),
        ]
        if seconds and seconds > 0:
            argv += ["--seconds", str(float(seconds))]
        if laps and laps > 0:
            argv += ["--laps", str(int(laps))]
        if per_lap:
            argv += ["--per-lap"]
        if include_out_lap:
            argv += ["--include-out-lap"]
        if not write_csv:
            argv += ["--no-csv"]

        # Sentinel file used by stop(): when the GUI parent has no
        # console (PyInstaller windowed exe), GenerateConsoleCtrlEvent
        # fails with WinError 6, so we fall back to a file the child
        # polls.
        stop_file = session_dir / ".stop"
        try:
            stop_file.unlink()
        except OSError:
            pass
        argv += ["--stop-file", str(stop_file)]

        # Live snapshot file (consumed by the Studio Live tab to drive
        # the in-game-style overlay). The CLI refreshes it at ~10 Hz.
        live_file = session_dir / "live.json"
        try:
            live_file.unlink()
        except OSError:
            pass
        argv += ["--live-file", str(live_file)]

        cmd = [sys.executable] + argv

        creationflags = 0
        if os.name == "nt":
            # New process group lets us send CTRL_BREAK_EVENT just to the
            # child (translated to KeyboardInterrupt inside the capture
            # coroutine, which then writes the CSVs cleanly).
            creationflags = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )

        with self._lock:
            self._log.clear()
            self._exit_code = None
            self._cmd_str = " ".join(cmd)
            self._out_path = out_path
            self._stop_file = stop_file
            self._live_file = live_file
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            self._reader = threading.Thread(
                target=self._pump_output,
                name="capture-reader",
                daemon=True,
            )
            self._reader.start()

        return f"Started: {out_path.name}"

    def _pump_output(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                with self._lock:
                    self._log.append(line.rstrip("\r\n"))
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._log.append(f"[runner] reader error: {exc}")
        finally:
            code = proc.wait()
            with self._lock:
                self._exit_code = code
                self._log.append(f"[runner] process exited (code={code})")

    def stop(self) -> str:
        with self._lock:
            proc = self._proc
            stop_file = getattr(self, "_stop_file", None)
        if proc is None or proc.poll() is not None:
            return "Not running"
        # Primary mechanism: write the sentinel file. The CLI polls for
        # it and shuts the capture loop down cleanly, then writes the
        # CSVs. This works regardless of whether the parent has a
        # console (the windowed Studio exe does not).
        if stop_file is not None:
            try:
                stop_file.parent.mkdir(parents=True, exist_ok=True)
                stop_file.write_text("stop\n", encoding="utf-8")
            except OSError as exc:
                return f"Stop failed: cannot write {stop_file}: {exc}"
            return "Stopping\u2026"
        # Fallback for older invocations (no stop file): try a CTRL
        # event. May raise WinError 6 if parent has no console.
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
        except Exception as exc:  # noqa: BLE001
            return f"Stop failed: {exc}"
        return "Stopping\u2026"