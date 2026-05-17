"""Shared helper: bulk-import ``*_CAR_info.bin`` from the user's LFS install.

Both :mod:`setup_tab` and :mod:`setup_advisor_tab` expose an "Import from
LFS folder…" button that calls into this helper. The LFS install folder
is resolved through :mod:`lfs_paths` (single source of truth across the
app), so this module no longer touches ``QSettings`` directly.

It also exposes :func:`launch_lfs_programmer_mode`, a one-click helper
that spawns ``LFS.exe /prog`` so the user can save the bin from inside
the simulator without going to the command line.
"""
from __future__ import annotations

import subprocess

from PySide6.QtWidgets import QMessageBox, QWidget

from ...lfs_config import find_lfs_car_info_bins
from ...lfs_paths import lfs_exe, require_lfs_dir
from ...telemetry.observables import (
    import_car_info_bins_from_lfs,
    user_car_info_bin_dir,
)


def import_bins_from_lfs_folder(parent: QWidget) -> int:
    """Run the bulk import workflow. Returns the number of files imported."""
    lfs_dir = require_lfs_dir(parent)
    if lfs_dir is None:
        return 0

    bins = find_lfs_car_info_bins(lfs_dir)
    if not bins:
        QMessageBox.information(
            parent, "Import from LFS folder",
            f"No <code>*_CAR_info.bin</code> files were found in "
            f"<code>{lfs_dir / 'data'}</code>.<br><br>"
            "Click <b>Generate CAR_info.bin (LFS)…</b> to launch LFS"
            " in Programmer Mode. Drive each car you want to advise"
            " on, choose <b>Save CAR_info.bin</b> from the Programmer"
            " menu, close LFS, then come back here and click"
            " <b>Import from LFS folder…</b> again.",
        )
        return 0

    imported, failed = import_car_info_bins_from_lfs(lfs_dir)

    parts: list[str] = []
    if imported:
        keys = ", ".join(k for k, _ in imported)
        parts.append(
            f"<b>{len(imported)}</b> car(s) imported into "
            f"<code>{user_car_info_bin_dir()}</code>:<br>{keys}"
        )
    if failed:
        details = "<br>".join(
            f"\u2022 {p.name}: {msg}" for p, msg in failed
        )
        parts.append(
            f"<br><b>{len(failed)}</b> file(s) could not be imported:"
            f"<br>{details}"
        )
    if not parts:
        parts.append("Nothing to import.")
    QMessageBox.information(
        parent, "Import from LFS folder", "<br>".join(parts),
    )
    return len(imported)


def launch_lfs_programmer_mode(parent: QWidget) -> bool:
    """Spawn ``LFS.exe /prog`` so the user can save CAR_info.bin files.

    Returns ``True`` if the process was started. The function does not
    block: LFS runs detached. On success, an instructional message tells
    the user how to save the bin from inside the simulator and where to
    click afterwards to bulk-import the results.
    """
    lfs_dir = require_lfs_dir(parent)
    if lfs_dir is None:
        return False
    exe = lfs_exe(lfs_dir)
    if not exe.is_file():
        QMessageBox.warning(
            parent, "Generate CAR_info.bin",
            f"<code>{exe}</code> not found.<br>"
            "The configured LFS folder no longer contains LFS.exe.",
        )
        return False
    try:
        subprocess.Popen(
            [str(exe), "/prog"],
            cwd=str(lfs_dir),
            close_fds=True,
        )
    except OSError as exc:
        QMessageBox.critical(
            parent, "Generate CAR_info.bin",
            f"Could not launch LFS:<br><code>{type(exc).__name__}:"
            f" {exc}</code>",
        )
        return False
    QMessageBox.information(
        parent, "Generate CAR_info.bin",
        "LFS is starting in <b>Programmer Mode</b>.<br><br>"
        "For each car you want to advise on:<br>"
        "&nbsp;1. Pick the car in Single Player.<br>"
        "&nbsp;2. From the Programmer menu choose"
        " <b>Save CAR_info.bin</b>.<br>"
        f"&nbsp;3. LFS writes <code>&lt;CAR&gt;_CAR_info.bin</code>"
        f" into <code>{lfs_dir / 'data'}</code>.<br><br>"
        "When done, close LFS and click"
        " <b>Import from LFS folder…</b> to copy them all at once.",
    )
    return True


__all__ = [
    "import_bins_from_lfs_folder",
    "launch_lfs_programmer_mode",
]
