"""Modal dialog that helps the user configure LFS for lfs-telemetry.

It can either patch ``cfg.txt`` automatically once the user points us to
their LFS install folder, or simply display the manual snippet to paste
into the file by hand.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...lfs_config import (
    cfg_path_for,
    find_default_lfs_dir,
    is_valid_lfs_dir,
    manual_instructions,
    patch_cfg,
)

_ORG = "LFS-Race-Engineer"
_APP = "LFS Telemetry Studio"
_KEY_LFS_DIR = "lfs/install_dir"


class LfsConfigDialog(QDialog):
    """Configure LFS dialog (LFS path picker + auto-patch + manual block)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configure LFS for telemetry")
        self.setModal(True)
        self.resize(640, 520)

        root = QVBoxLayout(self)

        intro = QLabel(
            "lfs-telemetry needs a few <b>OutSim</b>, <b>OutGauge</b> and "
            "<b>InSim</b> entries in your LFS <code>cfg.txt</code>. Point "
            "the app at your LFS install folder and click "
            "<i>Patch cfg.txt automatically</i>, or copy the block below "
            "into the file by hand.<br><br>"
            "<b>LFS must be closed</b> while the file is patched, "
            "otherwise it will overwrite your changes on exit.",
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(intro)

        # ---- Path row -------------------------------------------------
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("LFS folder:"))
        self._path_edit = QLineEdit(self)
        self._path_edit.setPlaceholderText(r"e.g. C:\LFS")
        path_row.addWidget(self._path_edit, 1)
        browse_btn = QPushButton("Browse…", self)
        browse_btn.clicked.connect(self._on_browse)
        path_row.addWidget(browse_btn)
        root.addLayout(path_row)

        self._status_label = QLabel("", self)
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        # ---- Action row ----------------------------------------------
        actions_row = QHBoxLayout()
        self._patch_btn = QPushButton("Patch cfg.txt automatically", self)
        self._patch_btn.clicked.connect(self._on_patch)
        actions_row.addWidget(self._patch_btn)
        copy_btn = QPushButton("Copy snippet", self)
        copy_btn.clicked.connect(self._on_copy_snippet)
        actions_row.addWidget(copy_btn)
        actions_row.addStretch(1)
        root.addLayout(actions_row)

        # ---- Manual block --------------------------------------------
        root.addWidget(QLabel("Manual snippet (paste at the end of cfg.txt):"))
        self._snippet = QPlainTextEdit(self)
        self._snippet.setReadOnly(True)
        self._snippet.setPlainText(manual_instructions())
        font = self._snippet.font()
        font.setFamily("Consolas")
        self._snippet.setFont(font)
        root.addWidget(self._snippet, 1)

        # ---- Close button --------------------------------------------
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        # Restore last-known LFS folder, otherwise probe defaults.
        self._settings = QSettings(_ORG, _APP)
        saved = self._settings.value(_KEY_LFS_DIR, "", type=str)
        if saved:
            self._path_edit.setText(saved)
        else:
            guess = find_default_lfs_dir()
            if guess is not None:
                self._path_edit.setText(str(guess))

        self._path_edit.textChanged.connect(self._refresh_status)
        self._refresh_status()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        start = self._path_edit.text().strip() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(
            self, "Select LFS install folder", start,
        )
        if chosen:
            self._path_edit.setText(chosen)

    def _on_copy_snippet(self) -> None:
        QGuiApplication.clipboard().setText(self._snippet.toPlainText())
        self._status_label.setText(
            "Snippet copied to clipboard. Paste it at the end of cfg.txt.",
        )

    def _on_patch(self) -> None:
        text = self._path_edit.text().strip()
        if not text:
            QMessageBox.warning(
                self, "Configure LFS",
                "Please choose your LFS install folder first.",
            )
            return
        lfs_dir = Path(text)
        if not is_valid_lfs_dir(lfs_dir):
            QMessageBox.warning(
                self, "Configure LFS",
                f"{lfs_dir}\n\nDoes not look like an LFS install folder "
                "(no LFS.exe or cfg.txt found).",
            )
            return

        try:
            result = patch_cfg(lfs_dir)
        except FileNotFoundError as exc:
            QMessageBox.warning(self, "Configure LFS", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(
                self, "Configure LFS",
                f"Could not write cfg.txt:\n\n{exc}\n\n"
                "Make sure LFS is closed and that the file is not "
                "read-only.",
            )
            return

        # Persist the chosen folder for next time.
        self._settings.setValue(_KEY_LFS_DIR, str(lfs_dir))

        QMessageBox.information(
            self, "Configure LFS",
            result.summary_text()
            + "\n\nDone. Launch LFS and enter a session.",
        )
        self._refresh_status()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_status(self) -> None:
        text = self._path_edit.text().strip()
        if not text:
            self._status_label.setText(
                "<i>No folder selected.</i>",
            )
            self._patch_btn.setEnabled(False)
            return
        path = Path(text)
        if not is_valid_lfs_dir(path):
            self._status_label.setText(
                f"<span style='color:#c0392b'>"
                f"Folder does not look like an LFS install: {path}"
                "</span>",
            )
            self._patch_btn.setEnabled(False)
            return
        cfg = cfg_path_for(path)
        if not cfg.exists():
            self._status_label.setText(
                f"<span style='color:#c0392b'>"
                f"{cfg} does not exist yet — launch LFS once to generate "
                "cfg.txt, then quit and try again.</span>",
            )
            self._patch_btn.setEnabled(False)
            return
        self._status_label.setText(
            f"<span style='color:#27ae60'>Ready: {cfg}</span>",
        )
        self._patch_btn.setEnabled(True)


__all__ = ["LfsConfigDialog"]
