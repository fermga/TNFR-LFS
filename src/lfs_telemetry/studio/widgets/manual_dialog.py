"""User manual dialog: renders ``docs/MANUAL.<lang>.md`` inside the app.

The manual ships bundled with the frozen application (see the
PyInstaller spec ``_bundle_dir("docs", "*.md")``) and is also kept at
``docs/`` in the source tree for development runs.

The file picked depends on the currently active UI language
(``i18n.current_language``):

* ``LANG_SPANISH`` → ``docs/MANUAL.es.md``
* anything else → ``docs/MANUAL.en.md``

If the localised file is missing, the dialog falls back to the English
one, then to a short inline message so the action never fails.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ...app_paths import manual_doc_path
from ..i18n import LANG_SPANISH, current_language, tr


def _resolve_manual_path(lang: str) -> Path | None:
    """Find the manual file for ``lang`` (with English fallback)."""
    return manual_doc_path(lang, spanish_code=LANG_SPANISH)


class ManualDialog(QDialog):
    """Modal dialog that displays the bundled user manual."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("User manual"))
        self.setModal(False)
        self.resize(900, 720)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(True)
        # Slightly larger reading font.
        font = browser.font()
        if isinstance(font, QFont):
            font.setPointSizeF(max(font.pointSizeF(), 10.0))
            browser.setFont(font)

        lang = current_language()
        path = _resolve_manual_path(lang)
        if path is not None:
            try:
                text = path.read_text(encoding="utf-8")
                browser.setMarkdown(text)
            except OSError as exc:  # pragma: no cover - I/O edge case
                browser.setPlainText(
                    tr("Could not open user manual: {err}").format(err=exc),
                )
        else:
            browser.setPlainText(
                tr(
                    "User manual file not found. It should be at "
                    "'docs/MANUAL.en.md' or 'docs/MANUAL.es.md' next "
                    "to the application.",
                ),
            )

        layout.addWidget(browser, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons, 0, Qt.AlignRight)


__all__ = ["ManualDialog"]
