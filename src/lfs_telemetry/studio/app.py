"""QApplication factory + theme bootstrap."""

from __future__ import annotations

from typing import Sequence

from PySide6.QtWidgets import QApplication

from . import __version__
from .theme import apply_dark_palette, configure_pyqtgraph


def create_app(argv: Sequence[str] | None = None) -> QApplication:
    """Return a configured :class:`QApplication`.

    Idempotent: if a ``QApplication`` already exists (e.g. inside the
    test runner) we reuse it and just (re-)apply the theme.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(list(argv or []))
    app.setApplicationName("LFS Telemetry Studio")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("LFS-Race-Engineer")
    app.setOrganizationDomain("lfs-race-engineer.local")
    app.setStyle("Fusion")
    apply_dark_palette(app)
    configure_pyqtgraph()
    return app


__all__ = ["create_app"]
