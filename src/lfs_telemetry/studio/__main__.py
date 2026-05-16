"""``python -m lfs_telemetry.studio`` entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lfs-telemetry-studio",
        description="Native LFS telemetry analyser (PySide6 + pyqtgraph).",
    )
    parser.add_argument(
        "workspace", nargs="?", default="captures",
        help="Workspace folder containing capture .csv files (default: ./captures).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Hidden switch used by the in-app capture runner: when running as a
    # PyInstaller-frozen launcher there is no ``python -m lfs_telemetry``
    # available, so the launcher re-exec's itself with ``--cli <subcmd>
    # ...`` and we delegate straight to the regular CLI dispatcher.
    if raw and raw[0] == "--cli":
        from lfs_telemetry.cli import main as cli_main
        return cli_main(raw[1:])

    args = _parse_args(raw)
    workspace = Path(args.workspace).expanduser().resolve()

    # Imported lazily so ``--help`` stays snappy and a missing PySide6
    # gives a focused error instead of a dependency tower-of-babel
    # traceback at import time.
    from lfs_telemetry.studio.app import create_app
    from lfs_telemetry.studio.main_window import MainWindow

    app = create_app(sys.argv)
    window = MainWindow(workspace)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
