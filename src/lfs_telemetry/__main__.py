"""Module entrypoint so ``python -m lfs_telemetry ...`` works.

Delegates to :func:`lfs_telemetry.cli.main` which dispatches the
``capture`` / ``calibrate`` / ``reslice`` / ``advise`` subcommands.
"""

from __future__ import annotations

import sys

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
