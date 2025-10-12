"""Custom MkDocs macros for the TNFR × LFS documentation."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_cli_block(pyproject_path: Path) -> str:
    """Extract the ``[tool.tnfr_lfs]`` block from ``pyproject.toml`` as TOML text."""

    if not pyproject_path.exists():
        return "# `[tool.tnfr_lfs]` section not found"

    contents = pyproject_path.read_text(encoding="utf-8")
    captured: list[str] = []
    recording = False
    for line in contents.splitlines():
        if line.startswith("[tool.tnfr_lfs"):
            recording = True
        elif recording and line.startswith("[tool."):
            break
        if recording:
            captured.append(line)

    if not captured:
        return "# `[tool.tnfr_lfs]` section not found"

    return "\n".join(captured)


def define_env(env) -> None:  # pragma: no cover - exercised during docs build
    """Register macros exposed to the documentation build."""

    pyproject_path = PROJECT_ROOT / "pyproject.toml"

    @env.macro  # type: ignore[attr-defined]
    def render_config_defaults() -> str:
        """Return the canonical CLI defaults as a fenced TOML block."""

        contents = _read_cli_block(pyproject_path)
        return f"```toml\n{contents}\n```"
