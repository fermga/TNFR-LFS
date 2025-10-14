"""Custom MkDocs macros for the TNFR × LFS documentation."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_CACHE = PROJECT_ROOT / "docs" / "_fixtures"
PRESEEDED_FIXTURES: tuple[str, ...] = (
    "tests/data/synthetic_stint.csv",
)
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


def _ensure_fixture(relative_path: str) -> Path:
    """Make sure a fixture is available within the published docs tree."""

    source = PROJECT_ROOT / relative_path
    if not source.exists():
        raise FileNotFoundError(f"Fixture not found: {relative_path}")

    target = FIXTURE_CACHE / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)

    if not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
        shutil.copy2(source, target)

    return Path("_fixtures") / relative_path


def define_env(env) -> None:  # pragma: no cover - exercised during docs build
    """Register macros exposed to the documentation build."""

    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    for fixture in PRESEEDED_FIXTURES:
        _ensure_fixture(fixture)

    @env.macro  # type: ignore[attr-defined]
    def render_config_defaults() -> str:
        """Return the canonical CLI defaults as a fenced TOML block."""

        contents = _read_cli_block(pyproject_path)
        return f"```toml\n{contents}\n```"

    @env.macro  # type: ignore[attr-defined]
    def fixture_link(relative_path: str, text: str | None = None) -> str:
        """Return a Markdown link to a repository fixture copied into the docs."""

        doc_path = _ensure_fixture(relative_path)
        label = text or relative_path
        return f"[{label}]({doc_path.as_posix()})"
