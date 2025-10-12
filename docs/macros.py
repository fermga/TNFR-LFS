"""Custom MkDocs macros for the TNFR × LFS documentation."""

from __future__ import annotations

from pathlib import Path

from tnfr_lfs.configuration import canonical_cli_config_block


def define_env(env) -> None:  # pragma: no cover - exercised during docs build
    """Register macros exposed to the documentation build."""

    project_root = Path(__file__).resolve().parent.parent
    pyproject_path = project_root / "pyproject.toml"

    @env.macro  # type: ignore[attr-defined]
    def render_config_defaults() -> str:
        """Return the canonical CLI defaults as a fenced TOML block."""

        contents = canonical_cli_config_block(pyproject_path)
        return f"```toml\n{contents}\n```"
