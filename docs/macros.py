"""Custom MkDocs macros for the TNFR × LFS documentation."""

from __future__ import annotations

from pathlib import Path


def define_env(env) -> None:  # pragma: no cover - exercised during docs build
    """Register macros exposed to the documentation build."""

    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "tnfr_lfs.toml"

    @env.macro  # type: ignore[attr-defined]
    def render_config_defaults() -> str:
        """Return the bundled ``tnfr_lfs.toml`` as a fenced TOML block."""

        contents = config_path.read_text(encoding="utf-8").rstrip()
        return f"```toml\n{contents}\n```"
