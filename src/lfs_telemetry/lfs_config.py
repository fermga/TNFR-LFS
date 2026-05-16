"""Helpers to read and patch the LFS ``cfg.txt`` with the OutSim, OutGauge
and InSim settings required by lfs-telemetry.

This module is GUI-independent so it can be used both from the Studio
"Configure LFS…" dialog and from the standalone ``scripts/patch_lfs_cfg.py``
helper.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

# Required LFS cfg.txt key/value pairs (LFS keys are case-sensitive,
# space-separated). Values are the strings as they must appear on disk.
REQUIRED_SETTINGS: dict[str, str] = {
    "OutSim Mode":     "2",      # 2 = OutSimPack2 extended packets
    "OutSim Opts":     "1ff",    # full extended payload
    "OutSim Delay":    "1",      # ~10 ms (LFS ticks at 100 Hz)
    "OutSim IP":       "127.0.0.1",
    "OutSim Port":     "30000",
    "OutSim ID":       "0",
    "OutGauge Mode":   "1",
    "OutGauge Delay":  "1",
    "OutGauge IP":     "127.0.0.1",
    "OutGauge Port":   "30001",
    "OutGauge ID":     "0",
    "InSim Port":      "29999",
}

# Candidate install folders we probe when the user has not chosen one yet.
_DEFAULT_LFS_CANDIDATES: tuple[Path, ...] = (
    Path(r"C:\LFS"),
    Path(r"C:\Program Files\LFS"),
    Path(r"C:\Program Files (x86)\LFS"),
    Path(r"D:\LFS"),
    Path(r"D:\Games\LFS"),
    Path(r"C:\Games\LFS"),
)


@dataclass(slots=True)
class PatchResult:
    """Outcome of :func:`patch_cfg`."""

    cfg_path: Path
    backup_path: Path | None
    updated: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    final_values: dict[str, str] = field(default_factory=dict)

    def summary_text(self) -> str:
        lines = [f"Patched {self.cfg_path}"]
        if self.backup_path is not None:
            lines.append(f"Backup: {self.backup_path}")
        lines.append("")
        for key, value in self.final_values.items():
            tag = ("UPDATED" if key in self.updated
                   else "ADDED  " if key in self.added
                   else "OK     ")
            lines.append(f"  [{tag}] {key:18s} {value}")
        return "\n".join(lines)


def find_default_lfs_dir() -> Path | None:
    """Return the first plausible LFS install folder, or ``None``.

    A folder qualifies if it exists and contains either ``LFS.exe`` or
    ``cfg.txt``.
    """
    for cand in _DEFAULT_LFS_CANDIDATES:
        if _looks_like_lfs_dir(cand):
            return cand
    return None


def is_valid_lfs_dir(path: Path) -> bool:
    """True if *path* looks like an LFS install (has LFS.exe or cfg.txt)."""
    return _looks_like_lfs_dir(path)


def cfg_path_for(lfs_dir: Path) -> Path:
    """Return the cfg.txt path inside *lfs_dir*."""
    return Path(lfs_dir) / "cfg.txt"


def manual_instructions() -> str:
    """Return the copy/paste-ready snippet shown to users."""
    blocks: list[str] = []
    last_prefix: str | None = None
    for key, value in REQUIRED_SETTINGS.items():
        prefix = key.split(" ", 1)[0]
        if last_prefix is not None and prefix != last_prefix:
            blocks.append("")
        blocks.append(f"{key} {value}")
        last_prefix = prefix
    return "\n".join(blocks)


def patch_cfg(lfs_dir: Path) -> PatchResult:
    """Patch ``<lfs_dir>/cfg.txt`` with :data:`REQUIRED_SETTINGS`.

    * Existing keys are updated in place.
    * Missing keys are appended at the end under a marker comment.
    * On the first call a ``cfg.txt.bak`` backup is created next to it.

    Raises :class:`FileNotFoundError` if the cfg file does not exist
    (LFS must be launched once to generate it).
    """
    cfg = cfg_path_for(lfs_dir)
    if not cfg.exists():
        raise FileNotFoundError(
            f"{cfg} does not exist. Launch LFS once to generate cfg.txt, "
            "then quit and try again.",
        )

    backup = cfg.with_suffix(".txt.bak")
    backup_made: Path | None = None
    if not backup.exists():
        shutil.copy2(cfg, backup)
        backup_made = backup

    text = cfg.read_text(encoding="latin-1")
    # Preserve the file's original line ending style.
    eol = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()

    seen: dict[str, int] = {}
    updated: list[str] = []

    for i, line in enumerate(lines):
        for key, value in REQUIRED_SETTINGS.items():
            if key in seen:
                continue
            if line == key or line.startswith(key + " "):
                if line != f"{key} {value}":
                    lines[i] = f"{key} {value}"
                    updated.append(key)
                seen[key] = i
                break

    added = [k for k in REQUIRED_SETTINGS if k not in seen]
    if added:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("// Added by lfs-telemetry (lfs_config.patch_cfg)")
        for key in added:
            lines.append(f"{key} {REQUIRED_SETTINGS[key]}")

    new_text = eol.join(lines) + eol
    cfg.write_text(new_text, encoding="latin-1")

    return PatchResult(
        cfg_path=cfg,
        backup_path=backup_made if backup_made is not None else backup,
        updated=updated,
        added=added,
        final_values=dict(REQUIRED_SETTINGS),
    )


def _looks_like_lfs_dir(path: Path) -> bool:
    try:
        if not path.is_dir():
            return False
    except OSError:
        return False
    for marker in ("LFS.exe", "cfg.txt"):
        if (path / marker).exists():
            return True
    return False


__all__ = [
    "REQUIRED_SETTINGS",
    "PatchResult",
    "find_default_lfs_dir",
    "is_valid_lfs_dir",
    "cfg_path_for",
    "manual_instructions",
    "patch_cfg",
]


# Allow ``python -m lfs_telemetry.lfs_config <lfs_dir>``.
if __name__ == "__main__":  # pragma: no cover - small CLI helper
    import sys

    target: Path
    if len(sys.argv) >= 2:
        target = Path(sys.argv[1])
    else:
        guess = find_default_lfs_dir()
        if guess is None:
            print(
                "Usage: python -m lfs_telemetry.lfs_config "
                "<LFS install folder>",
            )
            sys.exit(1)
        target = guess

    try:
        result = patch_cfg(target)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print(result.summary_text())
    print()
    print("Done. Now launch LFS and enter a session.")
    sys.exit(0)
