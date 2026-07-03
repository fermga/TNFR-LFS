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
#
# NOTE: InSim has no cfg.txt key in LFS — it is started at runtime
# with ``/insim <port>`` in the LFS console or by launching with
# ``LFS.exe /insim=29999``. Writing an ``InSim Port`` entry makes LFS
# show a red "unknown setting" warning at the top of the screen.
REQUIRED_SETTINGS: dict[str, str] = {
    "OutSim Mode":     "2",      # 0=off 1=driving 2=driving+replay
    "OutSim Opts":     "1ff",    # OutSimPack2 extended payload (all blocks)
    "OutSim Delay":    "1",      # ~10 ms (LFS ticks at 100 Hz)
    "OutSim IP":       "127.0.0.1",
    "OutSim Port":     "30000",
    "OutSim ID":       "0",
    "OutGauge Mode":   "1",
    "OutGauge Delay":  "1",
    "OutGauge IP":     "127.0.0.1",
    "OutGauge Port":   "30001",
    "OutGauge ID":     "0",
}

# Candidate install folders are owned by :mod:`lfs_paths`
# (``lfs_paths._STATIC_CANDIDATES``). This module keeps only the
# cfg.txt patching logic and a couple of GUI-independent path
# helpers — it deliberately does not duplicate discovery logic.


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


def is_valid_lfs_dir(path: Path) -> bool:
    """True if *path* looks like an LFS install (has LFS.exe or cfg.txt).

    GUI-independent primitive. Application code should call
    :func:`lfs_telemetry.lfs_paths.is_valid_lfs_dir` instead — it
    accepts ``None`` and is the public entry point.
    """
    return _looks_like_lfs_dir(path)


def cfg_path_for(lfs_dir: Path) -> Path:
    """Return the cfg.txt path inside *lfs_dir*."""
    return Path(lfs_dir) / "cfg.txt"


def lfs_data_dir(lfs_dir: Path) -> Path:
    """Return the ``<lfs_dir>/data`` folder where LFS writes exports."""
    return Path(lfs_dir) / "data"


def lfs_setups_dir(lfs_dir: Path, car_key: str | None = None) -> Path:
    """Return ``<lfs_dir>/data/setups[/<CAR>]``.

    When ``car_key`` is provided the per-car subfolder is returned;
    otherwise the root setups folder is returned. The path may not
    exist on disk yet — this is purely a path helper used to seed
    the default location of a file-picker dialog.
    """
    root = lfs_data_dir(lfs_dir) / "setups"
    if car_key:
        return root / car_key.upper()
    return root


def find_lfs_car_info_bins(lfs_dir: Path) -> list[Path]:
    """Return every ``*_CAR_info.bin`` file found under ``<lfs_dir>/data``.

    LFS Programmer Mode writes these exports there. Returns an empty
    list if the data folder does not exist or contains no exports.
    """
    data = lfs_data_dir(lfs_dir)
    if not data.is_dir():
        return []
    # Be permissive about case (LFS uses ``FBM_CAR_info.bin``, mods may
    # use any case). Sort for deterministic output.
    found: list[Path] = []
    seen: set[str] = set()
    for p in data.iterdir():
        if not p.is_file():
            continue
        if p.name.lower().endswith("_car_info.bin"):
            key = p.name.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(p)
    found.sort(key=lambda q: q.name.lower())
    return found


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


def preview_cfg_patch(lfs_dir: Path) -> dict[str, tuple[str | None, str]]:
    """Return a dry-run preview of what :func:`patch_cfg` would change.

    The mapping is keyed by setting name and the value is
    ``(current_value, target_value)``:

    * ``current_value is None`` \u2192 key is missing and would be **added**
    * ``current_value == target_value`` \u2192 key is **already correct**
    * otherwise the key would be **updated** in place

    Raises :class:`FileNotFoundError` if cfg.txt does not exist.
    """
    cfg = cfg_path_for(lfs_dir)
    if not cfg.exists():
        raise FileNotFoundError(
            f"{cfg} does not exist. Launch LFS once to generate "
            "cfg.txt, then quit and try again.",
        )
    text = cfg.read_text(encoding="latin-1")
    current: dict[str, str] = {}
    for line in text.splitlines():
        for key in REQUIRED_SETTINGS:
            if key in current:
                continue
            if line == key:
                current[key] = ""
            elif line.startswith(key + " "):
                current[key] = line[len(key) + 1:]
    return {
        key: (current.get(key), target)
        for key, target in REQUIRED_SETTINGS.items()
    }


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
    return any((path / marker).exists() for marker in ("LFS.exe", "cfg.txt"))


def read_lfs_vr_mode(lfs_dir: Path) -> tuple[str, int] | None:
    """Detect whether LFS is set to render to a VR headset in ``cfg.txt``.

    Returns ``(backend, vr_system)`` where ``backend`` is ``"OpenVR"``
    or ``"Oculus"`` and ``vr_system`` is LFS's raw VR-system index, or
    ``None`` when LFS targets a flat monitor / 3D-display device (no
    headset), when ``cfg.txt`` is missing, or when the setting cannot
    be parsed.

    Modern LFS (0.6+) stores the display device on a single
    ``G3D_OPTIONS`` line::

        G3D_OPTIONS <a> <b> <c> <d> <device> <vr_system> <fmt>

    where ``device`` is ``0`` = TV / monitor / projector, ``1`` = 3D
    display device and ``2`` = VR headset, and ``vr_system`` is ``0`` =
    Oculus Rift, ``1`` = OpenVR (SteamVR). The field positions are
    confirmed against LFS's own language strings (``3h_g3ddev*`` for
    the device selector, ``3h_hsriftvr``/``3h_hsopenvr`` for the VR
    system). When present, this line is authoritative.

    Legacy builds used discrete ``OpenVR Mode N`` / ``Oculus Mode N``
    lines; those remain recognised for backward compatibility.
    """
    cfg = cfg_path_for(lfs_dir)
    if not cfg.exists():
        return None
    try:
        text = cfg.read_text(encoding="latin-1")
    except OSError:
        return None

    legacy: tuple[str, int] | None = None
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue

        # Modern single-line form. When present it decides the answer:
        # a non-headset device means VR is off, regardless of any other
        # (stale) line still sitting in the file. ``parts[0]`` is the
        # ``G3D_OPTIONS`` key itself, so the 5th value (device) is at
        # index 5 and the 6th (vr_system) at index 6.
        if parts[0] == "G3D_OPTIONS" and len(parts) >= 7:
            try:
                device = int(parts[5])
                vr_system = int(parts[6])
            except ValueError:
                continue
            if device != 2:  # 0 = monitor, 1 = 3D display — not a headset
                return None
            backend = "OpenVR" if vr_system == 1 else "Oculus"
            return backend, vr_system

        # Legacy discrete form: "OpenVR Mode 1" / "Oculus Mode 0".
        if (
            len(parts) >= 3
            and parts[0] in {"OpenVR", "Oculus", "VR"}
            and parts[1] == "Mode"
        ):
            try:
                value = int(parts[2])
            except ValueError:
                continue
            if value > 0:
                legacy = (parts[0], value)

    return legacy


__all__ = [
    "REQUIRED_SETTINGS",
    "PatchResult",
    "cfg_path_for",
    "find_lfs_car_info_bins",
    "is_valid_lfs_dir",
    "lfs_data_dir",
    "lfs_setups_dir",
    "manual_instructions",
    "patch_cfg",
    "preview_cfg_patch",
    "read_lfs_vr_mode",
]


# Allow ``python -m lfs_telemetry.lfs_config <lfs_dir>``.
if __name__ == "__main__":  # pragma: no cover - small CLI helper
    import sys

    from .lfs_paths import autodetect_lfs_dir

    target: Path
    if len(sys.argv) >= 2:
        target = Path(sys.argv[1])
    else:
        guess = autodetect_lfs_dir()
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
