"""Mod-car catalogue and dimension lookup.

LFS ships dimensions only for stock cars (and only via CAR_info.bin export
in Programmer Mode). Mod cars are identified on the wire by a 32-bit
SkinID rendered as a 6-char lowercase hex string (see
:func:`lfs_telemetry.telemetry.protocol.packets.decode_car_id`), and no
authoritative dimension table is shipped by LFS itself.

Detect&Monitor (the community proximity-alert tool) maintains a curated
``cars/mod_sizes.car`` YAML database keyed by these 6-hex SkinIDs. We
mirror that approach with a small JSON file under
``assets/source/mods/mod_sizes.json`` so the radar/proximity layer can
render mod cars at their real footprint instead of falling back to a
generic placeholder, and so the UI can report "X opponents are using mods
unknown to the local catalogue".

The module also supports runtime updates (e.g. an auto-measurer that
observes a mod's bounding box from MCI samples) via :func:`register_mod`.
Registry updates do not touch disk by default; callers that want to
persist them can use :func:`save_mod_database`.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .protocol.packets import _STOCK_CARS

__all__ = [
    "STOCK_CARS",
    "ModInfo",
    "is_stock_car",
    "is_mod_car",
    "classify_car",
    "mod_dimensions_m",
    "is_known_mod",
    "register_mod",
    "load_mod_database",
    "save_mod_database",
    "mod_database_path",
    "all_known_mods",
]

# Public alias for the stock-car set. The leading underscore on the source
# symbol is historical (it predates external consumers).
STOCK_CARS: frozenset[str] = _STOCK_CARS

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_LOCK = threading.RLock()
_MODS: dict[str, "ModInfo"] = {}
_LOADED = False


@dataclass(frozen=True, slots=True)
class ModInfo:
    """Bounding-box footprint and provenance for a single mod car."""

    skin_id: str            # 6-char lowercase hex
    length_m: float
    width_m: float
    name: str | None = None
    source: str | None = None


def is_stock_car(car_id: str) -> bool:
    """True iff ``car_id`` is a 3-letter stock car short name."""
    return bool(car_id) and car_id.upper() in STOCK_CARS


def is_mod_car(car_id: str) -> bool:
    """True iff ``car_id`` looks like a 6-hex mod SkinID."""
    if not car_id or len(car_id) != 6:
        return False
    try:
        int(car_id, 16)
    except ValueError:
        return False
    return car_id.lower() == car_id


def classify_car(car_id: str) -> str:
    """Return ``"stock"``, ``"mod"`` or ``"unknown"`` for a decoded car id."""
    if is_stock_car(car_id):
        return "stock"
    if is_mod_car(car_id):
        return "mod"
    return "unknown"


def mod_database_path() -> Path:
    """Resolve the on-disk JSON catalogue path.

    Honours ``$LFS_TELEMETRY_MOD_DB`` for tests/installers. Otherwise
    prefers ``./assets/source/mods/mod_sizes.json`` (developer checkout)
    and falls back to the bundled copy alongside the package.
    """
    env = os.environ.get("LFS_TELEMETRY_MOD_DB")
    if env:
        return Path(env)
    cwd_path = Path.cwd() / "assets" / "source" / "mods" / "mod_sizes.json"
    if cwd_path.exists():
        return cwd_path
    return _PACKAGE_ROOT / "assets" / "source" / "mods" / "mod_sizes.json"


def _coerce_entry(skin_id: str, raw: dict) -> ModInfo | None:
    try:
        length = float(raw["length_m"])
        width = float(raw["width_m"])
    except (KeyError, TypeError, ValueError):
        return None
    if length <= 0 or width <= 0:
        return None
    return ModInfo(
        skin_id=skin_id.lower(),
        length_m=length,
        width_m=width,
        name=raw.get("name") if isinstance(raw.get("name"), str) else None,
        source=(
            raw.get("source")
            if isinstance(raw.get("source"), str)
            else None
        ),
    )


def load_mod_database(
    path: Path | None = None,
    *,
    force: bool = False,
) -> dict[str, ModInfo]:
    """Load the mod catalogue from JSON, populating the module cache.

    Idempotent: subsequent calls return the cached dict unless ``force``.
    Missing or malformed files yield an empty catalogue (the radar still
    works, mods just appear as generic markers).
    """
    global _LOADED
    with _LOCK:
        if _LOADED and not force and path is None:
            return dict(_MODS)
        target = path or mod_database_path()
        catalogue: dict[str, ModInfo] = {}
        if target.is_file():
            try:
                raw_doc = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw_doc = {}
            entries = (
                raw_doc.get("mods") if isinstance(raw_doc, dict) else None
            )
            if isinstance(entries, dict):
                for skin_id, value in entries.items():
                    if not isinstance(skin_id, str):
                        continue
                    if not isinstance(value, dict):
                        continue
                    if not is_mod_car(skin_id.lower()):
                        continue
                    info = _coerce_entry(skin_id, value)
                    if info is not None:
                        catalogue[info.skin_id] = info
        _MODS.clear()
        _MODS.update(catalogue)
        _LOADED = True
        return dict(_MODS)


def _ensure_loaded() -> None:
    if not _LOADED:
        load_mod_database()


def mod_dimensions_m(car_id: str) -> tuple[float, float] | None:
    """Return ``(length_m, width_m)`` for a mod, or ``None`` if unknown."""
    if not is_mod_car(car_id):
        return None
    _ensure_loaded()
    with _LOCK:
        info = _MODS.get(car_id.lower())
    if info is None:
        return None
    return (info.length_m, info.width_m)


def is_known_mod(car_id: str) -> bool:
    """True iff ``car_id`` is a mod AND present in the local catalogue."""
    return mod_dimensions_m(car_id) is not None


def all_known_mods() -> dict[str, ModInfo]:
    """Snapshot copy of the in-memory catalogue."""
    _ensure_loaded()
    with _LOCK:
        return dict(_MODS)


def register_mod(
    car_id: str,
    length_m: float,
    width_m: float,
    *,
    name: str | None = None,
    source: str | None = "auto_measure",
) -> ModInfo | None:
    """Register or overwrite a mod entry in memory (does not write disk)."""
    if not is_mod_car(car_id) or length_m <= 0 or width_m <= 0:
        return None
    _ensure_loaded()
    info = ModInfo(
        skin_id=car_id.lower(),
        length_m=float(length_m),
        width_m=float(width_m),
        name=name,
        source=source,
    )
    with _LOCK:
        _MODS[info.skin_id] = info
    return info


def save_mod_database(path: Path | None = None) -> Path:
    """Persist the current in-memory catalogue to disk as JSON."""
    target = path or mod_database_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    _ensure_loaded()
    with _LOCK:
        entries: dict[str, dict] = {}
        for skin_id, info in sorted(_MODS.items()):
            row: dict[str, object] = {
                "length_m": info.length_m,
                "width_m": info.width_m,
            }
            if info.name:
                row["name"] = info.name
            if info.source:
                row["source"] = info.source
            entries[skin_id] = row
        doc = {"mods": entries}
    target.write_text(
        json.dumps(doc, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return target


def summarise_cars(car_ids: Iterable[str]) -> dict[str, int]:
    """Bucket a stream of decoded car ids into stock/known-mod/unknown-mod."""
    out = {"stock": 0, "known_mod": 0, "unknown_mod": 0, "other": 0}
    for cid in car_ids:
        kind = classify_car(cid)
        if kind == "stock":
            out["stock"] += 1
        elif kind == "mod":
            if is_known_mod(cid):
                out["known_mod"] += 1
            else:
                out["unknown_mod"] += 1
        else:
            out["other"] += 1
    return out
