"""Resolve configuration parameter overrides for telemetry sessions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping as MappingABC
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

__all__ = ["get_params", "load_detection_config"]


_DETECTION_RESOURCE_PACKAGE = "tnfr_lfs.resources.config"
_DETECTION_RESOURCE_NAME = "detection.yaml"


def get_params(
    config: Mapping[str, Any],
    *,
    car_class: str | None = None,
    car_model: str | None = None,
    track_name: str | None = None,
    tyre_compound: str | None = None,
) -> Mapping[str, Any]:
    """Merge parameter overrides for the requested session metadata."""

    result: dict[str, Any] = {}

    def merge(payload: Mapping[str, Any] | None) -> None:
        if not isinstance(payload, MappingABC):
            return
        _deep_merge(result, payload)

    def merge_section(section: Mapping[str, Any] | None) -> None:
        if not isinstance(section, MappingABC):
            return
        defaults = section.get("defaults")
        if isinstance(defaults, MappingABC):
            merge(defaults)
        base_entries = {
            str(key): value
            for key, value in section.items()
            if key not in {"defaults", "tracks", "compounds"}
        }
        merge(base_entries)
        _merge_compounds(section)

    def _merge_compounds(section: Mapping[str, Any]) -> None:
        compounds = section.get("compounds")
        if not isinstance(compounds, MappingABC):
            return
        default_entry = None
        for key in ("__default__", "*"):
            candidate = compounds.get(key)
            if isinstance(candidate, MappingABC):
                default_entry = candidate
                break
        if default_entry is not None:
            merge(default_entry)
        compound_key = _normalise_identifier(tyre_compound)
        if compound_key is None:
            return
        for raw_key, payload in compounds.items():
            if not isinstance(payload, MappingABC):
                continue
            if _normalise_identifier(raw_key) == compound_key:
                merge(payload)
                break

    merge(config.get("defaults"))
    _merge_compounds(config)

    tracks_table = config.get("tracks")
    if isinstance(tracks_table, MappingABC):
        merge_section(_lookup_section(tracks_table, "__default__"))
        merge_section(_lookup_section(tracks_table, track_name))

    classes_table = config.get("classes")
    if isinstance(classes_table, MappingABC):
        merge_section(_lookup_section(classes_table, "__default__"))
        class_section = _lookup_section(classes_table, car_class)
        merge_section(class_section)
        if isinstance(class_section, MappingABC):
            track_overrides = class_section.get("tracks")
            if isinstance(track_overrides, MappingABC):
                merge_section(_lookup_section(track_overrides, "__default__"))
                merge_section(_lookup_section(track_overrides, track_name))

    cars_table = config.get("cars")
    if isinstance(cars_table, MappingABC):
        merge_section(_lookup_section(cars_table, "__default__"))
        car_section = _lookup_section(cars_table, car_model)
        merge_section(car_section)
        if isinstance(car_section, MappingABC):
            track_overrides = car_section.get("tracks")
            if isinstance(track_overrides, MappingABC):
                merge_section(_lookup_section(track_overrides, "__default__"))
                merge_section(_lookup_section(track_overrides, track_name))

    return MappingProxyType(dict(result))


def load_detection_config(
    path: str | Path | None = None,
    *,
    search_paths: Iterable[str | Path] | None = None,
    pack_root: Path | None = None,
) -> Mapping[str, Any]:
    """Load the detection override table honouring site-specific fallbacks.

    Parameters
    ----------
    path:
        Absolute or relative path to a YAML file. When supplied the loader
        skips the search order and reads this file directly.
    search_paths:
        Optional iterable of directories or files to inspect. Entries pointing
        to directories are resolved against ``detection.yaml``. The first
        existing file wins.
    pack_root:
        Optional configuration pack root. The loader inspects
        ``pack_root / "config" / "detection.yaml"`` followed by
        ``pack_root / "detection.yaml"`` before falling back to the packaged
        defaults bundled with :mod:`tnfr_lfs`.
    """

    if path is not None:
        candidate = Path(path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return _load_detection_payload(candidate)

    candidates: list[Path] = []

    if search_paths is not None:
        for entry in search_paths:
            entry_path = Path(entry).expanduser()
            if entry_path.is_dir():
                candidates.append(entry_path / _DETECTION_RESOURCE_NAME)
            else:
                candidates.append(entry_path)

    if pack_root is not None:
        root = Path(pack_root).expanduser()
        candidates.extend(
            [
                root / "config" / _DETECTION_RESOURCE_NAME,
                root / _DETECTION_RESOURCE_NAME,
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return _load_detection_payload(candidate)

    resource = resources.files(_DETECTION_RESOURCE_PACKAGE).joinpath(
        _DETECTION_RESOURCE_NAME
    )
    payload = resource.read_text(encoding="utf-8")
    return _load_detection_from_text(payload, source=str(resource))


def _deep_merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        key_str = str(key)
        existing = target.get(key_str)
        if isinstance(existing, MappingABC) and isinstance(value, MappingABC):
            merged = dict(existing)
            _deep_merge(merged, value)
            target[key_str] = merged
        elif isinstance(value, MappingABC):
            target[key_str] = _deep_copy_mapping(value)
        else:
            target[key_str] = value


def _deep_copy_mapping(source: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in source.items():
        key_str = str(key)
        if isinstance(value, MappingABC):
            copied[key_str] = _deep_copy_mapping(value)
        else:
            copied[key_str] = value
    return copied


def _load_detection_payload(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as buffer:
        payload = buffer.read()
    return _load_detection_from_text(payload, source=str(path))


def _load_detection_from_text(payload: str, *, source: str) -> Mapping[str, Any]:
    try:
        data = yaml.safe_load(payload)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid YAML in detection configuration: {source}") from exc

    if data is None:
        return MappingProxyType({})
    if not isinstance(data, MappingABC):
        raise TypeError(
            f"Detection configuration in {source!s} must decode to a mapping"
        )
    return MappingProxyType(_deep_copy_mapping(data))


def _lookup_section(
    table: Mapping[str, Any] | None, key: str | None
) -> Mapping[str, Any] | None:
    if not isinstance(table, MappingABC) or key is None:
        return None
    candidate = table.get(key)
    if isinstance(candidate, MappingABC):
        return candidate
    normalised = _normalise_identifier(key)
    if normalised is None:
        return None
    for raw_key, value in table.items():
        if not isinstance(value, MappingABC):
            continue
        if _normalise_identifier(raw_key) == normalised:
            return value
    return None


def _normalise_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    filtered = [char for char in str(value).lower() if char.isalnum() or char == "_"]
    cleaned = "".join(filtered).strip("_")
    return cleaned or None

