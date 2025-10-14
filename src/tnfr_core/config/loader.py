"""Resolve configuration parameter overrides for telemetry sessions."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from types import MappingProxyType
from typing import Any, Mapping

__all__ = ["get_params"]


def get_params(
    config: Mapping[str, Any],
    *,
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

    tracks_table = config.get("tracks")
    if isinstance(tracks_table, MappingABC):
        merge_section(_lookup_section(tracks_table, "__default__"))
        merge_section(_lookup_section(tracks_table, track_name))

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

