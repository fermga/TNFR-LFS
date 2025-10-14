"""Helpers for handling tyre compound metadata.

The helpers centralise normalisation logic so the ingestion layers and
offline loaders apply consistent rules when decoding compound hints.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Iterable

__all__ = [
    "extract_compound_hint",
    "normalise_compound_label",
    "normalise_compound_key",
    "resolve_compound_metadata",
]


_NAME_NORMALISER = re.compile(r"[^a-z0-9]+", re.IGNORECASE)
_DIGIT_PATTERN = re.compile(r"^[A-Z]{1,3}\d{1,2}$")

_COMPOUND_ALIASES: dict[str, str] = {
    "soft": "soft",
    "supersoft": "supersoft",
    "super_soft": "supersoft",
    "medium": "medium",
    "med": "medium",
    "hard": "hard",
    "superhard": "superhard",
    "super_hard": "superhard",
    "road_normal": "road_normal",
    "roadnormal": "road_normal",
    "road_super": "road_super",
    "roadsuper": "road_super",
    "hybrid": "hybrid",
    "knobbly": "knobbly",
    "dirt": "dirt",
    "wet": "wet",
}

_METADATA_KEYS: tuple[str, ...] = ("tyre_compound", "compound", "tyre", "tyres")


def normalise_compound_label(value: object) -> str | None:
    """Return a stripped string representation of ``value`` when possible."""

    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            return candidate
    return None


def normalise_compound_key(value: object) -> str | None:
    """Return a normalised key suitable for comparisons."""

    label = normalise_compound_label(value)
    if label is None:
        return None
    cleaned = _NAME_NORMALISER.sub("_", label.lower()).strip("_")
    return cleaned or None


def extract_compound_hint(*texts: str | None) -> str | None:
    """Attempt to extract a tyre compound hint from human-readable ``texts``."""

    for text in texts:
        if not text:
            continue
        for raw_token in re.split(r"[^A-Za-z0-9]+", text):
            if not raw_token:
                continue
            label = normalise_compound_label(raw_token)
            if label is None:
                continue
            upper = label.upper()
            if _DIGIT_PATTERN.fullmatch(upper):
                return upper
            key = normalise_compound_key(label)
            if key is None:
                continue
            alias = _COMPOUND_ALIASES.get(key)
            if alias:
                return alias
    return None


def resolve_compound_metadata(metadata: Mapping[str, object] | None) -> str | None:
    """Search ``metadata`` recursively for a tyre compound declaration."""

    if metadata is None:
        return None

    queue: list[Mapping[str, object]] = [metadata]
    visited: set[int] = set()

    while queue:
        current = queue.pop()
        identifier = id(current)
        if identifier in visited:
            continue
        visited.add(identifier)
        for key, value in current.items():
            key_str = str(key).strip().lower()
            if key_str in _METADATA_KEYS:
                label = normalise_compound_label(value)
                if label is not None:
                    return label
            if isinstance(value, Mapping):
                queue.append(value)
    return None


def merge_compound_sources(sources: Iterable[Mapping[str, object] | None]) -> dict[str, object]:
    """Merge ``sources`` ignoring ``None`` values, later entries overriding."""

    merged: dict[str, object] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            merged[str(key)] = value
    return merged

