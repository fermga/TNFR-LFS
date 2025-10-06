"""Shared definitions for TNFR phase nomenclature."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence, Tuple, TypeVar

PHASE_SEQUENCE: Tuple[str, ...] = ("entry1", "entry2", "apex3a", "apex3b", "exit4")

LEGACY_PHASE_MAP: dict[str, Tuple[str, ...]] = {
    "entry": ("entry1", "entry2"),
    "apex": ("apex3a", "apex3b"),
    "exit": ("exit4",),
}

T = TypeVar("T")

_PHASE_TO_FAMILY: dict[str, str] = {}
for legacy, phases in LEGACY_PHASE_MAP.items():
    for phase in phases:
        _PHASE_TO_FAMILY[phase] = legacy
for phase in PHASE_SEQUENCE:
    _PHASE_TO_FAMILY.setdefault(phase, phase)


def normalise_phase_key(phase: str) -> str:
    """Return the canonical lower-case identifier for ``phase``."""

    return str(phase).lower()


def expand_phase_alias(phase: str) -> Tuple[str, ...]:
    """Return all concrete phases associated with ``phase``."""

    key = normalise_phase_key(phase)
    if key in LEGACY_PHASE_MAP:
        return LEGACY_PHASE_MAP[key] + (key,)
    return (key,)


def phase_family(phase: str) -> str:
    """Return the legacy family identifier for ``phase``."""

    key = normalise_phase_key(phase)
    return _PHASE_TO_FAMILY.get(key, key)


def replicate_phase_aliases(
    payload: Mapping[str, T],
    *,
    combine: Callable[[Sequence[T]], T] | None = None,
) -> dict[str, T]:
    """Return ``payload`` enriched with legacy phase aliases.

    Parameters
    ----------
    payload:
        Mapping keyed by phase identifiers. Keys are normalised to lower case in
        the resulting dictionary. Values are copied verbatim unless ``combine``
        is provided.
    combine:
        Optional reducer applied when populating a legacy alias from multiple
        concrete phase values. When omitted the last available concrete phase
        value is reused for the alias.
    """

    normalised: dict[str, T] = {
        normalise_phase_key(str(key)): value for key, value in payload.items()
    }
    for alias, phases in LEGACY_PHASE_MAP.items():
        if alias in normalised:
            continue
        candidates = [
            normalised[phase]
            for phase in phases
            if phase in normalised
        ]
        if not candidates:
            continue
        if combine is not None:
            normalised[alias] = combine(tuple(candidates))
        else:
            normalised[alias] = candidates[-1]
    return normalised
