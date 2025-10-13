"""Shared definitions for TNFR phase nomenclature."""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Callable, Mapping, Sequence, Tuple, TypeVar

from tnfr_core._canonical import CANONICAL_REQUESTED, import_tnfr

_LOCAL_PHASE_SEQUENCE: Tuple[str, ...] = (
    "entry1",
    "entry2",
    "apex3a",
    "apex3b",
    "exit4",
)

_LOCAL_LEGACY_PHASE_MAP: dict[str, Tuple[str, ...]] = {
    "entry": ("entry1", "entry2"),
    "apex": ("apex3a", "apex3b"),
    "exit": ("exit4",),
}

PHASE_SEQUENCE: Tuple[str, ...] = _LOCAL_PHASE_SEQUENCE
LEGACY_PHASE_MAP: dict[str, Tuple[str, ...]] = dict(_LOCAL_LEGACY_PHASE_MAP)

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
        warnings.warn(
            "Phase alias %r is deprecated and will be removed in a future release" % phase,
            DeprecationWarning,
            stacklevel=2,
        )
        return LEGACY_PHASE_MAP[key] + (key,)
    return (key,)


def phase_family(phase: str) -> str:
    """Return the legacy family identifier for ``phase``."""

    key = normalise_phase_key(phase)
    if key in LEGACY_PHASE_MAP:
        warnings.warn(
            "Phase alias %r is deprecated and will be removed in a future release" % phase,
            DeprecationWarning,
            stacklevel=2,
        )
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
    for alias in LEGACY_PHASE_MAP:
        if alias in normalised:
            warnings.warn(
                "Phase alias %r is deprecated and will be removed in a future release" % alias,
                DeprecationWarning,
                stacklevel=2,
            )
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
        warnings.warn(
            "Phase alias %r is deprecated and will be removed in a future release" % alias,
            DeprecationWarning,
            stacklevel=2,
        )
        if combine is not None:
            normalised[alias] = combine(tuple(candidates))
        else:
            normalised[alias] = candidates[-1]
    return normalised


if CANONICAL_REQUESTED:  # pragma: no cover - depends on optional package
    tnfr = import_tnfr()
    canonical_phases = import_module(f"{tnfr.__name__}.equations.phases")

    PHASE_SEQUENCE = getattr(
        canonical_phases, "PHASE_SEQUENCE", _LOCAL_PHASE_SEQUENCE
    )
    LEGACY_PHASE_MAP = dict(
        getattr(canonical_phases, "LEGACY_PHASE_MAP", _LOCAL_LEGACY_PHASE_MAP)
    )

    _PHASE_TO_FAMILY = getattr(canonical_phases, "_PHASE_TO_FAMILY", _PHASE_TO_FAMILY)

    normalise_phase_key = getattr(
        canonical_phases, "normalise_phase_key", normalise_phase_key
    )
    expand_phase_alias = getattr(
        canonical_phases, "expand_phase_alias", expand_phase_alias
    )
    phase_family = getattr(canonical_phases, "phase_family", phase_family)
    replicate_phase_aliases = getattr(
        canonical_phases, "replicate_phase_aliases", replicate_phase_aliases
    )
