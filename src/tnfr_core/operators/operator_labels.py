"""Shared utilities for working with structural operator identifiers."""

from __future__ import annotations

import warnings
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from typing import List, Mapping, Sequence, Tuple

__all__ = [
    "STRUCTURAL_OPERATOR_LABELS",
    "_STRUCTURAL_IDENTIFIER_ALIASES",
    "normalize_structural_operator_identifier",
    "canonical_operator_label",
    "silence_event_payloads",
]


STRUCTURAL_OPERATOR_LABELS: Mapping[str, str] = {
    "AL": "Support",
    "EN": "Reception",
    "IL": "Coherence",
    "OZ": "Dissonance",
    "UM": "Coupling",
    "RA": "Propagation",
    "SILENCE": "Structural silence",
    "VAL": "Amplification",
    "NUL": "Contraction",
    "THOL": "Auto-organisation",
    "ZHIR": "Transformation",
    "NAV": "Transition",
    "REMESH": "Remeshing",
}

_STRUCTURAL_IDENTIFIER_ALIASES: Mapping[str, str] = {
    "A'L": "AL",
    "A’L": "AL",
    "E'N": "EN",
    "E’N": "EN",
    "I'L": "IL",
    "I’L": "IL",
    "O'Z": "OZ",
    "O’Z": "OZ",
    "U'M": "UM",
    "U’M": "UM",
    "R'A": "RA",
    "R’A": "RA",
    "SH'A": "SILENCE",
    "SH’A": "SILENCE",
    "SHA": "SILENCE",
    "VA'L": "VAL",
    "VA’L": "VAL",
    "NU'L": "NUL",
    "NU’L": "NUL",
    "T'HOL": "THOL",
    "T’HOL": "THOL",
    "AUTOORGANISATION": "THOL",
    "AUTO ORGANISATION": "THOL",
    "AUTO-ORGANISATION": "THOL",
    "AUTOORGANIZATION": "THOL",
    "AUTO ORGANIZATION": "THOL",
    "AUTO-ORGANIZATION": "THOL",
    "AUTOORGANIZACION": "THOL",
    "AUTO ORGANIZACION": "THOL",
    "AUTO-ORGANIZACION": "THOL",
    "AUTOORGANIZACIÓN": "THOL",
    "AUTO ORGANIZACIÓN": "THOL",
    "AUTO-ORGANIZACIÓN": "THOL",
    "Z'HIR": "ZHIR",
    "Z’HIR": "ZHIR",
    "NA'V": "NAV",
    "NA’V": "NAV",
    "NAV": "NAV",
    "TRANSITION": "NAV",
    "TRANSICION": "NAV",
    "TRANSICIÓN": "NAV",
    "REMESH": "REMESH",
    "RE'MESH": "REMESH",
    "RE’MESH": "REMESH",
}

try:
    if isinstance(STRUCTURAL_OPERATOR_LABELS, MappingABC):
        labels = dict(STRUCTURAL_OPERATOR_LABELS)
        labels.setdefault("NAV", "Transition")
        labels.setdefault("THOL", "Auto-organisation")
        STRUCTURAL_OPERATOR_LABELS = labels
    else:
        STRUCTURAL_OPERATOR_LABELS = tuple(  # type: ignore[assignment]
            sorted(set(STRUCTURAL_OPERATOR_LABELS) | {"NAV", "THOL"})
        )
except Exception:
    pass

_DEPRECATED_STRUCTURAL_IDENTIFIER_ALIASES: Mapping[str, str] = {
    "SILENCIO": "SILENCE",
}


def normalize_structural_operator_identifier(identifier: str) -> str:
    """Return the canonical structural identifier for ``identifier``."""

    if not isinstance(identifier, str):
        return str(identifier)
    key = identifier.upper()
    if key in _DEPRECATED_STRUCTURAL_IDENTIFIER_ALIASES:
        warnings.warn(
            "The 'SILENCIO' structural identifier is deprecated; use 'SILENCE' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        key = _DEPRECATED_STRUCTURAL_IDENTIFIER_ALIASES[key]
    return _STRUCTURAL_IDENTIFIER_ALIASES.get(key, key)


def canonical_operator_label(identifier: str) -> str:
    """Return the canonical structural label for an operator identifier."""

    if not isinstance(identifier, str):
        return str(identifier)
    key = normalize_structural_operator_identifier(identifier)
    return STRUCTURAL_OPERATOR_LABELS.get(key, identifier)


def silence_event_payloads(
    events: Mapping[str, Sequence[Mapping[str, object]] | None] | None,
) -> Tuple[Mapping[str, object], ...]:
    """Return all silence payloads, accepting case-insensitive identifiers."""

    if not events:
        return ()

    collected: List[Mapping[str, object]] = []
    for name, payload in events.items():
        if normalize_structural_operator_identifier(name) != "SILENCE":
            continue
        if not payload:
            continue
        if isinstance(payload, SequenceABC) and not isinstance(payload, MappingABC):
            collected.extend(payload)  # type: ignore[list-item]
        else:
            collected.append(payload)  # type: ignore[arg-type]
    return tuple(collected)
