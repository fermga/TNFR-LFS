"""Tyre compound compatibility data for Live for Speed cars."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tnfr_lfs.telemetry._tyre_compound import normalise_compound_key

__all__ = [
    "CAR_COMPOUND_COMPATIBILITY",
    "get_allowed_compounds",
    "normalise_car_model",
    "normalise_compound_identifier",
]


def normalise_car_model(value: object) -> str | None:
    """Normalise ``value`` into an uppercase car model identifier."""

    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if text in {"__UNKNOWN__", "__DEFAULT__"}:
        return None
    return text


def normalise_compound_identifier(value: object) -> str | None:
    """Normalise ``value`` into the canonical compound key."""

    key = normalise_compound_key(value)
    if not key:
        return None
    if key in {"__unknown__", "__default__", "default", "unknown"}:
        return None
    return key


def _freeze_compounds(values: Sequence[str]) -> frozenset[str]:
    tokens: set[str] = set()
    for entry in values:
        normalised = normalise_compound_identifier(entry)
        if normalised:
            tokens.add(normalised)
    return frozenset(tokens)


_ROAD_COMPOUNDS = _freeze_compounds(
    (
        "R1",
        "R2",
        "R3",
        "R4",
        "Road Normal",
        "Road Super",
        "Hybrid",
        "Knobbly",
        "Dirt",
        "Wet",
    )
)

_SLICK_COMPOUNDS = _freeze_compounds(
    (
        "Super Soft",
        "Soft",
        "Medium",
        "Hard",
        "Super Hard",
        "Wet",
    )
)

_CAR_COMPOUND_SPECS: Mapping[str, frozenset[str]] = {
    "UF1": _ROAD_COMPOUNDS,
    "XFG": _ROAD_COMPOUNDS,
    "XRG": _ROAD_COMPOUNDS,
    "LX4": _ROAD_COMPOUNDS,
    "LX6": _ROAD_COMPOUNDS,
    "RB4": _ROAD_COMPOUNDS,
    "FXO": _ROAD_COMPOUNDS,
    "XRT": _ROAD_COMPOUNDS,
    "RAC": _ROAD_COMPOUNDS,
    "FZ5": _ROAD_COMPOUNDS,
    "MRT": _ROAD_COMPOUNDS,
    "FBM": _SLICK_COMPOUNDS,
    "FOX": _SLICK_COMPOUNDS,
    "FO8": _SLICK_COMPOUNDS,
    "BF1": _SLICK_COMPOUNDS,
    "UFR": _SLICK_COMPOUNDS,
    "XFR": _SLICK_COMPOUNDS,
    "FXR": _SLICK_COMPOUNDS,
    "XRR": _SLICK_COMPOUNDS,
    "FZR": _SLICK_COMPOUNDS,
}


CAR_COMPOUND_COMPATIBILITY: Mapping[str, frozenset[str]] = dict(_CAR_COMPOUND_SPECS)


def get_allowed_compounds(car_model: object) -> frozenset[str]:
    """Return the allowed compound keys for ``car_model`` if known."""

    normalised = normalise_car_model(car_model)
    if normalised is None:
        return frozenset()
    return CAR_COMPOUND_COMPATIBILITY.get(normalised, frozenset())
