"""Centralised constant definitions shared across TNFR × LFS modules."""

from __future__ import annotations

from importlib import import_module
from types import MappingProxyType
from typing import Mapping

from tnfr_core._canonical import CANONICAL_REQUESTED, import_tnfr


_LOCAL_WHEEL_SUFFIXES: tuple[str, ...] = ("fl", "fr", "rl", "rr")

_LOCAL_WHEEL_LABELS: Mapping[str, str] = MappingProxyType({
    "fl": "FL",
    "fr": "FR",
    "rl": "RL",
    "rr": "RR",
})

_LOCAL_TEMPERATURE_MEAN_KEYS: Mapping[str, str] = MappingProxyType(
    {suffix: f"tyre_temp_{suffix}" for suffix in _LOCAL_WHEEL_SUFFIXES}
)

_LOCAL_TEMPERATURE_STD_KEYS: Mapping[str, str] = MappingProxyType(
    {
        suffix: f"{_LOCAL_TEMPERATURE_MEAN_KEYS[suffix]}_std"
        for suffix in _LOCAL_WHEEL_SUFFIXES
    }
)

_LOCAL_BRAKE_TEMPERATURE_MEAN_KEYS: Mapping[str, str] = MappingProxyType(
    {suffix: f"brake_temp_{suffix}" for suffix in _LOCAL_WHEEL_SUFFIXES}
)

_LOCAL_BRAKE_TEMPERATURE_STD_KEYS: Mapping[str, str] = MappingProxyType(
    {
        suffix: f"{_LOCAL_BRAKE_TEMPERATURE_MEAN_KEYS[suffix]}_std"
        for suffix in _LOCAL_WHEEL_SUFFIXES
    }
)

_LOCAL_PRESSURE_MEAN_KEYS: Mapping[str, str] = MappingProxyType(
    {suffix: f"tyre_pressure_{suffix}" for suffix in _LOCAL_WHEEL_SUFFIXES}
)

_LOCAL_PRESSURE_STD_KEYS: Mapping[str, str] = MappingProxyType(
    {
        suffix: f"{_LOCAL_PRESSURE_MEAN_KEYS[suffix]}_std"
        for suffix in _LOCAL_WHEEL_SUFFIXES
    }
)

WHEEL_SUFFIXES: tuple[str, ...] = _LOCAL_WHEEL_SUFFIXES
WHEEL_LABELS: Mapping[str, str] = _LOCAL_WHEEL_LABELS
TEMPERATURE_MEAN_KEYS: Mapping[str, str] = _LOCAL_TEMPERATURE_MEAN_KEYS
TEMPERATURE_STD_KEYS: Mapping[str, str] = _LOCAL_TEMPERATURE_STD_KEYS
BRAKE_TEMPERATURE_MEAN_KEYS: Mapping[str, str] = _LOCAL_BRAKE_TEMPERATURE_MEAN_KEYS
BRAKE_TEMPERATURE_STD_KEYS: Mapping[str, str] = _LOCAL_BRAKE_TEMPERATURE_STD_KEYS
PRESSURE_MEAN_KEYS: Mapping[str, str] = _LOCAL_PRESSURE_MEAN_KEYS
PRESSURE_STD_KEYS: Mapping[str, str] = _LOCAL_PRESSURE_STD_KEYS


if CANONICAL_REQUESTED:  # pragma: no cover - depends on optional package
    tnfr = import_tnfr()
    canonical_constants = import_module(f"{tnfr.__name__}.equations.constants")

    WHEEL_SUFFIXES = getattr(
        canonical_constants, "WHEEL_SUFFIXES", _LOCAL_WHEEL_SUFFIXES
    )
    WHEEL_LABELS = getattr(
        canonical_constants, "WHEEL_LABELS", _LOCAL_WHEEL_LABELS
    )
    TEMPERATURE_MEAN_KEYS = getattr(
        canonical_constants, "TEMPERATURE_MEAN_KEYS", _LOCAL_TEMPERATURE_MEAN_KEYS
    )
    TEMPERATURE_STD_KEYS = getattr(
        canonical_constants, "TEMPERATURE_STD_KEYS", _LOCAL_TEMPERATURE_STD_KEYS
    )
    BRAKE_TEMPERATURE_MEAN_KEYS = getattr(
        canonical_constants,
        "BRAKE_TEMPERATURE_MEAN_KEYS",
        _LOCAL_BRAKE_TEMPERATURE_MEAN_KEYS,
    )
    BRAKE_TEMPERATURE_STD_KEYS = getattr(
        canonical_constants,
        "BRAKE_TEMPERATURE_STD_KEYS",
        _LOCAL_BRAKE_TEMPERATURE_STD_KEYS,
    )
    PRESSURE_MEAN_KEYS = getattr(
        canonical_constants, "PRESSURE_MEAN_KEYS", _LOCAL_PRESSURE_MEAN_KEYS
    )
    PRESSURE_STD_KEYS = getattr(
        canonical_constants, "PRESSURE_STD_KEYS", _LOCAL_PRESSURE_STD_KEYS
    )


__all__ = [
    "WHEEL_SUFFIXES",
    "WHEEL_LABELS",
    "TEMPERATURE_MEAN_KEYS",
    "TEMPERATURE_STD_KEYS",
    "BRAKE_TEMPERATURE_MEAN_KEYS",
    "BRAKE_TEMPERATURE_STD_KEYS",
    "PRESSURE_MEAN_KEYS",
    "PRESSURE_STD_KEYS",
]
