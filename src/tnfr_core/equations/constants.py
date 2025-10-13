"""Centralised constant definitions shared across TNFR × LFS modules."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


WHEEL_SUFFIXES: tuple[str, ...] = ("fl", "fr", "rl", "rr")

WHEEL_LABELS: Mapping[str, str] = MappingProxyType({
    "fl": "FL",
    "fr": "FR",
    "rl": "RL",
    "rr": "RR",
})

TEMPERATURE_MEAN_KEYS: Mapping[str, str] = MappingProxyType(
    {suffix: f"tyre_temp_{suffix}" for suffix in WHEEL_SUFFIXES}
)

TEMPERATURE_STD_KEYS: Mapping[str, str] = MappingProxyType(
    {suffix: f"{TEMPERATURE_MEAN_KEYS[suffix]}_std" for suffix in WHEEL_SUFFIXES}
)

BRAKE_TEMPERATURE_MEAN_KEYS: Mapping[str, str] = MappingProxyType(
    {suffix: f"brake_temp_{suffix}" for suffix in WHEEL_SUFFIXES}
)

BRAKE_TEMPERATURE_STD_KEYS: Mapping[str, str] = MappingProxyType(
    {suffix: f"{BRAKE_TEMPERATURE_MEAN_KEYS[suffix]}_std" for suffix in WHEEL_SUFFIXES}
)

PRESSURE_MEAN_KEYS: Mapping[str, str] = MappingProxyType(
    {suffix: f"tyre_pressure_{suffix}" for suffix in WHEEL_SUFFIXES}
)

PRESSURE_STD_KEYS: Mapping[str, str] = MappingProxyType(
    {suffix: f"{PRESSURE_MEAN_KEYS[suffix]}_std" for suffix in WHEEL_SUFFIXES}
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

