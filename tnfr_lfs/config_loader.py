"""Utility helpers to load bundled TNFR × LFS configuration packs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import tomllib


_PACKAGE_ROOT = Path(__file__).resolve().parent
_DATA_ROOT = _PACKAGE_ROOT.parent / "data"


def _freeze_value(value: Any) -> Any:
    """Recursively convert mappings to immutable views and lists to tuples."""

    if isinstance(value, dict):
        return MappingProxyType({str(k): _freeze_value(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_dict(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an immutable view for a mapping, freezing nested structures."""

    return MappingProxyType({str(key): _freeze_value(value) for key, value in payload.items()})


@dataclass(frozen=True, slots=True)
class Car:
    """Car metadata that links vehicle abbreviations to tuning profiles."""

    abbrev: str
    name: str
    license: str
    engine_layout: str
    drive: str
    weight_kg: int
    wheel_rotation_group_deg: int
    profile: str
    lfs_class: str | None = None


@dataclass(frozen=True, slots=True)
class Profile:
    """Top-level structures declared in a TNFR tuning profile."""

    meta: Mapping[str, Any]
    targets: Mapping[str, Mapping[str, Any]]
    policy: Mapping[str, Mapping[str, Any]]
    recommender: Mapping[str, Mapping[str, Any]]


def load_cars(cars_dir: str | Path | None = None) -> dict[str, Car]:
    """Load all car definitions, keyed by their abbreviation.

    Parameters
    ----------
    cars_dir:
        Optional path to the directory containing ``*.toml`` car manifests.
        When omitted, the bundled ``data/cars`` directory inside the project
        repository is used.
    """

    path = Path(cars_dir) if cars_dir is not None else _DATA_ROOT / "cars"
    cars: dict[str, Car] = {}

    for manifest in sorted(path.glob("*.toml")):
        with manifest.open("rb") as buffer:
            payload = tomllib.load(buffer)

        lfs_class = payload.get("lfs_class")

        car = Car(
            abbrev=str(payload["abbrev"]),
            name=str(payload["name"]),
            license=str(payload["license"]),
            engine_layout=str(payload["engine_layout"]),
            drive=str(payload["drive"]),
            weight_kg=int(payload["weight_kg"]),
            wheel_rotation_group_deg=int(payload["wheel_rotation_group_deg"]),
            profile=str(payload["profile"]),
            lfs_class=str(lfs_class) if lfs_class is not None else None,
        )

        if car.abbrev in cars:
            raise ValueError(f"Duplicated car abbreviation: {car.abbrev}")
        cars[car.abbrev] = car

    return cars


def load_profiles(profiles_dir: str | Path | None = None) -> dict[str, Profile]:
    """Load profile definitions, keyed by ``meta.id`` or filename fallback."""

    path = Path(profiles_dir) if profiles_dir is not None else _DATA_ROOT / "profiles"
    profiles: dict[str, Profile] = {}

    for manifest in sorted(path.glob("*.toml")):
        with manifest.open("rb") as buffer:
            payload = tomllib.load(buffer)

        meta_raw = payload.get("meta", {})
        identifier = str(meta_raw.get("id") or manifest.stem)

        if identifier in profiles:
            raise ValueError(f"Duplicated profile identifier: {identifier}")

        profile = Profile(
            meta=_freeze_dict(meta_raw) if meta_raw else MappingProxyType({}),
            targets=_freeze_dict(payload.get("targets", {})),
            policy=_freeze_dict(payload.get("policy", {})),
            recommender=_freeze_dict(payload.get("recommender", {})),
        )

        profiles[identifier] = profile

    return profiles


def resolve_targets(
    car_abbrev: str,
    cars: Mapping[str, Car],
    profiles: Mapping[str, Profile],
) -> Mapping[str, Mapping[str, Any]]:
    """Resolve the profile referenced by ``car_abbrev``.

    Parameters
    ----------
    car_abbrev:
        Abbreviation of the car to look up.
    cars:
        Mapping containing the loaded :class:`Car` instances.
    profiles:
        Mapping containing the loaded :class:`Profile` instances.
    """

    car = cars[car_abbrev]

    try:
        profile = profiles[car.profile]
    except KeyError as exc:
        raise KeyError(
            f"Car '{car.abbrev}' references unknown profile '{car.profile}'"
        ) from exc

    return MappingProxyType(
        {
            "meta": profile.meta,
            "targets": profile.targets,
            "policy": profile.policy,
            "recommender": profile.recommender,
        }
    )


def example_pipeline(
    car_abbrev: str,
    *,
    data_root: str | Path | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    """Convenience helper for quick experiments in notebooks and docs."""

    base = Path(data_root) if data_root is not None else _DATA_ROOT
    cars = load_cars(base / "cars") if data_root is not None else load_cars()
    profiles = (
        load_profiles(base / "profiles") if data_root is not None else load_profiles()
    )

    return resolve_targets(car_abbrev, cars, profiles)


__all__ = [
    "Car",
    "Profile",
    "load_cars",
    "load_profiles",
    "resolve_targets",
    "example_pipeline",
]
