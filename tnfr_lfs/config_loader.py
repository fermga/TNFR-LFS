"""Utility helpers to load bundled TNFR × LFS configuration packs."""

from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore


from ._pack_resources import data_root
from .cache_settings import CacheOptions, DEFAULT_RECOMMENDER_CACHE_SIZE
from .utils.immutables import _freeze_dict, _freeze_value


_DATA_ROOT = data_root()
_LFS_CLASS_OVERRIDES_CACHE: dict[Path, Mapping[str, Mapping[str, Any]]] = {}


def _load_pack_cache_defaults(pack_root: Path | None = None) -> Mapping[str, Any]:
    """Load cache defaults from the canonical ``config/global.toml`` location."""

    if pack_root is None:
        base_root = data_root().parent
    else:
        base_root = Path(pack_root).expanduser()

    candidate = base_root / "config" / "global.toml"
    if not candidate.exists():
        return MappingProxyType({})

    with candidate.open("rb") as buffer:
        payload = tomllib.load(buffer)

    cache_section = payload.get("cache")
    if not isinstance(cache_section, ABCMapping):
        return MappingProxyType({})

    defaults: dict[str, Any] = {}

    for key in ("enable_delta_cache", "nu_f_cache_size", "recommender_cache_size"):
        if key in cache_section:
            defaults[key] = cache_section.get(key)

    telemetry_section = cache_section.get("telemetry")
    if isinstance(telemetry_section, ABCMapping):
        defaults["telemetry"] = MappingProxyType(
            {str(key): value for key, value in telemetry_section.items()}
        )

    return MappingProxyType(defaults)


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return fallback


def _coerce_int(value: Any, fallback: int, *, minimum: int = 0) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return fallback
    if numeric < minimum:
        return minimum
    return numeric


def _deep_merge(
    base: ABCMapping[str, Any], overlay: ABCMapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` returning a new mapping."""

    merged: dict[str, Any] = {str(key): value for key, value in base.items()}

    for key, overlay_value in overlay.items():
        key_str = str(key)
        base_value = merged.get(key_str)
        if isinstance(base_value, ABCMapping) and isinstance(overlay_value, ABCMapping):
            merged[key_str] = _deep_merge(base_value, overlay_value)
        else:
            merged[key_str] = overlay_value

    return merged


def _load_lfs_class_overrides(
    overrides_path: str | Path | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    """Return cached LFS class overrides, loading them lazily when required."""

    if overrides_path is None:
        candidate = _DATA_ROOT / "lfs_class_overrides.toml"
    else:
        candidate = Path(overrides_path)

    candidate = candidate.expanduser()
    cache_key = candidate.resolve(strict=False)

    cached = _LFS_CLASS_OVERRIDES_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not candidate.exists():
        overrides: Mapping[str, Mapping[str, Any]] = MappingProxyType({})
        _LFS_CLASS_OVERRIDES_CACHE[cache_key] = overrides
        return overrides

    with candidate.open("rb") as buffer:
        payload = tomllib.load(buffer)

    overrides_dict: dict[str, Mapping[str, Any]] = {}

    for class_name, section in payload.items():
        if not isinstance(section, ABCMapping):
            continue
        overrides_section = section.get("overrides")
        if isinstance(overrides_section, ABCMapping):
            overrides_dict[str(class_name)] = _freeze_value(overrides_section)

    overrides = MappingProxyType(overrides_dict)
    _LFS_CLASS_OVERRIDES_CACHE[cache_key] = overrides
    return overrides


def load_lfs_class_overrides(
    overrides_path: str | Path | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    """Public wrapper to access the cached LFS class overrides mapping."""

    return _load_lfs_class_overrides(overrides_path)


def parse_cache_options(
    config: Mapping[str, Any] | None = None,
    *,
    pack_root: str | Path | None = None,
) -> CacheOptions:
    """Normalise cache configuration from CLI and pack TOML payloads."""

    pack_root_override: Path | None = None
    if pack_root is not None:
        pack_root_override = Path(pack_root).expanduser()
    elif config is not None:
        paths_cfg = config.get("paths")
        if isinstance(paths_cfg, ABCMapping):
            pack_root_value = paths_cfg.get("pack_root")
            if isinstance(pack_root_value, str) and pack_root_value.strip():
                pack_root_override = Path(pack_root_value).expanduser()

    overrides: Mapping[str, Any]
    if config is None:
        overrides = MappingProxyType({})
    else:
        candidate = config.get("cache")
        if isinstance(candidate, ABCMapping):
            overrides = candidate
        else:
            overrides = MappingProxyType({})

    defaults = _load_pack_cache_defaults(pack_root_override)
    if defaults:
        payload = _deep_merge(defaults, overrides)
    else:
        payload = dict(overrides)

    telemetry_raw = payload.get("telemetry")
    telemetry_cfg = telemetry_raw if isinstance(telemetry_raw, ABCMapping) else {}

    options = CacheOptions(
        enable_delta_cache=_coerce_bool(payload.get("enable_delta_cache"), True),
        nu_f_cache_size=_coerce_int(payload.get("nu_f_cache_size"), 256, minimum=0),
        telemetry_cache_size=_coerce_int(
            telemetry_cfg.get("telemetry_cache_size"), 1, minimum=0
        ),
        recommender_cache_size=_coerce_int(
            payload.get("recommender_cache_size"),
            DEFAULT_RECOMMENDER_CACHE_SIZE,
            minimum=0,
        ),
    )
    return options.with_defaults()


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
    *,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
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

    payload = {
        "meta": profile.meta,
        "targets": profile.targets,
        "policy": profile.policy,
        "recommender": profile.recommender,
    }

    if overrides is None:
        overrides = _load_lfs_class_overrides()

    if car.lfs_class is None:
        return MappingProxyType(payload)

    class_overrides = overrides.get(car.lfs_class)
    if not class_overrides:
        return MappingProxyType(payload)

    merged = _deep_merge(payload, class_overrides)
    return _freeze_dict(merged)


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
    overrides = (
        load_lfs_class_overrides(base / "lfs_class_overrides.toml")
        if data_root is not None
        else load_lfs_class_overrides()
    )

    return resolve_targets(car_abbrev, cars, profiles, overrides=overrides)


__all__ = [
    "Car",
    "Profile",
    "load_cars",
    "load_profiles",
    "resolve_targets",
    "example_pipeline",
    "load_lfs_class_overrides",
    "CacheOptions",
    "parse_cache_options",
]
