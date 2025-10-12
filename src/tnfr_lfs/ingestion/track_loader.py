"""Helpers to resolve track manifests and recommendation profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore


from ..resources import data_root, modifiers_root
from ..common.immutables import _freeze_dict, _freeze_value


_DATA_ROOT = data_root()
_TRACKS_ROOT = _DATA_ROOT / "tracks"
_TRACK_PROFILES_ROOT = _DATA_ROOT / "track_profiles"
_MODIFIERS_ROOT = modifiers_root()


def _normalise_weights(
    payload: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Mapping[str, float]]:
    """Coerce nested weight mappings to floats."""

    if not payload:
        return MappingProxyType({})

    normalised: dict[str, Mapping[str, float]] = {}

    for phase, values in payload.items():
        phase_name = str(phase)
        numeric = {str(key): float(value) for key, value in values.items()}
        normalised[phase_name] = MappingProxyType(numeric)

    return MappingProxyType(normalised)


def _clone_weights(weights: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Create a mutable deep copy of the weights mapping."""

    return {phase: dict(values) for phase, values in weights.items()}


def _clone_hints(hints: Mapping[str, Any]) -> dict[str, Any]:
    """Create a mutable deep copy of the hints mapping."""

    cloned: dict[str, Any] = {}
    for key, value in hints.items():
        if isinstance(value, Mapping):
            cloned[str(key)] = _clone_hints(value)
        elif isinstance(value, list):
            cloned[str(key)] = [_clone_hints(item) if isinstance(item, Mapping) else item for item in value]
        else:
            cloned[str(key)] = value
    return cloned


def _apply_scale(
    weights: MutableMapping[str, MutableMapping[str, float]],
    scale: Mapping[str, Mapping[str, float]],
) -> None:
    """Mutate ``weights`` in-place applying the scale factors."""

    if not scale:
        return

    global_default = 1.0
    if "__default__" in scale:
        default_mapping = scale["__default__"]
        if isinstance(default_mapping, Mapping):
            global_default = float(default_mapping.get("__default__", 1.0))
        else:
            global_default = float(default_mapping)  # type: ignore[arg-type]

    for phase, phase_scale in scale.items():
        if phase == "__default__":
            continue
        weights.setdefault(phase, {})

    for phase, phase_weights in weights.items():
        phase_scale = scale.get(phase, {})
        phase_default = float(phase_scale.get("__default__", global_default))

        for key, value in list(phase_weights.items()):
            factor = phase_scale.get(key)
            if factor is None:
                factor = phase_default
            phase_weights[key] = float(value) * float(factor)

        for key, factor in phase_scale.items():
            if key == "__default__":
                continue
            if key not in phase_weights:
                phase_weights[key] = float(factor)


@dataclass(frozen=True, slots=True)
class TrackConfig:
    """Description of a specific Live for Speed layout."""

    identifier: str
    name: str
    length_km: float | None
    surface: str
    track_profile: str
    extras: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Track:
    """Track manifest containing the available layouts."""

    slug: str
    path: Path
    configs: Mapping[str, TrackConfig]


def load_track(track_slug: str, tracks_dir: str | Path | None = None) -> Track:
    """Load a track manifest and expose its ``[config.*]`` sections."""

    base = Path(tracks_dir) if tracks_dir is not None else _TRACKS_ROOT
    manifest = base / f"{track_slug.upper()}.toml"

    with manifest.open("rb") as buffer:
        payload = tomllib.load(buffer)

    configs_raw = payload.get("config", {})
    if not configs_raw:
        raise ValueError(f"Track manifest '{manifest.name}' does not define any [config.*] block")

    raw_configs: dict[str, dict[str, Any]] = {
        str(identifier): dict(values) for identifier, values in configs_raw.items()
    }

    configs: dict[str, TrackConfig] = {}
    resolving: set[str] = set()

    def resolve(identifier: str) -> TrackConfig:
        if identifier in configs:
            return configs[identifier]
        if identifier in resolving:
            raise ValueError(
                f"Track manifest '{manifest.name}' has a circular alias definition involving '{identifier}'"
            )
        if identifier not in raw_configs:
            raise ValueError(
                f"Track manifest '{manifest.name}' references undefined layout '{identifier}'"
            )

        resolving.add(identifier)
        try:
            values = raw_configs[identifier]
            alias_target_raw = values.get("alias_of")
            if alias_target_raw is not None:
                target_identifier = str(alias_target_raw)
                if target_identifier == identifier:
                    raise ValueError(
                        f"Track layout '{identifier}' from '{manifest.name}' cannot alias itself"
                    )
                disallowed_overrides = {
                    key for key in values if key not in {"alias_of", "name"}
                }
                if disallowed_overrides:
                    overrides = ", ".join(sorted(disallowed_overrides))
                    raise ValueError(
                        f"Track layout '{identifier}' from '{manifest.name}' declares alias_of "
                        f"but also overrides: {overrides}"
                    )
                base_config = resolve(target_identifier)
                name = str(values.get("name", base_config.name))
                config = TrackConfig(
                    identifier=identifier,
                    name=name,
                    length_km=base_config.length_km,
                    surface=base_config.surface,
                    track_profile=base_config.track_profile,
                    extras=base_config.extras,
                )
            else:
                name = str(values.get("name", identifier))
                length_raw = values.get("length_km")
                length = float(length_raw) if length_raw is not None else None
                surface_raw = values.get("surface")
                surface = str(surface_raw).lower() if surface_raw is not None else "unknown"
                if "track_profile" not in values or values["track_profile"] is None:
                    raise ValueError(
                        f"Track layout '{identifier}' from '{manifest.name}' is missing a track_profile"
                    )
                profile = str(values["track_profile"])

                extras = {
                    str(key): value
                    for key, value in values.items()
                    if key
                    not in {"name", "length_km", "surface", "track_profile", "alias_of"}
                }

                config = TrackConfig(
                    identifier=identifier,
                    name=name,
                    length_km=length,
                    surface=surface,
                    track_profile=profile,
                    extras=_freeze_dict(extras),
                )

            configs[identifier] = config
            return config
        finally:
            resolving.remove(identifier)

    for identifier in raw_configs:
        resolve(identifier)

    return Track(slug=track_slug.upper(), path=manifest, configs=MappingProxyType(configs))


def load_track_profiles(
    profiles_dir: str | Path | None = None,
) -> dict[str, Mapping[str, Any]]:
    """Load threshold track profiles keyed by ``meta.id`` or filename stem."""

    base = Path(profiles_dir) if profiles_dir is not None else _TRACK_PROFILES_ROOT
    profiles: dict[str, Mapping[str, Any]] = {}

    for manifest in sorted(base.glob("*.toml")):
        with manifest.open("rb") as buffer:
            payload = tomllib.load(buffer)

        meta_raw = payload.get("meta", {})
        identifier = str(meta_raw.get("id") or manifest.stem)

        if identifier in profiles:
            raise ValueError(f"Duplicated track profile identifier: {identifier}")

        weights_raw = payload.get("weights", {})
        hints_raw = payload.get("hints", {})

        profiles[identifier] = MappingProxyType(
            {
                "meta": _freeze_dict(meta_raw),
                "weights": _normalise_weights(weights_raw),
                "hints": _freeze_dict(hints_raw),
            }
        )

    return profiles


def load_modifiers(
    modifiers_dir: str | Path | None = None,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Load car/track modifiers keyed by ``(car_profile, track_profile)``."""

    base = Path(modifiers_dir) if modifiers_dir is not None else _MODIFIERS_ROOT
    modifiers: dict[tuple[str, str], Mapping[str, Any]] = {}

    for manifest in sorted(base.glob("*.toml")):
        with manifest.open("rb") as buffer:
            payload = tomllib.load(buffer)

        meta_raw = payload.get("meta", {})
        car_profile = str(meta_raw.get("car_group"))
        track_profile = str(meta_raw.get("base_profile"))

        if not car_profile or car_profile == "None":
            raise ValueError(
                f"Modifier '{manifest.name}' is missing 'meta.car_group'"
            )
        if not track_profile or track_profile == "None":
            raise ValueError(
                f"Modifier '{manifest.name}' is missing 'meta.base_profile'"
            )

        key = (car_profile, track_profile)
        if key in modifiers:
            raise ValueError(f"Duplicated modifier for combination {key!r}")

        scale_raw = payload.get("scale", {})
        weights_raw = scale_raw.get("weights", {}) if isinstance(scale_raw, Mapping) else {}

        modifiers[key] = MappingProxyType(
            {
                "meta": _freeze_dict(meta_raw),
                "scale": MappingProxyType({"weights": _normalise_weights(weights_raw)}),
                "hints": _freeze_dict(payload.get("hints", {})),
            }
        )

    return modifiers


def assemble_session_weights(
    car_profile: str,
    track_profile: str,
    *,
    track_profiles: Mapping[str, Mapping[str, Any]],
    modifiers: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    """Combine a car profile and track profile into session phase weights."""

    try:
        profile = track_profiles[track_profile]
    except KeyError as exc:
        raise KeyError(f"Unknown track profile '{track_profile}'") from exc

    weights = _clone_weights(profile.get("weights", MappingProxyType({})))
    hints = _clone_hints(profile.get("hints", MappingProxyType({})))

    modifier_payload = (modifiers or {}).get((car_profile, track_profile)) if modifiers else None
    if modifier_payload:
        scale_payload = modifier_payload.get("scale", {})
        scale_weights = scale_payload.get("weights", {}) if isinstance(scale_payload, Mapping) else {}
        _apply_scale(weights, scale_weights)

        hints_update = modifier_payload.get("hints", {})
        if hints_update:
            cloned_update = _clone_hints(hints_update)
            hints.update(cloned_update)

    frozen_weights = MappingProxyType(
        {phase: MappingProxyType({key: float(value) for key, value in values.items()}) for phase, values in weights.items()}
    )
    frozen_hints = _freeze_dict(hints)

    return MappingProxyType({"weights": frozen_weights, "hints": frozen_hints})


__all__ = [
    "Track",
    "TrackConfig",
    "load_track",
    "load_track_profiles",
    "load_modifiers",
    "assemble_session_weights",
]
