# `tnfr_lfs.telemetry.config_loader` module
Utility helpers to load bundled TNFR × LFS configuration packs.

## Classes
### `Car`
Car metadata that links vehicle abbreviations to tuning profiles.

### `Profile`
Top-level structures declared in a TNFR tuning profile.

## Functions
- `load_lfs_class_overrides(overrides_path: str | Path | None = None) -> Mapping[str, Mapping[str, Any]]`
  - Public wrapper to access the cached LFS class overrides mapping.
- `parse_cache_options(config: Mapping[str, Any] | None = None, *, pack_root: str | Path | None = None) -> CacheOptions`
  - Normalise cache configuration from CLI and pack TOML payloads.
- `load_cars(cars_dir: str | Path | None = None) -> dict[str, Car]`
  - Load all car definitions, keyed by their abbreviation.

Parameters
----------
cars_dir:
    Optional path to the directory containing ``*.toml`` car manifests.
    When omitted, the bundled ``data/cars`` directory inside the project
    repository is used.
- `load_profiles(profiles_dir: str | Path | None = None) -> dict[str, Profile]`
  - Load profile definitions, keyed by ``meta.id`` or filename fallback.
- `resolve_targets(car_abbrev: str, cars: Mapping[str, Car], profiles: Mapping[str, Profile], *, overrides: Mapping[str, Mapping[str, Any]] | None = None) -> Mapping[str, Mapping[str, Any]]`
  - Resolve the profile referenced by ``car_abbrev``.

Parameters
----------
car_abbrev:
    Abbreviation of the car to look up.
cars:
    Mapping containing the loaded :class:`Car` instances.
profiles:
    Mapping containing the loaded :class:`Profile` instances.
- `example_pipeline(car_abbrev: str, *, data_root: str | Path | None = None) -> Mapping[str, Mapping[str, Any]]`
  - Convenience helper for quick experiments in notebooks and docs.

