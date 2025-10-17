# `tnfr_core.operators.cache_settings` module
Compatibility layer for runtime cache configuration models.

## Classes
### `CacheOptions`
Immutable cache configuration parsed from TOML sources.

#### Methods
- `from_config(cls, config: Mapping[str, Any] | None = None) -> 'CacheOptions'`
  - Coerce raw configuration mappings into cache options.

The helper accepts mappings produced either by the new
``[performance]`` table or legacy ``[cache]`` payloads and normalises
them into a :class:`CacheOptions` instance.  Unknown values fall back to
the existing defaults, and booleans/integers are coerced to sensible
values mirroring the previous parsing logic.
- `with_defaults(self) -> 'CacheOptions'`
  - Return an instance with normalised field values.
- `to_performance_config(self) -> dict[str, int | bool]`
  - Serialise the cache options into a ``[performance]`` mapping.
- `cache_enabled(self) -> bool`
  - Backward compatible alias describing whether caches are active.
- `max_cache_size(self) -> int`
  - Largest cache size configured for runtime helpers.

## Functions
- `resolve_recommender_cache_size(cache_size: int | None) -> int`
  - Normalise cache sizes used by recommendation helpers.

## Attributes
- `DEFAULT_RECOMMENDER_CACHE_SIZE = 32`
- `DEFAULT_DYNAMIC_CACHE_SIZE = 256`
- `LEGACY_TELEMETRY_CACHE_KEY = 'telemetry_cache_size'`

