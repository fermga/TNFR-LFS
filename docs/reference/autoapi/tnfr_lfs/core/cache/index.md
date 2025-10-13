# `tnfr_core.cache` module
Caching helpers for ΔNFR maps and dynamic ν_f multipliers.

## Classes
### `LRUCache` (Generic[_K, _V])
Public wrapper around :class:`_LRUCache` supporting ``maxsize >= 0``.

#### Methods
- `maxsize(self) -> int`
  - Return the configured capacity for the cache.
- `get_or_create(self, key: _K, factory: Callable[[], _V]) -> _V`
  - Return cached value for ``key`` or materialise it via ``factory``.
- `invalidate(self, predicate: Callable[[_K], bool]) -> None`
  - Remove cached entries matching ``predicate`` when caching is active.
- `clear(self) -> None`
  - Drop all cached entries when caching is active.

## Functions
- `cached_delta_nfr_map(record: _T, factory: Callable[[], Mapping[str, float]]) -> Mapping[str, float]`
  - Return a cached ΔNFR-by-node map for *record*.
- `invalidate_delta_record(record: _T) -> None`
  - Invalidate cached ΔNFR entries that reference *record*.
- `clear_delta_cache() -> None`
  - Clear all cached ΔNFR entries.
- `cached_dynamic_multipliers(car_model: str | None, history: Sequence[_T], factory: Callable[[], Tuple[Mapping[str, float], float]]) -> Tuple[Mapping[str, float], float]`
  - Return cached dynamic ν_f multipliers for the supplied *history*.
- `invalidate_dynamic_record(record: _T) -> None`
  - Drop cached ν_f multipliers that are older than *record*.
- `clear_dynamic_cache() -> None`
  - Clear the dynamic ν_f multiplier cache.
- `configure_cache(*, enable_delta_cache: bool | None = None, nu_f_cache_size: int | None = None) -> None`
  - Configure cache toggles and capacities used by EPI helpers.
- `configure_cache_from_options(options: CacheOptions) -> None`
  - Normalise and apply cache settings declared via :class:`CacheOptions`.
- `should_use_delta_cache(cache_options: CacheOptions | None) -> bool`
  - Return ``True`` when ΔNFR caching should be used.

The helper honours per-call overrides supplied via :class:`CacheOptions`
instances, falling back to the global cache toggle when no override is
provided.
- `should_use_dynamic_cache(cache_options: CacheOptions | None) -> bool`
  - Return ``True`` when ν_f multiplier caching should be used.

When explicit cache options are supplied they take priority, otherwise the
helper defers to the module-level cache flag.
- `delta_cache_enabled() -> bool`
  - Return ``True`` when ΔNFR caching is active.
- `dynamic_cache_enabled() -> bool`
  - Return ``True`` when ν_f multipliers caching is active.

