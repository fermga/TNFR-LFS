"""Caching helpers for ΔNFR maps and dynamic ν_f multipliers."""

from __future__ import annotations

from collections import OrderedDict
import threading
from typing import Callable, Generic, Hashable, Mapping, Sequence, Tuple, TypeVar

_T = TypeVar("_T")
_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")


class _LRUCache(Generic[_K, _V]):
    """Minimal LRU cache with targeted invalidation support."""

    __slots__ = ("_maxsize", "_data", "_lock")

    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self._maxsize = int(maxsize)
        self._data: "OrderedDict[_K, _V]" = OrderedDict()
        self._lock = threading.RLock()

    def get_or_create(self, key: _K, factory: Callable[[], _V]) -> _V:
        with self._lock:
            try:
                value = self._data.pop(key)
            except KeyError:
                value = factory()
            else:
                self._data[key] = value
                return value
            self._data[key] = value
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)
            return value

    def invalidate(self, predicate: Callable[[_K], bool]) -> None:
        with self._lock:
            for candidate in list(self._data.keys()):
                if predicate(candidate):
                    self._data.pop(candidate, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_DEFAULT_DELTA_CACHE_SIZE = 1024

_ENABLE_DELTA_CACHE = True
_DELTA_NFR_CACHE: _LRUCache[Tuple[int, int | None], Mapping[str, float]] | None = _LRUCache(
    maxsize=_DEFAULT_DELTA_CACHE_SIZE
)
_DYNAMIC_MULTIPLIER_CACHE: _LRUCache[
    Tuple[str | None, int, float | None], Tuple[Mapping[str, float], float]
] | None = _LRUCache(maxsize=256)


def cached_delta_nfr_map(record: _T, factory: Callable[[], Mapping[str, float]]) -> Mapping[str, float]:
    """Return a cached ΔNFR-by-node map for *record*."""

    if not delta_cache_enabled():
        return dict(factory())

    reference = getattr(record, "reference", None)
    reference_id: int | None = id(reference) if reference is not None else None
    key = (id(record), reference_id)
    assert _DELTA_NFR_CACHE is not None
    value = _DELTA_NFR_CACHE.get_or_create(key, lambda: dict(factory()))
    return dict(value)


def invalidate_delta_record(record: _T) -> None:
    """Invalidate cached ΔNFR entries that reference *record*."""

    if _DELTA_NFR_CACHE is None:
        return
    record_id = id(record)

    def _should_remove(key: Tuple[int, int | None]) -> bool:
        current_id, reference_id = key
        return current_id == record_id or reference_id == record_id

    _DELTA_NFR_CACHE.invalidate(_should_remove)


def clear_delta_cache() -> None:
    """Clear all cached ΔNFR entries."""

    if _DELTA_NFR_CACHE is not None:
        _DELTA_NFR_CACHE.clear()


def cached_dynamic_multipliers(
    car_model: str | None,
    history: Sequence[_T],
    factory: Callable[[], Tuple[Mapping[str, float], float]],
) -> Tuple[Mapping[str, float], float]:
    """Return cached dynamic ν_f multipliers for the supplied *history*."""

    timestamp = getattr(history[-1], "timestamp", None) if history else None
    key = (car_model, len(history), timestamp)
    cache = _DYNAMIC_MULTIPLIER_CACHE
    if cache is None:
        return _prepare_dynamic_entry(factory)
    value = cache.get_or_create(
        key,
        lambda: _prepare_dynamic_entry(factory),
    )
    multipliers, frequency = value
    return dict(multipliers), frequency


def _prepare_dynamic_entry(
    factory: Callable[[], Tuple[Mapping[str, float], float]]
) -> Tuple[Mapping[str, float], float]:
    multipliers, frequency = factory()
    return dict(multipliers), frequency


def invalidate_dynamic_record(record: _T) -> None:
    """Drop cached ν_f multipliers that are older than *record*."""

    if _DYNAMIC_MULTIPLIER_CACHE is None:
        return
    threshold = getattr(record, "timestamp", None)

    def _should_remove(key: Tuple[str | None, int, float | None]) -> bool:
        _, _, cached_timestamp = key
        if cached_timestamp is None or threshold is None:
            return False
        return cached_timestamp <= threshold

    _DYNAMIC_MULTIPLIER_CACHE.invalidate(_should_remove)


def clear_dynamic_cache() -> None:
    """Clear the dynamic ν_f multiplier cache."""

    if _DYNAMIC_MULTIPLIER_CACHE is not None:
        _DYNAMIC_MULTIPLIER_CACHE.clear()


def configure_cache(*, enable_delta_cache: bool | None = None, nu_f_cache_size: int | None = None) -> None:
    """Configure cache toggles and capacities used by EPI helpers."""

    global _ENABLE_DELTA_CACHE, _DELTA_NFR_CACHE, _DYNAMIC_MULTIPLIER_CACHE

    if enable_delta_cache is not None:
        enable = bool(enable_delta_cache)
        cache = _DELTA_NFR_CACHE
        if not enable:
            if cache is not None:
                with cache._lock:
                    cache.clear()
                    _DELTA_NFR_CACHE = None
        elif cache is None:
            _DELTA_NFR_CACHE = _LRUCache(maxsize=_DEFAULT_DELTA_CACHE_SIZE)
        _ENABLE_DELTA_CACHE = enable

    if nu_f_cache_size is not None:
        size = int(nu_f_cache_size)
        if size <= 0:
            cache = _DYNAMIC_MULTIPLIER_CACHE
            if cache is not None:
                with cache._lock:
                    cache.clear()
                    _DYNAMIC_MULTIPLIER_CACHE = None
        else:
            cache = _DYNAMIC_MULTIPLIER_CACHE
            if cache is None or cache._maxsize != size:
                _DYNAMIC_MULTIPLIER_CACHE = _LRUCache(maxsize=size)


def delta_cache_enabled() -> bool:
    """Return ``True`` when ΔNFR caching is active."""

    return _ENABLE_DELTA_CACHE and _DELTA_NFR_CACHE is not None


def dynamic_cache_enabled() -> bool:
    """Return ``True`` when ν_f multipliers caching is active."""

    return _DYNAMIC_MULTIPLIER_CACHE is not None


__all__ = [
    "cached_delta_nfr_map",
    "invalidate_delta_record",
    "clear_delta_cache",
    "cached_dynamic_multipliers",
    "invalidate_dynamic_record",
    "clear_dynamic_cache",
    "configure_cache",
    "delta_cache_enabled",
    "dynamic_cache_enabled",
]
