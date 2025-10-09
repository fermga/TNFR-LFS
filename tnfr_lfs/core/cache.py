"""Caching helpers for ΔNFR maps and dynamic ν_f multipliers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Generic, Hashable, Mapping, Sequence, Tuple, TypeVar

_T = TypeVar("_T")
_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")


class _LRUCache(Generic[_K, _V]):
    """Minimal LRU cache with targeted invalidation support."""

    __slots__ = ("_maxsize", "_data")

    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self._maxsize = int(maxsize)
        self._data: "OrderedDict[_K, _V]" = OrderedDict()

    def get_or_create(self, key: _K, factory: Callable[[], _V]) -> _V:
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
        for candidate in list(self._data.keys()):
            if predicate(candidate):
                self._data.pop(candidate, None)

    def clear(self) -> None:
        self._data.clear()


_DELTA_NFR_CACHE: _LRUCache[Tuple[int, int | None], Mapping[str, float]] = _LRUCache(maxsize=1024)
_DYNAMIC_MULTIPLIER_CACHE: _LRUCache[
    Tuple[str | None, int, float | None], Tuple[Mapping[str, float], float]
] = _LRUCache(maxsize=256)


def cached_delta_nfr_map(record: _T, factory: Callable[[], Mapping[str, float]]) -> Mapping[str, float]:
    """Return a cached ΔNFR-by-node map for *record*."""

    reference = getattr(record, "reference", None)
    reference_id: int | None = id(reference) if reference is not None else None
    key = (id(record), reference_id)
    value = _DELTA_NFR_CACHE.get_or_create(key, lambda: dict(factory()))
    return dict(value)


def invalidate_delta_record(record: _T) -> None:
    """Invalidate cached ΔNFR entries that reference *record*."""

    record_id = id(record)

    def _should_remove(key: Tuple[int, int | None]) -> bool:
        current_id, reference_id = key
        return current_id == record_id or reference_id == record_id

    _DELTA_NFR_CACHE.invalidate(_should_remove)


def clear_delta_cache() -> None:
    """Clear all cached ΔNFR entries."""

    _DELTA_NFR_CACHE.clear()


def cached_dynamic_multipliers(
    car_model: str | None,
    history: Sequence[_T],
    factory: Callable[[], Tuple[Mapping[str, float], float]],
) -> Tuple[Mapping[str, float], float]:
    """Return cached dynamic ν_f multipliers for the supplied *history*."""

    timestamp = getattr(history[-1], "timestamp", None) if history else None
    key = (car_model, len(history), timestamp)
    value = _DYNAMIC_MULTIPLIER_CACHE.get_or_create(
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

    threshold = getattr(record, "timestamp", None)

    def _should_remove(key: Tuple[str | None, int, float | None]) -> bool:
        _, _, cached_timestamp = key
        if cached_timestamp is None or threshold is None:
            return False
        return cached_timestamp <= threshold

    _DYNAMIC_MULTIPLIER_CACHE.invalidate(_should_remove)


def clear_dynamic_cache() -> None:
    """Clear the dynamic ν_f multiplier cache."""

    _DYNAMIC_MULTIPLIER_CACHE.clear()


__all__ = [
    "cached_delta_nfr_map",
    "invalidate_delta_record",
    "clear_delta_cache",
    "cached_dynamic_multipliers",
    "invalidate_dynamic_record",
    "clear_dynamic_cache",
]
