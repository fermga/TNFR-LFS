from __future__ import annotations

import pytest

import tnfr_lfs.recommender.search as search_module
from tnfr_lfs.core.cache_settings import (
    CacheOptions,
    DEFAULT_RECOMMENDER_CACHE_SIZE,
    resolve_recommender_cache_size,
)


def test_resolve_recommender_cache_size_normalises_values() -> None:
    assert resolve_recommender_cache_size(None) == DEFAULT_RECOMMENDER_CACHE_SIZE
    assert resolve_recommender_cache_size(7) == 7
    assert resolve_recommender_cache_size(-3) == 0


def test_planner_cache_helper_prefers_options() -> None:
    options = CacheOptions(recommender_cache_size=17)

    assert (
        search_module._resolve_planner_cache_size(
            cache_options=options, cache_size=3
        )
        == 17
    )


def test_planner_cache_helper_supports_legacy_size() -> None:
    assert search_module._resolve_planner_cache_size(cache_size=-5) == 0


@pytest.mark.parametrize(
    "cache_options_value, cache_size_value, resolver_return",
    [
        pytest.param(
            CacheOptions(recommender_cache_size=64),
            None,
            42,
            id="options",
        ),
        pytest.param(
            None,
            21,
            13,
            id="legacy-size",
        ),
    ],
)
def test_setup_planner_cache_resolution(
    monkeypatch, cache_options_value, cache_size_value, resolver_return
) -> None:
    captured: dict[str, object] = {}

    def _resolver(cache_options=None, cache_size=None):  # type: ignore[no-untyped-def]
        captured["cache_options"] = cache_options
        captured["cache_size"] = cache_size
        return resolver_return

    monkeypatch.setattr(search_module, "_resolve_planner_cache_size", _resolver)

    init_kwargs: dict[str, object] = {}
    if cache_options_value is not None:
        init_kwargs["cache_options"] = cache_options_value
    if cache_size_value is not None:
        init_kwargs["cache_size"] = cache_size_value

    planner = search_module.SetupPlanner(**init_kwargs)

    assert captured["cache_options"] is cache_options_value
    assert captured["cache_size"] == cache_size_value
    assert planner.cache_size == resolver_return


def test_sweep_candidates_uses_cache_helper(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _resolver(cache_options=None, cache_size=None):  # type: ignore[no-untyped-def]
        captured["cache_options"] = cache_options
        captured["cache_size"] = cache_size
        return 3

    monkeypatch.setattr(search_module, "_resolve_planner_cache_size", _resolver)

    class DummyCache:
        def __init__(self, maxsize: int) -> None:
            captured["maxsize"] = maxsize
            self._store: dict[object, object] = {}

        def get_or_create(self, key, factory):  # type: ignore[no-untyped-def]
            if key not in self._store:
                self._store[key] = factory()
            return self._store[key]

    monkeypatch.setattr(search_module, "LRUCache", DummyCache)
    monkeypatch.setattr(search_module, "evaluate_candidate", lambda *args, **kwargs: "ok")

    class DummySpace:
        def clamp(self, vector):  # type: ignore[no-untyped-def]
            return dict(vector)

    result = search_module.sweep_candidates(
        DummySpace(),
        {"alpha": 1.0},
        baseline=(),
        candidates=[{"alpha": 1.5}],
        cache_size=-11,
        cache_options=CacheOptions(recommender_cache_size=9),
    )

    assert result == ["ok"]
    assert isinstance(captured["cache_options"], CacheOptions)
    assert captured["cache_size"] == -11
    assert captured["maxsize"] == 3
