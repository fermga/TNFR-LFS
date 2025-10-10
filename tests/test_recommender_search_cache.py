from __future__ import annotations

import tnfr_lfs.recommender.search as search_module
from tnfr_lfs.core.cache_settings import (
    DEFAULT_RECOMMENDER_CACHE_SIZE,
    resolve_recommender_cache_size,
)


def test_resolve_recommender_cache_size_normalises_values() -> None:
    assert resolve_recommender_cache_size(None) == DEFAULT_RECOMMENDER_CACHE_SIZE
    assert resolve_recommender_cache_size(7) == 7
    assert resolve_recommender_cache_size(-3) == 0


def test_setup_planner_uses_cache_resolver(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _resolver(value):  # type: ignore[no-untyped-def]
        captured["value"] = value
        return 42

    monkeypatch.setattr(search_module, "resolve_recommender_cache_size", _resolver)

    planner = search_module.SetupPlanner(cache_size=17)

    assert captured["value"] == 17
    assert planner.cache_size == 42


def test_sweep_candidates_uses_cache_resolver(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _resolver(value):  # type: ignore[no-untyped-def]
        captured["value"] = value
        return 3

    monkeypatch.setattr(search_module, "resolve_recommender_cache_size", _resolver)

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
    )

    assert result == ["ok"]
    assert captured["value"] == -11
    assert captured["maxsize"] == 3
