"""CLI-related test helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import pytest

from tnfr_lfs.cli import workflows as workflows_module


@contextmanager
def instrument_prepare_pack_context(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[dict[str, int]]:
    """Patch ``_prepare_pack_context`` to track how often it is invoked."""

    calls = {"count": 0}
    original = workflows_module._prepare_pack_context

    def _instrumented_prepare(
        namespace: Any,
        config: Mapping[str, Any] | None,
        *,
        car_model: str,
    ) -> Any:
        calls["count"] += 1
        return original(namespace, config, car_model=car_model)

    monkeypatch.setattr(workflows_module, "_prepare_pack_context", _instrumented_prepare)
    try:
        yield calls
    finally:
        setattr(workflows_module, "_prepare_pack_context", original)
