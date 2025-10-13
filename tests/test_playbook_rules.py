from __future__ import annotations

import io
from types import TracebackType
from typing import Any, Mapping

import pytest

from tnfr_lfs.cli import workflows as workflows_module
from tnfr_lfs.cli.workflows import (
    _augment_session_with_playbook,
    _resolve_playbook_suggestions,
)
from tnfr_lfs.telemetry.offline import load_playbook


_PLAYBOOK_RESOURCE_TARGET = f"{load_playbook.__module__}.resources.open_binary"


def _playbook() -> Mapping[str, tuple[str, ...]]:
    return {
        "delta_surplus": ("Trim delta", "Balance exit"),
        "delta_deficit": ("Reinforce load",),
        "sense_index_low": ("Driver consistency",),
        "coherence_low": ("Synchronise phases",),
        "aero_balance_low": ("Aero balance",),
        "support_low": ("More support",),
    }


def _rich_metrics() -> Mapping[str, Any]:
    return {
        "delta_nfr": 2.0,
        "sense_index": 0.6,
        "coherence_index": 0.45,
        "aero_mechanical_coherence": 0.4,
        "load_support_ratio": 0.3,
        "objectives": {
            "target_delta_nfr": 0.5,
            "target_sense_index": 0.8,
        },
    }


def test_resolve_playbook_suggestions_collects_active_rules() -> None:
    metrics = _rich_metrics()
    playbook = _playbook()
    suggestions = _resolve_playbook_suggestions(metrics, playbook=playbook)
    assert suggestions == (
        "Trim delta",
        "Balance exit",
        "Driver consistency",
        "Synchronise phases",
        "Aero balance",
        "More support",
    )


def test_augment_session_with_playbook_preserves_existing() -> None:
    metrics = _rich_metrics()
    playbook = _playbook()
    session = {"car_model": "XFG"}
    updated = _augment_session_with_playbook(session, metrics, playbook=playbook)
    assert session == {"car_model": "XFG"}
    assert updated["playbook_suggestions"] == _resolve_playbook_suggestions(
        metrics, playbook=playbook
    )
    existing = {
        "car_model": "XFG",
        "playbook_suggestions": ("Trim delta",),
    }
    enriched = _augment_session_with_playbook(existing, metrics, playbook=playbook)
    assert enriched["playbook_suggestions"] == (
        "Trim delta",
        "Balance exit",
        "Driver consistency",
        "Synchronise phases",
        "Aero balance",
        "More support",
    )


class _MemoryBuffer(io.BytesIO):
    def __enter__(self) -> "_MemoryBuffer":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self.close()
        return False


def test_cli_workflows_ignore_missing_playbook(monkeypatch: pytest.MonkeyPatch) -> None:
    workflows_module._reset_playbook_rules_cache()

    def _raise_missing(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(_PLAYBOOK_RESOURCE_TARGET, _raise_missing)

    try:
        metrics = _rich_metrics()
        assert workflows_module._resolve_playbook_suggestions(metrics) == ()
    finally:
        workflows_module._reset_playbook_rules_cache()


def test_cli_workflows_loads_playbook_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    workflows_module._reset_playbook_rules_cache()
    attempts = {"count": 0}

    def _open_playbook(*_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise FileNotFoundError
        payload = b"[rules]\ndelta_surplus = [\"Reinforce delta\"]\n"
        return _MemoryBuffer(payload)

    monkeypatch.setattr(_PLAYBOOK_RESOURCE_TARGET, _open_playbook)

    try:
        metrics = _rich_metrics()
        assert workflows_module._resolve_playbook_suggestions(metrics) == ()
        assert workflows_module._resolve_playbook_suggestions(metrics) == ("Reinforce delta",)
        assert attempts["count"] == 2
    finally:
        workflows_module._reset_playbook_rules_cache()


def test_cli_workflows_ignores_invalid_playbook(monkeypatch: pytest.MonkeyPatch) -> None:
    workflows_module._reset_playbook_rules_cache()

    def _open_invalid(*_args, **_kwargs):
        return _MemoryBuffer(b"not toml =")

    monkeypatch.setattr(_PLAYBOOK_RESOURCE_TARGET, _open_invalid)

    try:
        metrics = _rich_metrics()
        assert workflows_module._resolve_playbook_suggestions(metrics) == ()
    finally:
        workflows_module._reset_playbook_rules_cache()
