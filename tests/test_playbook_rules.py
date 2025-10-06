from typing import Any, Mapping

from tnfr_lfs.cli.tnfr_lfs_cli import (
    _augment_session_with_playbook,
    _resolve_playbook_suggestions,
)


def _playbook() -> Mapping[str, tuple[str, ...]]:
    return {
        "delta_surplus": ("Ajusta delta", "Equilibra salida"),
        "delta_deficit": ("Refuerza carga",),
        "sense_index_low": ("Constancia piloto",),
        "coherence_low": ("Sincroniza fases",),
        "aero_balance_low": ("Balance aero",),
        "support_low": ("Más soporte",),
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
        "Ajusta delta",
        "Equilibra salida",
        "Constancia piloto",
        "Sincroniza fases",
        "Balance aero",
        "Más soporte",
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
        "playbook_suggestions": ("Ajusta delta",),
    }
    enriched = _augment_session_with_playbook(existing, metrics, playbook=playbook)
    assert enriched["playbook_suggestions"] == (
        "Ajusta delta",
        "Equilibra salida",
        "Constancia piloto",
        "Sincroniza fases",
        "Balance aero",
        "Más soporte",
    )
