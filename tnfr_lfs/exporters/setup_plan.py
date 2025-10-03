"""Utilities to serialise setup plans into exportable payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class SetupChange:
    """Represents an actionable change in the car setup."""

    parameter: str
    delta: float
    rationale: str
    expected_effect: str


@dataclass(frozen=True)
class SetupPlan:
    """Structured plan combining optimisation insights with explainability."""

    car_model: str
    session: str | None
    changes: Sequence[SetupChange] = field(default_factory=tuple)
    rationales: Sequence[str] = field(default_factory=tuple)
    expected_effects: Sequence[str] = field(default_factory=tuple)
    sensitivities: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    tnfr_rationale_by_node: Mapping[str, Sequence[str]] = field(default_factory=dict)
    tnfr_rationale_by_phase: Mapping[str, Sequence[str]] = field(default_factory=dict)
    expected_effects_by_node: Mapping[str, Sequence[str]] = field(default_factory=dict)
    expected_effects_by_phase: Mapping[str, Sequence[str]] = field(default_factory=dict)


def serialise_setup_plan(plan: SetupPlan) -> Dict[str, Any]:
    """Convert a :class:`SetupPlan` into a serialisable mapping."""

    changes_payload = [
        {
            "parameter": change.parameter,
            "delta": change.delta,
            "rationale": change.rationale,
            "expected_effect": change.expected_effect,
        }
        for change in plan.changes
    ]

    def _merge_unique(items: Iterable[str]) -> List[str]:
        seen: set[str] = set()
        merged: List[str] = []
        for item in items:
            if not item:
                continue
            if item not in seen:
                merged.append(item)
                seen.add(item)
        return merged

    rationales = _merge_unique(list(plan.rationales) + [change.rationale for change in plan.changes])
    expected_effects = _merge_unique(
        list(plan.expected_effects) + [change.expected_effect for change in plan.changes]
    )

    sensitivities_payload = {
        metric: {parameter: float(value) for parameter, value in derivatives.items()}
        for metric, derivatives in plan.sensitivities.items()
    }

    def _normalise_mapping(
        mapping: Mapping[str, Sequence[str]] | None,
    ) -> Dict[str, List[str]]:
        if not mapping:
            return {}
        return {str(key): _merge_unique(list(values)) for key, values in mapping.items()}

    dsi_payload = {
        parameter: float(value)
        for parameter, value in plan.sensitivities.get("sense_index", {}).items()
    }

    tnfr_node = _normalise_mapping(plan.tnfr_rationale_by_node)
    tnfr_phase = _normalise_mapping(plan.tnfr_rationale_by_phase)
    effects_node = _normalise_mapping(plan.expected_effects_by_node)
    effects_phase = _normalise_mapping(plan.expected_effects_by_phase)

    return {
        "car_model": plan.car_model,
        "session": plan.session,
        "changes": changes_payload,
        "rationales": rationales,
        "expected_effects": expected_effects,
        "sensitivities": sensitivities_payload,
        "dsi_dparam": dsi_payload,
        "tnfr_rationale_by_node": tnfr_node,
        "tnfr_rationale_by_phase": tnfr_phase,
        "expected_effects_by_node": effects_node,
        "expected_effects_by_phase": effects_phase,
    }


__all__ = ["SetupChange", "SetupPlan", "serialise_setup_plan"]
