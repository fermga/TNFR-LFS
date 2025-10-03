"""Utilities to serialise setup plans into exportable payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence


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

    return {
        "car_model": plan.car_model,
        "session": plan.session,
        "changes": changes_payload,
        "rationales": rationales,
        "expected_effects": expected_effects,
    }


__all__ = ["SetupChange", "SetupPlan", "serialise_setup_plan"]
