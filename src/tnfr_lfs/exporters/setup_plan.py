"""Utilities to serialise setup plans into exportable payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tnfr_lfs.math.conversions import _safe_float


_PHASE_SUMMARY_ORDER: Tuple[str, ...] = ("entry", "apex", "exit")
_PHASE_SUMMARY_LABELS: Dict[str, str] = {
    "entry": "Entry",
    "apex": "Apex",
    "exit": "Exit",
}
_AXIS_ORDER: Tuple[str, ...] = ("longitudinal", "lateral")
_AXIS_SYMBOLS: Dict[str, str] = {"longitudinal": "∥", "lateral": "⊥"}

def _format_phase_axis_cell(target: float, weight: float) -> str:
    emphasis = max(0.0, min(1.0, weight))
    magnitude = abs(target)
    if magnitude < 0.01 and emphasis < 0.05:
        return "·0.00"
    if magnitude < 0.01:
        arrow = "⇒"
    elif target > 0:
        if emphasis >= 0.66:
            arrow = "⇈"
        elif emphasis >= 0.33:
            arrow = "↑"
        else:
            arrow = "↗"
    elif target < 0:
        if emphasis >= 0.66:
            arrow = "⇊"
        elif emphasis >= 0.33:
            arrow = "↓"
        else:
            arrow = "↘"
    else:
        arrow = "→"
    return f"{arrow}{target:+.2f}"


def compute_phase_axis_summary(
    targets: Mapping[str, Mapping[str, float]] | None,
    weights: Mapping[str, Mapping[str, float]] | None,
) -> Tuple[Dict[str, Dict[str, str]], Tuple[str, ...]]:
    summary: Dict[str, Dict[str, str]] = {}
    suggestions: List[str] = []
    ranking: List[Tuple[float, str, str, str]] = []
    targets = targets or {}
    weights = weights or {}
    for axis in _AXIS_ORDER:
        axis_summary: Dict[str, str] = {}
        for phase in _PHASE_SUMMARY_ORDER:
            phase_targets = targets.get(phase, {})
            phase_weights = weights.get(phase, {})
            target_value = _safe_float(phase_targets.get(axis, 0.0))
            weight_value = _safe_float(phase_weights.get(axis, 0.0))
            cell = _format_phase_axis_cell(target_value, weight_value)
            axis_summary[phase] = cell
            score = abs(target_value) * max(weight_value, 0.05)
            if cell != "·0.00" and score > 0.0:
                ranking.append((score, phase, axis, cell))
        summary[axis] = axis_summary
    ranking.sort(reverse=True)
    for _, phase, axis, cell in ranking[:3]:
        phase_label = _PHASE_SUMMARY_LABELS.get(phase, phase)
        axis_label = _AXIS_SYMBOLS.get(axis, axis)
        suggestions.append(f"{phase_label} {axis_label} {cell}")
    return summary, tuple(suggestions)


def phase_axis_summary_lines(
    summary: Mapping[str, Mapping[str, str]] | None,
) -> Tuple[str, ...]:
    if not summary:
        return ()
    header_parts = ["Phase"]
    for phase in _PHASE_SUMMARY_ORDER:
        header_parts.append(f"{_PHASE_SUMMARY_LABELS.get(phase, phase):>7}")
    lines: List[str] = [" ".join(header_parts)]
    for axis in _AXIS_ORDER:
        axis_label = _AXIS_SYMBOLS.get(axis, axis[:1].upper())
        row_parts = [f"{axis_label:>4}"]
        for phase in _PHASE_SUMMARY_ORDER:
            cell = summary.get(axis, {}).get(phase, "·0.00")
            row_parts.append(f"{cell:>7}")
        lines.append(" ".join(row_parts))
    return tuple(lines)


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
    sci: float = 0.0
    changes: Sequence[SetupChange] = field(default_factory=tuple)
    rationales: Sequence[str] = field(default_factory=tuple)
    expected_effects: Sequence[str] = field(default_factory=tuple)
    sensitivities: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    phase_sensitivities: Mapping[str, Mapping[str, Mapping[str, float]]] = field(
        default_factory=dict
    )
    clamped_parameters: Sequence[str] = field(default_factory=tuple)
    tnfr_rationale_by_node: Mapping[str, Sequence[str]] = field(default_factory=dict)
    tnfr_rationale_by_phase: Mapping[str, Sequence[str]] = field(default_factory=dict)
    expected_effects_by_node: Mapping[str, Sequence[str]] = field(default_factory=dict)
    expected_effects_by_phase: Mapping[str, Sequence[str]] = field(default_factory=dict)
    phase_axis_targets: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    phase_axis_weights: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    phase_axis_summary: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    phase_axis_suggestions: Sequence[str] = field(default_factory=tuple)
    aero_guidance: str = ""
    aero_metrics: Mapping[str, float] = field(default_factory=dict)
    aero_mechanical_coherence: float = 0.0
    sci_breakdown: Mapping[str, float] = field(default_factory=dict)


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

    phase_sensitivities_payload: Dict[str, Dict[str, Dict[str, float]]] = {}
    for phase, metrics in plan.phase_sensitivities.items():
        phase_payload: Dict[str, Dict[str, float]] = {}
        for metric, derivatives in metrics.items():
            phase_payload[str(metric)] = {
                parameter: float(value)
                for parameter, value in derivatives.items()
            }
        phase_sensitivities_payload[str(phase)] = phase_payload

    def _normalise_mapping(
        mapping: Mapping[str, Sequence[str]] | None,
    ) -> Dict[str, List[str]]:
        if not mapping:
            return {}
        return {str(key): _merge_unique(list(values)) for key, values in mapping.items()}

    def _normalise_axis_mapping(
        mapping: Mapping[str, Mapping[str, float]] | None,
    ) -> Dict[str, Dict[str, float]]:
        if not mapping:
            return {}
        return {
            str(phase): {str(axis): float(value) for axis, value in values.items()}
            for phase, values in mapping.items()
        }

    dsi_payload = {
        parameter: float(value)
        for parameter, value in plan.sensitivities.get("sense_index", {}).items()
    }

    dnfr_payload = {
        parameter: float(value)
        for parameter, value in plan.sensitivities.get("delta_nfr_integral", {}).items()
    }

    phase_dnfr_payload = {
        phase: {
            parameter: float(value)
            for parameter, value in metrics.get("delta_nfr_integral", {}).items()
        }
        for phase, metrics in plan.phase_sensitivities.items()
        if "delta_nfr_integral" in metrics
    }

    tnfr_node = _normalise_mapping(plan.tnfr_rationale_by_node)
    tnfr_phase = _normalise_mapping(plan.tnfr_rationale_by_phase)
    effects_node = _normalise_mapping(plan.expected_effects_by_node)
    effects_phase = _normalise_mapping(plan.expected_effects_by_phase)
    axis_targets = _normalise_axis_mapping(plan.phase_axis_targets)
    axis_weights = _normalise_axis_mapping(plan.phase_axis_weights)
    summary_mapping = plan.phase_axis_summary
    suggestions_seq = plan.phase_axis_suggestions
    if (not summary_mapping or not suggestions_seq) and (axis_targets or axis_weights):
        computed_summary, computed_suggestions = compute_phase_axis_summary(
            plan.phase_axis_targets,
            plan.phase_axis_weights,
        )
        if not summary_mapping:
            summary_mapping = computed_summary
        if not suggestions_seq:
            suggestions_seq = computed_suggestions
    summary_payload = {
        str(axis): {str(phase): str(value) for phase, value in phases.items()}
        for axis, phases in (summary_mapping or {}).items()
    }
    suggestions_payload = [str(entry) for entry in (suggestions_seq or ())]

    sci_breakdown = {str(key): float(value) for key, value in plan.sci_breakdown.items()}

    payload = {
        "car_model": plan.car_model,
        "session": plan.session,
        "sci": float(plan.sci),
        "changes": changes_payload,
        "rationales": rationales,
        "expected_effects": expected_effects,
        "sensitivities": sensitivities_payload,
        "dsi_dparam": dsi_payload,
        "phase_sensitivities": phase_sensitivities_payload,
        "dnfr_integral_dparam": dnfr_payload,
        "phase_dnfr_integral_dparam": phase_dnfr_payload,
        "clamped_parameters": list(plan.clamped_parameters),
        "tnfr_rationale_by_node": tnfr_node,
        "tnfr_rationale_by_phase": tnfr_phase,
        "expected_effects_by_node": effects_node,
        "expected_effects_by_phase": effects_phase,
        "phase_axis_targets": axis_targets,
        "phase_axis_weights": axis_weights,
        "phase_axis_summary": summary_payload,
        "phase_axis_suggestions": suggestions_payload,
        "aero_guidance": plan.aero_guidance,
        "aero_metrics": {str(key): float(value) for key, value in plan.aero_metrics.items()},
        "aero_mechanical_coherence": float(plan.aero_mechanical_coherence),
        "sci_breakdown": sci_breakdown,
    }

    if sci_breakdown:
        payload["ics_breakdown"] = sci_breakdown

    return payload


__all__ = [
    "SetupChange",
    "SetupPlan",
    "serialise_setup_plan",
    "compute_phase_axis_summary",
    "phase_axis_summary_lines",
]
