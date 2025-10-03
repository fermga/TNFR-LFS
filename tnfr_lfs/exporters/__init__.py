"""Exporter registry for TNFR × LFS outputs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Protocol

from ..core.epi_models import EPIBundle
from .setup_plan import SetupPlan, serialise_setup_plan


class Exporter(Protocol):
    """Exporter callable protocol."""

    def __call__(self, results: Dict[str, Any]) -> str:  # pragma: no cover - interface only
        ...


def _normalise(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalise(item) for key, item in value.items()}
    return value


def json_exporter(results: Dict[str, Any]) -> str:
    import json

    payload = _normalise(results)
    return json.dumps(payload, indent=2, sort_keys=True)


def csv_exporter(results: Dict[str, Any]) -> str:
    from io import StringIO

    buffer = StringIO()
    buffer.write("timestamp,epi,delta_nfr,sense_index\n")
    for result in results.get("series", []):
        if not isinstance(result, EPIBundle):
            raise TypeError("CSV exporter expects EPIBundle instances")
        buffer.write(
            f"{result.timestamp:.3f},{result.epi:.4f},{result.delta_nfr:.3f},{result.sense_index:.3f}\n"
        )
    return buffer.getvalue()


def _extract_setup_plan(results: Mapping[str, Any] | SetupPlan | None) -> Mapping[str, Any]:
    if results is None:
        raise TypeError("Markdown exporter requires a setup plan payload")
    if isinstance(results, SetupPlan):
        return serialise_setup_plan(results)
    if isinstance(results, Mapping):
        maybe_plan = results.get("setup_plan") if "setup_plan" in results else results
        if isinstance(maybe_plan, SetupPlan):
            return serialise_setup_plan(maybe_plan)
        if isinstance(maybe_plan, Mapping):
            return maybe_plan
    raise TypeError("Markdown exporter requires a SetupPlan instance or mapping under 'setup_plan'")


def markdown_exporter(results: Dict[str, Any] | SetupPlan) -> str:
    """Render a setup plan as a Markdown table with rationales."""

    plan = _extract_setup_plan(results)
    header = "| Cambio | Ajuste | Racional | Efecto esperado |"
    separator = "| --- | --- | --- | --- |"
    lines = [header, separator]

    for change in plan.get("changes", []):
        parameter = change.get("parameter", "-")
        delta = change.get("delta", 0.0)
        if isinstance(delta, float):
            delta_repr = f"{delta:+.3f}"
        else:
            delta_repr = str(delta)
        rationale = change.get("rationale", "-") or "-"
        expected = change.get("expected_effect", "-") or "-"
        lines.append(f"| {parameter} | {delta_repr} | {rationale} | {expected} |")

    rationales = [item for item in plan.get("rationales", []) if item]
    if rationales:
        lines.append("")
        lines.append("**Racionales**")
        lines.extend(f"- {item}" for item in rationales)

    expected_effects = [item for item in plan.get("expected_effects", []) if item]
    if expected_effects:
        lines.append("")
        lines.append("**Efectos esperados**")
        lines.extend(f"- {item}" for item in expected_effects)

    return "\n".join(lines)


exporters_registry = {
    "json": json_exporter,
    "csv": csv_exporter,
    "markdown": markdown_exporter,
}

__all__ = [
    "Exporter",
    "json_exporter",
    "csv_exporter",
    "markdown_exporter",
    "exporters_registry",
]
