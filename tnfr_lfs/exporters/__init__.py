"""Exporter registry for TNFR × LFS outputs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol

from ..core.epi_models import EPIBundle
from .setup_plan import SetupPlan, serialise_setup_plan


CAR_MODEL_PREFIXES = {
    "generic_gt": "GEN",
}

INVALID_NAME_CHARS = set("\\/:*?\"<>|")


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


def _resolve_car_prefix(car_model: str) -> str:
    try:
        return CAR_MODEL_PREFIXES[car_model]
    except KeyError as exc:
        raise ValueError(f"No hay prefijo LFS registrado para '{car_model}'.") from exc


def _normalise_delta_value(delta: Any) -> str:
    try:
        value = float(delta)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Delta inválido para exportación .set: {delta!r}") from exc
    return f"{value:+.3f}"


def _normalise_setup_name(name: str, prefix: str) -> str:
    candidate = (name or "").strip()
    if not candidate:
        raise ValueError("Nombre de setup vacío; utilice --set-output para definirlo.")
    if candidate.lower().endswith(".set"):
        candidate = candidate[:-4]
    if Path(candidate).name != candidate:
        raise ValueError("El nombre de setup no debe incluir rutas relativas o absolutas.")
    if any(char in INVALID_NAME_CHARS for char in candidate):
        raise ValueError(
            "Nombre de setup inválido: use solo letras, números y guiones bajos."
        )
    if not candidate.upper().startswith(prefix.upper()):
        raise ValueError(
            f"El nombre '{candidate}' debe comenzar con el prefijo de coche '{prefix}'."
        )
    return f"{candidate}.set"


def lfs_set_exporter(results: Dict[str, Any] | SetupPlan) -> str:
    """Persist a setup plan using the textual LFS ``.set`` representation."""

    set_output: str | None = None
    if isinstance(results, Mapping):
        raw = results.get("set_output")
        if raw is not None:
            set_output = str(raw)

    plan = _extract_setup_plan(results)
    car_model = str(plan.get("car_model") or "").strip()
    if not car_model:
        raise ValueError("El plan de setup no define 'car_model'.")
    prefix = _resolve_car_prefix(car_model)

    if set_output is None:
        session_label = str(plan.get("session") or "setup").strip().replace(" ", "_") or "setup"
        set_output = f"{prefix}_{session_label}"

    file_name = _normalise_setup_name(set_output, prefix)
    output_dir = Path("LFS/data/setups")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / file_name

    lines = [
        "; TNFR-LFS setup export",
        f"; Car model: {car_model}",
        f"; Session: {plan.get('session') or 'N/A'}",
        f"; File: {destination.name}",
        "",
        "[setup]",
        f"car={prefix}",
        f"name={destination.stem}",
        "",
        "[changes]",
    ]

    for change in plan.get("changes", []):
        parameter = str(change.get("parameter", ""))
        delta = _normalise_delta_value(change.get("delta", 0.0))
        lines.append(f"{parameter}={delta}")

    rationales = [item for item in plan.get("rationales", []) if item]
    if rationales:
        lines.append("")
        lines.append("[notes]")
        lines.extend(f"; {note}" for note in rationales)

    destination.write_text("\n".join(lines) + "\n", encoding="utf8")
    return f"Setup guardado en {destination.resolve()}"


exporters_registry = {
    "json": json_exporter,
    "csv": csv_exporter,
    "markdown": markdown_exporter,
    "set": lfs_set_exporter,
}

__all__ = [
    "Exporter",
    "json_exporter",
    "csv_exporter",
    "markdown_exporter",
    "lfs_set_exporter",
    "exporters_registry",
]
