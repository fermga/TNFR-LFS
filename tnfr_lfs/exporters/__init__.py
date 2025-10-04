"""Exporter registry for TNFR × LFS outputs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Protocol, Sequence

from ..core.epi_models import EPIBundle
from .setup_plan import SetupPlan, serialise_setup_plan


CAR_MODEL_PREFIXES = {
    "UF1": "UF1",
    "XFG": "XFG",
    "XRG": "XRG",
    "LX4": "LX4",
    "LX6": "LX6",
    "RB4": "RB4",
    "FXO": "FXO",
    "XRT": "XRT",
    "RAC": "RAC",
    "FZ5": "FZ5",
    "MRT": "MRT",
    "FBM": "FBM",
    "FOX": "FOX",
    "FO8": "FO8",
    "BF1": "BF1",
    "XFR": "XFR",
    "UFR": "UFR",
    "FXR": "FXR",
    "XRR": "XRR",
    "FZR": "FZR",
    "generic_gt": "GEN",
    "formula": "FOR",
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
    buffer.write("timestamp,epi,delta_nfr,delta_nfr_longitudinal,delta_nfr_lateral,sense_index\n")
    for result in results.get("series", []):
        if not isinstance(result, EPIBundle):
            raise TypeError("CSV exporter expects EPIBundle instances")
        buffer.write(
            f"{result.timestamp:.3f},{result.epi:.4f},{result.delta_nfr:.3f},{result.delta_nfr_longitudinal:.3f},{result.delta_nfr_lateral:.3f},{result.sense_index:.3f}\n"
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

    sensitivities = plan.get("sensitivities", {})
    if sensitivities:
        lines.append("")
        lines.append("**Sensibilidades agregadas**")
        lines.append("| Métrica | Parámetro | Derivada |")
        lines.append("| --- | --- | --- |")
        for metric in sorted(sensitivities):
            derivatives = sensitivities[metric]
            for parameter in sorted(derivatives):
                value = derivatives[parameter]
                try:
                    formatted = f"{float(value):+.6f}"
                except (TypeError, ValueError):
                    formatted = str(value)
                lines.append(f"| {metric} | {parameter} | {formatted} |")

    dsi_dparam = plan.get("dsi_dparam", {})
    if dsi_dparam:
        lines.append("")
        lines.append("**dSi/dparam**")
        lines.append("| Parámetro | Sensibilidad |")
        lines.append("| --- | --- |")
        for parameter in sorted(dsi_dparam):
            value = dsi_dparam[parameter]
            try:
                formatted = f"{float(value):+.6f}"
            except (TypeError, ValueError):
                formatted = str(value)
            lines.append(f"| {parameter} | {formatted} |")

    dnfr_dparam = plan.get("dnfr_integral_dparam", {})
    if dnfr_dparam:
        lines.append("")
        lines.append("**d∫|ΔNFR|/dparam**")
        lines.append("| Parámetro | Sensibilidad |")
        lines.append("| --- | --- |")
        for parameter in sorted(dnfr_dparam):
            value = dnfr_dparam[parameter]
            try:
                formatted = f"{float(value):+.6f}"
            except (TypeError, ValueError):
                formatted = str(value)
            lines.append(f"| {parameter} | {formatted} |")

    phase_dnfr = plan.get("phase_dnfr_integral_dparam", {})
    if phase_dnfr:
        lines.append("")
        lines.append("**Gradientes de ∫|ΔNFR| por fase**")
        lines.append("| Fase | Parámetro | Sensibilidad |")
        lines.append("| --- | --- | --- |")
        for phase in sorted(phase_dnfr):
            derivatives = phase_dnfr[phase]
            for parameter in sorted(derivatives):
                value = derivatives[parameter]
                try:
                    formatted = f"{float(value):+.6f}"
                except (TypeError, ValueError):
                    formatted = str(value)
                lines.append(f"| {phase} | {parameter} | {formatted} |")

    axis_targets = plan.get("phase_axis_targets", {}) or {}
    axis_weights = plan.get("phase_axis_weights", {}) or {}
    if axis_targets or axis_weights:
        lines.append("")
        lines.append("**Objetivos ΔNFR∥/ΔNFR⊥ por fase**")
        lines.append("| Fase | ΔNFR∥ obj | ΔNFR⊥ obj | Peso ∥ | Peso ⊥ |")
        lines.append("| --- | --- | --- | --- | --- |")
        for phase in sorted(set(axis_targets) | set(axis_weights)):
            target = axis_targets.get(phase, {})
            weight = axis_weights.get(phase, {})
            target_long = float(target.get("longitudinal", 0.0))
            target_lat = float(target.get("lateral", 0.0))
            weight_long = float(weight.get("longitudinal", 0.0))
            weight_lat = float(weight.get("lateral", 0.0))
            lines.append(
                f"| {phase} | {target_long:+.3f} | {target_lat:+.3f} | {weight_long:.2f} | {weight_lat:.2f} |"
            )

    def _extend_mapping_section(title: str, mapping: Mapping[str, Sequence[str]] | None) -> None:
        if not mapping:
            return
        items = []
        for key in sorted(mapping):
            for entry in mapping[key]:
                if entry:
                    items.append((key, entry))
        if not items:
            return
        lines.append("")
        lines.append(title)
        for key, entry in items:
            lines.append(f"- **{key}**: {entry}")

    _extend_mapping_section("**Racionales TNFR por nodo**", plan.get("tnfr_rationale_by_node"))
    _extend_mapping_section("**Racionales TNFR por fase**", plan.get("tnfr_rationale_by_phase"))
    _extend_mapping_section("**Efectos esperados por nodo**", plan.get("expected_effects_by_node"))
    _extend_mapping_section("**Efectos esperados por fase**", plan.get("expected_effects_by_phase"))

    return "\n".join(lines)


def _format_key_instruction(delta: Any) -> tuple[str, str, str]:
    try:
        value = float(delta)
    except (TypeError, ValueError):
        return ("Manual", "N/A", "Ajuste manual requerido")

    if abs(value) < 1e-9:
        return ("N/A", "0", "Sin cambios necesarios")

    key = "F12" if value > 0 else "F11"
    steps = f"{abs(value):.3f}".rstrip("0").rstrip(".")
    return (key, steps, f"Pulsa {key} × {steps}")


def lfs_notes_exporter(results: Dict[str, Any] | SetupPlan) -> str:
    """Render TNFR adjustments as F11/F12 instructions for Live for Speed."""

    plan = _extract_setup_plan(results)

    header = "| Cambio | Δ | Acción | Pasos | Racional | Efecto esperado |"
    separator = "| --- | --- | --- | --- | --- | --- |"
    lines = ["# Instrucciones rápidas TNFR → LFS", ""]
    lines.append(header)
    lines.append(separator)

    for change in plan.get("changes", []):
        parameter = change.get("parameter", "-") or "-"
        delta = change.get("delta", 0.0)
        try:
            delta_repr = f"{float(delta):+.3f}"
        except (TypeError, ValueError):
            delta_repr = str(delta)
        key, steps, instruction = _format_key_instruction(delta)
        rationale = change.get("rationale", "-") or "-"
        expected = change.get("expected_effect", "-") or "-"
        lines.append(
            f"| {parameter} | {delta_repr} | {instruction} | {steps} | {rationale} | {expected} |"
        )

    def _extend_list_section(title: str, items: Iterable[str] | None) -> None:
        entries = [item for item in items or () if item]
        if not entries:
            return
        lines.append("")
        lines.append(title)
        lines.extend(f"- {entry}" for entry in entries)

    _extend_list_section("**Notas agregadas**", plan.get("rationales"))
    _extend_list_section("**Efectos agregados**", plan.get("expected_effects"))

    clamped = [item for item in plan.get("clamped_parameters", []) if item]
    if clamped:
        lines.append("")
        lines.append("**Parámetros bloqueados**")
        lines.extend(f"- {item}" for item in clamped)

    return "\n".join(lines)


def resolve_car_prefix(car_model: str) -> str:
    key = (car_model or "").strip()
    if not key:
        raise ValueError("Debe especificarse un 'car_model' para exportar setups.")
    for candidate in (key, key.upper(), key.lower()):
        if candidate in CAR_MODEL_PREFIXES:
            return CAR_MODEL_PREFIXES[candidate]
    raise ValueError(f"No hay prefijo LFS registrado para '{car_model}'.")


def normalise_set_output_name(name: str, car_model: str) -> str:
    prefix = resolve_car_prefix(car_model)
    return _normalise_setup_name(name, prefix)


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
    prefix = resolve_car_prefix(car_model)

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
    "lfs-notes": lfs_notes_exporter,
}

__all__ = [
    "CAR_MODEL_PREFIXES",
    "Exporter",
    "json_exporter",
    "csv_exporter",
    "markdown_exporter",
    "lfs_notes_exporter",
    "lfs_set_exporter",
    "normalise_set_output_name",
    "resolve_car_prefix",
    "exporters_registry",
]
