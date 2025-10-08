"""Exporter registry for TNFR × LFS outputs."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Protocol, Sequence, Tuple

from ..core.epi_models import EPIBundle
from ..core.operator_detection import canonical_operator_label
from .setup_plan import (
    SetupPlan,
    phase_axis_summary_lines,
    serialise_setup_plan,
)
from .report_extended import html_exporter
from ..session import format_session_messages


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
    "gt_fzr": "FZR",
    "gt_xrr": "XRR",
    "formula_fo8": "FO8",
    "formula_fox": "FOX",
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
    if isinstance(value, Mapping):
        return {key: _normalise(item) for key, item in value.items()}
    return value


def json_exporter(results: Dict[str, Any]) -> str:
    payload = _normalise(results)
    return json.dumps(payload, indent=2, sort_keys=True)


def csv_exporter(results: Dict[str, Any]) -> str:
    from io import StringIO

    buffer = StringIO()
    buffer.write("timestamp,epi,delta_nfr,delta_nfr_proj_longitudinal,delta_nfr_proj_lateral,sense_index\n")
    for result in results.get("series", []):
        if not isinstance(result, EPIBundle):
            raise TypeError("CSV exporter expects EPIBundle instances")
        buffer.write(
            f"{result.timestamp:.3f},{result.epi:.4f},{result.delta_nfr:.3f},{result.delta_nfr_proj_longitudinal:.3f},{result.delta_nfr_proj_lateral:.3f},{result.sense_index:.3f}\n"
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


def _resolve_session_messages(
    payload: Mapping[str, Any] | None,
) -> Tuple[Tuple[str, ...], Dict[str, Any] | None]:
    session_messages: Tuple[str, ...] = ()
    abtest_payload: Dict[str, Any] | None = None
    if not isinstance(payload, Mapping):
        return session_messages, abtest_payload

    session_messages = format_session_messages(payload.get("session"))
    if not session_messages:
        extras = payload.get("session_messages")
        if isinstance(extras, Sequence):
            session_messages = tuple(str(item) for item in extras if item)

    session_section = payload.get("session")
    if isinstance(session_section, Mapping):
        abtest_candidate = session_section.get("abtest")
        if is_dataclass(abtest_candidate):
            abtest_payload = asdict(abtest_candidate)
        elif isinstance(abtest_candidate, Mapping):
            abtest_payload = dict(abtest_candidate)

    return session_messages, abtest_payload


def markdown_exporter(results: Dict[str, Any] | SetupPlan) -> str:
    """Render a setup plan as a Markdown table with rationales."""

    plan = _extract_setup_plan(results)
    session_messages, abtest_payload = _resolve_session_messages(
        results if isinstance(results, Mapping) else None
    )
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

    def _fmt(value: Any, *, signed: bool = False, decimals: int = 4) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "-"
        if not math.isfinite(numeric):
            return "-"
        template = f"{{:+.{decimals}f}}" if signed else f"{{:.{decimals}f}}"
        return template.format(numeric)

    if abtest_payload:
        lines.append("")
        lines.append("**A/B comparison**")
        metric_label = abtest_payload.get("metric")
        if metric_label:
            lines.append(f"Metric: `{metric_label}`")
        lines.append("| Statistic | Baseline | Variant | Δ / details |")
        lines.append("| --- | --- | --- | --- |")
        lines.append(
            "| Mean | {baseline} | {variant} | {delta} |".format(
                baseline=_fmt(abtest_payload.get("baseline_mean")),
                variant=_fmt(abtest_payload.get("variant_mean")),
                delta=_fmt(abtest_payload.get("mean_difference"), signed=True),
            )
        )
        alpha = float(abtest_payload.get("alpha", 0.05))
        ci_low = _fmt(abtest_payload.get("bootstrap_low"), signed=True)
        ci_high = _fmt(abtest_payload.get("bootstrap_high"), signed=True)
        interval = f"[{ci_low}, {ci_high}]"
        confidence_label = f"CI {int(round((1.0 - alpha) * 100))}%"
        lines.append(f"| {confidence_label} | - | - | {interval} |")
        lines.append(
            "| p (perm) | - | - | {value} |".format(
                value=_fmt(abtest_payload.get("permutation_p_value"), decimals=4)
            )
        )
        lines.append(
            "| Power | - | - | {value} |".format(
                value=_fmt(abtest_payload.get("estimated_power"), decimals=3)
            )
        )
        baseline_laps = abtest_payload.get("baseline_laps")
        variant_laps = abtest_payload.get("variant_laps")
        if isinstance(baseline_laps, Sequence) or isinstance(variant_laps, Sequence):
            baseline_count = len(baseline_laps) if isinstance(baseline_laps, Sequence) else 0
            variant_count = len(variant_laps) if isinstance(variant_laps, Sequence) else 0
            lines.append(
                f"| Laps | {baseline_count} | {variant_count} | - |"
            )
        if isinstance(baseline_laps, Sequence) and baseline_laps:
            formatted = ", ".join(_fmt(value) for value in baseline_laps)
            lines.append("")
            lines.append(f"Baseline laps: {formatted}")
        if isinstance(variant_laps, Sequence) and variant_laps:
            formatted = ", ".join(_fmt(value) for value in variant_laps)
            lines.append(f"Variant laps: {formatted}")

    aero_guidance = plan.get("aero_guidance")
    aero_metrics = plan.get("aero_metrics", {}) or {}
    amc_value = plan.get("aero_mechanical_coherence")
    if amc_value is None:
        amc_value = aero_metrics.get("aero_mechanical_coherence")
    if aero_guidance or aero_metrics or amc_value is not None:
        lines.append("")
        lines.append("**Aerodynamic indicators**")
        if amc_value is not None:
            try:
                lines.append(f"- C(c/d/a) {float(amc_value):.2f}")
            except (TypeError, ValueError):
                lines.append(f"- C(c/d/a) {amc_value}")
        if aero_guidance:
            lines.append(f"- {aero_guidance}")
        high = aero_metrics.get("high_speed_imbalance")
        low = aero_metrics.get("low_speed_imbalance")
        if high is not None and low is not None:
            lines.append(
                f"- High-speed aero Δ {float(high):+.2f} · low-speed {float(low):+.2f}"
            )

    sensitivities = plan.get("sensitivities", {})
    if sensitivities:
        lines.append("")
        lines.append("**Aggregated sensitivities**")
        lines.append("| Metric | Parameter | Derivative |")
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
        lines.append("| Parameter | Sensitivity |")
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
        lines.append("| Parameter | Sensitivity |")
        lines.append("| --- | --- |")
        for parameter in sorted(dnfr_dparam):
            value = dnfr_dparam[parameter]
            try:
                formatted = f"{float(value):+.6f}"
            except (TypeError, ValueError):
                formatted = str(value)
            lines.append(f"| {parameter} | {formatted} |")

    sci_breakdown = (
        plan.get("sci_breakdown")
        or plan.get("ics_breakdown")
        or plan.get("objective_breakdown")
    )
    if sci_breakdown:
        lines.append("")
        lines.append("**SCI contribution**")
        lines.append("| Term | Contribution |")
        lines.append("| --- | --- |")
        for term in sorted(sci_breakdown):
            value = sci_breakdown[term]
            try:
                formatted = f"{float(value):.4f}"
            except (TypeError, ValueError):
                formatted = str(value)
            lines.append(f"| {term} | {formatted} |")

    phase_dnfr = plan.get("phase_dnfr_integral_dparam", {})
    if phase_dnfr:
        lines.append("")
        lines.append("**∫|ΔNFR| gradients per phase**")
        lines.append("| Phase | Parameter | Sensitivity |")
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
        lines.append("**∇NFR∥/∇NFR⊥ projection targets per phase**")
        lines.append("| Phase | ∇NFR∥ target | ∇NFR⊥ target | Weight ∥ | Weight ⊥ |")
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
    summary_lines = phase_axis_summary_lines(plan.get("phase_axis_summary"))
    if summary_lines:
        lines.append("")
        lines.append("**∇NFR∥/∇NFR⊥ projection map per phase**")
        lines.append("```")
        lines.extend(summary_lines)
        lines.append("```")
    suggestions = [hint for hint in plan.get("phase_axis_suggestions", []) if hint]
    if suggestions:
        lines.append("")
        lines.append("**Priority phase suggestions**")
        for hint in suggestions:
            lines.append(f"- {hint}")

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

    _extend_mapping_section("**TNFR rationales per node**", plan.get("tnfr_rationale_by_node"))
    _extend_mapping_section("**TNFR rationales per phase**", plan.get("tnfr_rationale_by_phase"))
    _extend_mapping_section("**Expected effects per node**", plan.get("expected_effects_by_node"))
    _extend_mapping_section("**Expected effects per phase**", plan.get("expected_effects_by_phase"))
    if session_messages:
        lines.append("")
        lines.append("**Session profile**")
        lines.extend(f"- {message}" for message in session_messages)

    return "\n".join(lines)


def _format_key_instruction(delta: Any) -> tuple[str, str, str]:
    try:
        value = float(delta)
    except (TypeError, ValueError):
        return ("Manual", "N/A", "Manual adjustment required")

    if abs(value) < 1e-9:
        return ("N/A", "0", "No changes required")

    key = "F12" if value > 0 else "F11"
    steps = f"{abs(value):.3f}".rstrip("0").rstrip(".")
    return (key, steps, f"Press {key} × {steps}")


def lfs_notes_exporter(results: Dict[str, Any] | SetupPlan) -> str:
    """Render TNFR adjustments as F11/F12 instructions for Live for Speed."""

    plan = _extract_setup_plan(results)
    session_messages, _ = _resolve_session_messages(
        results if isinstance(results, Mapping) else None
    )

    header = "| Change | Δ | Action | Steps | Rationale | Expected effect |"
    separator = "| --- | --- | --- | --- | --- | --- |"
    lines = ["# Quick TNFR notes → LFS", ""]
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

    _extend_list_section("**Aggregated notes**", plan.get("rationales"))
    _extend_list_section("**Aggregated expected effects**", plan.get("expected_effects"))

    clamped = [item for item in plan.get("clamped_parameters", []) if item]
    if clamped:
        lines.append("")
        lines.append("**Locked parameters**")
        lines.extend(f"- {item}" for item in clamped)
    if session_messages:
        lines.append("")
        lines.append("**Session profile**")
        lines.extend(f"- {message}" for message in session_messages)

    return "\n".join(lines)


def resolve_car_prefix(car_model: str) -> str:
    key = (car_model or "").strip()
    if not key:
        raise ValueError("A 'car_model' must be provided to export setups.")
    for candidate in (key, key.upper(), key.lower()):
        if candidate in CAR_MODEL_PREFIXES:
            return CAR_MODEL_PREFIXES[candidate]
    raise ValueError(f"No LFS prefix is registered for '{car_model}'.")


def normalise_set_output_name(name: str, car_model: str) -> str:
    prefix = resolve_car_prefix(car_model)
    return _normalise_setup_name(name, prefix)


def _normalise_delta_value(delta: Any) -> str:
    try:
        value = float(delta)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid delta for .set export: {delta!r}") from exc
    return f"{value:+.3f}"


def _normalise_setup_name(name: str, prefix: str) -> str:
    candidate = (name or "").strip()
    if not candidate:
        raise ValueError("Setup name cannot be empty; use --set-output to define it.")
    if candidate.lower().endswith(".set"):
        candidate = candidate[:-4]
    if Path(candidate).name != candidate:
        raise ValueError("The setup name must not include relative or absolute paths.")
    if any(char in INVALID_NAME_CHARS for char in candidate):
        raise ValueError(
            "Invalid setup name: use only letters, numbers, and underscores."
        )
    if not candidate.upper().startswith(prefix.upper()):
        raise ValueError(
            f"The name '{candidate}' must start with the car prefix '{prefix}'."
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
        raise ValueError("The setup plan does not define 'car_model'.")
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
    return f"Setup saved to {destination.resolve()}"


REPORT_ARTIFACT_FORMATS = ("json", "markdown", "visual")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _series_from_results(results: Mapping[str, Any]) -> Sequence[EPIBundle]:
    series = results.get("series")
    if isinstance(series, Sequence):
        return series
    raise TypeError(
        "Exporters require a 'series' sequence containing EPIBundle entries."
    )


def _microsectors_from_results(results: Mapping[str, Any]) -> Sequence[Any]:
    microsectors = results.get("microsectors")
    if isinstance(microsectors, Sequence):
        return microsectors
    raise TypeError(
        "Exporters require a 'microsectors' sequence derived from segmentation."
    )


def _structural_lookup(series: Sequence[EPIBundle]) -> Sequence[Mapping[str, float]]:
    lookup: list[Mapping[str, float]] = []
    for index, bundle in enumerate(series):
        timestamp = _to_float(getattr(bundle, "timestamp", index), default=float(index))
        structural = getattr(bundle, "structural_timestamp", None)
        if structural is None or not math.isfinite(structural):
            structural = timestamp if math.isfinite(timestamp) else float(index)
        lookup.append(
            {
                "index": float(index),
                "timestamp": float(timestamp),
                "structural": float(structural),
            }
        )
    return lookup


def _track_progress(series: Sequence[EPIBundle]) -> Sequence[Mapping[str, float]]:
    progress: list[Mapping[str, float]] = []
    distance = 0.0
    prev_timestamp: float | None = None
    for index, bundle in enumerate(series):
        timestamp = _to_float(getattr(bundle, "timestamp", index), default=float(index))
        structural = getattr(bundle, "structural_timestamp", None)
        if structural is None or not math.isfinite(structural):
            structural = timestamp if math.isfinite(timestamp) else float(index)
        if prev_timestamp is None:
            dt = 0.0
        else:
            dt = max(0.0, timestamp - prev_timestamp)
        speed = 0.0
        transmission = getattr(bundle, "transmission", None)
        if transmission is not None:
            speed = _to_float(getattr(transmission, "speed", 0.0))
        distance += speed * dt
        progress.append(
            {
                "index": float(index),
                "timestamp": float(timestamp),
                "structural": float(structural),
                "distance": float(distance),
            }
        )
        prev_timestamp = timestamp
    return progress


def _microsector_indices(microsector: Any) -> Sequence[int]:
    indices: set[int] = set()
    samples = getattr(microsector, "phase_samples", {}) or {}
    for payload in samples.values():
        try:
            indices.update(int(value) for value in payload)
        except TypeError:
            continue
    if not indices:
        boundaries = getattr(microsector, "phase_boundaries", {}) or {}
        for start, end in boundaries.values():
            start_idx = int(start)
            end_idx = int(end)
            indices.update(range(start_idx, max(start_idx, end_idx)))
    return tuple(sorted(indices))


def _sparkline(series: Sequence[float]) -> str:
    if not series:
        return ""
    minimum = min(series)
    maximum = max(series)
    if math.isclose(maximum, minimum, abs_tol=1e-9):
        return "▁" * len(series)
    span = maximum - minimum
    chars = "▁▂▃▄▅▆▇█"
    buckets = len(chars) - 1
    output = []
    for value in series:
        normalised = max(0.0, min(1.0, (value - minimum) / span))
        index = int(round(normalised * buckets))
        output.append(chars[index])
    return "".join(output)


def build_coherence_map_payload(results: Mapping[str, Any]) -> Dict[str, Any]:
    series = list(_series_from_results(results))
    microsectors = _microsectors_from_results(results)
    progress = list(_track_progress(series))
    index_lookup = {int(entry["index"]): entry for entry in progress}
    entries: list[Dict[str, Any]] = []
    mean_values: list[float] = []
    coherence_extrema: list[float] = []
    distances: list[float] = []

    for microsector in microsectors:
        indices = [index for index in _microsector_indices(microsector) if index < len(series)]
        if not indices:
            continue
        coherence_values: list[float] = []
        delta_values: list[float] = []
        samples: list[Dict[str, float]] = []
        for index in indices:
            bundle = series[index]
            coherence = _to_float(getattr(bundle, "coherence_index", 0.0))
            delta = _to_float(getattr(bundle, "delta_nfr", 0.0))
            coherence_values.append(coherence)
            delta_values.append(delta)
            progress_entry = index_lookup.get(index)
            sample_payload: Dict[str, float] = {
                "index": float(index),
                "coherence": coherence,
                "delta_nfr": delta,
            }
            if progress_entry:
                sample_payload.update(
                    {
                        "timestamp": progress_entry["timestamp"],
                        "structural": progress_entry["structural"],
                        "distance": progress_entry["distance"],
                    }
                )
            samples.append(sample_payload)
        if samples:
            distances.extend(entry.get("distance", 0.0) for entry in samples)
        coherence_mean = mean(coherence_values) if coherence_values else 0.0
        coherence_peak = max(coherence_values) if coherence_values else 0.0
        coherence_floor = min(coherence_values) if coherence_values else 0.0
        coherence_span = coherence_peak - coherence_floor
        delta_peak = max(delta_values) if delta_values else 0.0
        delta_floor = min(delta_values) if delta_values else 0.0
        delta_span = delta_peak - delta_floor
        mean_values.append(coherence_mean)
        coherence_extrema.extend([coherence_peak, coherence_floor])
        context_factors = {}
        raw_context = getattr(microsector, "context_factors", {}) or {}
        for key, value in raw_context.items():
            context_factors[str(key)] = _to_float(value)
        entry = {
            "microsector": int(getattr(microsector, "index", len(entries))),
            "start_time": _to_float(getattr(microsector, "start_time", 0.0)),
            "end_time": _to_float(getattr(microsector, "end_time", 0.0)),
            "phase": str(getattr(microsector, "active_phase", "")),
            "coherence": {
                "mean": coherence_mean,
                "max": coherence_peak,
                "min": coherence_floor,
                "span": coherence_span,
                "series": coherence_values,
            },
            "delta_nfr": {
                "max": delta_peak,
                "min": delta_floor,
                "span": delta_span,
                "series": delta_values,
            },
            "track": {
                "samples": samples,
                "start_distance": samples[0].get("distance", 0.0) if samples else 0.0,
                "end_distance": samples[-1].get("distance", 0.0) if samples else 0.0,
            },
            "context": context_factors,
            "dominant_nodes": {
                str(phase): tuple(nodes)
                for phase, nodes in (getattr(microsector, "dominant_nodes", {}) or {}).items()
            },
        }
        entries.append(entry)

    global_summary: Dict[str, Any] = {
        "mean_coherence": mean(mean_values) if mean_values else 0.0,
        "max_coherence": max(coherence_extrema) if coherence_extrema else 0.0,
        "min_coherence": min(coherence_extrema) if coherence_extrema else 0.0,
    }
    if distances:
        global_summary["distance_span"] = {
            "start": float(min(distances)),
            "end": float(max(distances)),
        }
    else:
        global_summary["distance_span"] = {"start": 0.0, "end": 0.0}

    return {
        "microsectors": entries,
        "global": global_summary,
        "track_progress": progress,
    }


def _structural_for_time(
    lookup: Sequence[Mapping[str, float]],
    timestamp: float,
    *,
    index: int | None = None,
) -> float:
    if index is not None and 0 <= index < len(lookup):
        return float(lookup[index]["structural"])
    if not lookup:
        return float(timestamp)
    closest = min(
        lookup,
        key=lambda entry: abs(entry["timestamp"] - float(timestamp)),
    )
    return float(closest["structural"])


def _segment_delta_metrics(
    series: Sequence[EPIBundle], start_index: int | None, end_index: int | None
) -> Mapping[str, float]:
    if not series:
        return {"mean": 0.0, "peak": 0.0, "span": 0.0}
    if start_index is None:
        start_index = end_index if end_index is not None else 0
    if end_index is None:
        end_index = start_index
    start = max(0, int(start_index))
    stop = min(len(series) - 1, max(start, int(end_index)))
    values = [
        _to_float(getattr(series[index], "delta_nfr", 0.0))
        for index in range(start, stop + 1)
    ]
    if not values:
        return {"mean": 0.0, "peak": 0.0, "span": 0.0}
    peak_index = max(range(len(values)), key=lambda idx: abs(values[idx]))
    peak_value = float(values[peak_index])
    return {
        "mean": float(mean(values)),
        "peak": peak_value,
        "span": float(max(values) - min(values)),
        "start_index": float(start),
        "end_index": float(stop),
    }


def build_operator_trajectories_payload(results: Mapping[str, Any]) -> Dict[str, Any]:
    series = list(_series_from_results(results))
    microsectors = _microsectors_from_results(results)
    lookup = list(_structural_lookup(series))
    events: list[Dict[str, Any]] = []
    summary: MutableMapping[str, Dict[str, float]] = {}
    structural_values: list[float] = []

    for microsector in microsectors:
        raw_events = getattr(microsector, "operator_events", {}) or {}
        base_index = 0
        boundaries = getattr(microsector, "phase_boundaries", {}) or {}
        if boundaries:
            base_index = min(int(start) for start, _ in boundaries.values())
        for name, payloads in raw_events.items():
            operator_code = str(name)
            operator_label = canonical_operator_label(operator_code)
            for payload in payloads:
                start_idx = payload.get("global_start_index")
                if start_idx is None:
                    start_idx = base_index + int(payload.get("start_index", 0))
                end_idx = payload.get("global_end_index")
                if end_idx is None:
                    end_idx = base_index + int(payload.get("end_index", start_idx))
                start_idx = int(start_idx)
                end_idx = int(end_idx)
                start_time = _to_float(payload.get("start_time"))
                end_time = _to_float(payload.get("end_time"), default=start_time)
                structural_start = _structural_for_time(lookup, start_time, index=start_idx)
                structural_end = _structural_for_time(lookup, end_time, index=end_idx)
                duration = max(0.0, end_time - start_time)
                if duration <= 0.0:
                    duration = max(0.0, _to_float(payload.get("duration"), default=0.0))
                delta_metrics = dict(
                    _segment_delta_metrics(series, start_idx, end_idx)
                )
                structural_values.extend([structural_start, structural_end])
                event_entry: Dict[str, Any] = {
                    "code": operator_code,
                    "type": operator_label,
                    "microsector": int(getattr(microsector, "index", 0)),
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "structural_start": structural_start,
                    "structural_end": structural_end,
                    "delta_metrics": delta_metrics,
                    "details": {
                        key: payload[key]
                        for key in sorted(payload)
                        if key not in {"start_index", "end_index"}
                    },
                }
                structural_duration = _to_float(
                    payload.get("structural_duration"), default=0.0
                )
                if structural_duration > 0.0:
                    event_entry["structural_duration"] = structural_duration
                events.append(event_entry)
                stats = summary.setdefault(
                    operator_label,
                    {
                        "code": operator_code,
                        "count": 0.0,
                        "duration": 0.0,
                        "peak": 0.0,
                    },
                )
                stats["count"] += 1.0
                stats["duration"] += duration
                stats["peak"] += abs(delta_metrics.get("peak", 0.0))
                if operator_code == "SILENCIO":
                    quiet_total = stats.setdefault("quiet_duration", 0.0)
                    stats["quiet_duration"] = quiet_total + duration
                    density_total = stats.setdefault("density_total", 0.0)
                    stats["density_total"] = density_total + max(
                        0.0, _to_float(payload.get("structural_density_mean"), default=0.0)
                    )
                    event_entry.setdefault("details", {})["latent_state"] = "SILENCIO"

    events.sort(key=lambda entry: (entry["structural_start"], entry["start_time"]))
    for name, stats in summary.items():
        count = stats.get("count", 0.0) or 1.0
        stats["mean_peak"] = stats.get("peak", 0.0) / count
        stats["count"] = int(round(count))
        total_duration = stats.pop("duration", 0.0)
        stats.pop("peak", None)
        quiet_total = stats.pop("quiet_duration", None)
        density_total = stats.pop("density_total", None)
        stats["mean_duration"] = total_duration / (count or 1.0)
        if quiet_total is not None:
            stats["quiet_duration"] = quiet_total
        if density_total is not None and stats["count"]:
            stats["mean_density"] = density_total / stats["count"]

    if structural_values:
        structural_span = {
            "start": float(min(structural_values)),
            "end": float(max(structural_values)),
        }
    else:
        structural_span = {"start": 0.0, "end": 0.0}

    latent_states: Mapping[str, Any] = {}
    aggregated = results.get("operator_events")
    if isinstance(aggregated, Mapping):
        candidate = aggregated.get("latent_states")
        if isinstance(candidate, Mapping):
            latent_states = {
                canonical_operator_label(str(key)): value
                for key, value in candidate.items()
            }

    return {
        "events": events,
        "summary": summary,
        "structural_span": structural_span,
        "latent_states": latent_states,
    }


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def build_delta_bifurcation_payload(results: Mapping[str, Any]) -> Dict[str, Any]:
    series = list(_series_from_results(results))
    lookup = list(_structural_lookup(series))
    microsectors = results.get("microsectors")
    if not isinstance(microsectors, Sequence):
        microsectors = []
    delta_series: list[Dict[str, float]] = []
    transitions: list[Dict[str, float]] = []
    extrema: list[Dict[str, float]] = []
    derivatives: list[float] = []

    for index, bundle in enumerate(series):
        delta_value = _to_float(getattr(bundle, "delta_nfr", 0.0))
        structural = lookup[index]["structural"]
        entry = {
            "index": float(index),
            "structural": float(structural),
            "timestamp": lookup[index]["timestamp"],
            "delta_nfr": delta_value,
        }
        delta_series.append(entry)
        if index == 0:
            continue
        prev = delta_series[-2]
        dt = max(1e-6, structural - prev["structural"])
        slope = (delta_value - prev["delta_nfr"]) / dt
        derivatives.append(slope)
        if _sign(prev["delta_nfr"]) != _sign(delta_value):
            transitions.append(
                {
                    "index": float(index),
                    "structural": float(structural),
                    "timestamp": entry["timestamp"],
                    "from": prev["delta_nfr"],
                    "to": delta_value,
                    "slope": slope,
                }
            )

    for index in range(1, len(delta_series) - 1):
        previous = delta_series[index - 1]["delta_nfr"]
        current = delta_series[index]["delta_nfr"]
        nxt = delta_series[index + 1]["delta_nfr"]
        if (current > previous and current > nxt) or (
            current < previous and current < nxt
        ):
            extrema.append(
                {
                    "index": delta_series[index]["index"],
                    "structural": delta_series[index]["structural"],
                    "timestamp": delta_series[index]["timestamp"],
                    "delta_nfr": current,
                    "classification": "maximum"
                    if current > previous
                    else "minimum",
                }
            )

    derivative_stats = {
        "max": max(derivatives) if derivatives else 0.0,
        "min": min(derivatives) if derivatives else 0.0,
        "mean": mean(derivatives) if derivatives else 0.0,
        "count": len(derivatives),
    }

    microsector_summary: Dict[str, Dict[str, float]] = {}
    if microsectors and transitions:
        transition_indices = [int(entry["index"]) for entry in transitions]
        for microsector in microsectors:
            indices = set(_microsector_indices(microsector))
            if not indices:
                continue
            matched = [idx for idx in transition_indices if idx in indices]
            if not matched:
                continue
            values = [delta_series[idx]["delta_nfr"] for idx in matched if idx < len(delta_series)]
            label = f"microsector_{int(getattr(microsector, 'index', 0))}"
            microsector_summary[label] = {
                "count": float(len(matched)),
                "mean_delta": mean(values) if values else 0.0,
            }

    return {
        "series": delta_series,
        "transitions": transitions,
        "extrema": extrema,
        "derivative_stats": derivative_stats,
        "microsector_bifurcations": microsector_summary,
    }


def render_coherence_map(payload: Mapping[str, Any], fmt: str = "json") -> str:
    fmt = fmt.lower()
    if fmt == "json":
        return json.dumps(payload, indent=2, sort_keys=True)
    if fmt == "markdown":
        lines = ["# Mapa de coherencia ΔNFR", ""]
        lines.append("| Microsector | Fase | C̄ | Cmax | Cmin | Distancia |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for entry in payload.get("microsectors", []):
            track = entry.get("track", {})
            start_distance = _to_float(track.get("start_distance"))
            end_distance = _to_float(track.get("end_distance"), default=start_distance)
            lines.append(
                "| {microsector} | {phase} | {mean:.3f} | {max_value:.3f} | {min_value:.3f} | {start:.1f}→{end:.1f} |".format(
                    microsector=entry.get("microsector"),
                    phase=entry.get("phase", "-"),
                    mean=_to_float(entry.get("coherence", {}).get("mean")),
                    max_value=_to_float(entry.get("coherence", {}).get("max")),
                    min_value=_to_float(entry.get("coherence", {}).get("min")),
                    start=start_distance,
                    end=end_distance,
                )
            )
        lines.append("")
        global_summary = payload.get("global", {})
        lines.append("## Resumen global")
        lines.append(
            f"- C̄ = {_to_float(global_summary.get('mean_coherence')):.3f}"
        )
        lines.append(
            f"- Cmax = {_to_float(global_summary.get('max_coherence')):.3f}"
        )
        lines.append(
            f"- Cmin = {_to_float(global_summary.get('min_coherence')):.3f}"
        )
        span = global_summary.get("distance_span", {})
        lines.append(
            f"- Cobertura pista = {_to_float(span.get('start')):.1f}→{_to_float(span.get('end')):.1f}"
        )
        return "\n".join(lines)
    if fmt == "visual":
        lines = ["# Coherencia por microsector", ""]
        for entry in payload.get("microsectors", []):
            series = entry.get("coherence", {}).get("series", []) or []
            sparkline = _sparkline([_to_float(value) for value in series])
            lines.append(
                f"MS{entry.get('microsector'):02d} {entry.get('phase', '-')}: "
                f"{sparkline} (C̄ {_to_float(entry.get('coherence', {}).get('mean')):.3f})"
            )
        return "\n".join(lines)
    raise ValueError(f"Formato desconocido para el mapa de coherencia: {fmt}")


def render_operator_trajectories(payload: Mapping[str, Any], fmt: str = "json") -> str:
    fmt = fmt.lower()
    if fmt == "json":
        return json.dumps(payload, indent=2, sort_keys=True)
    if fmt == "markdown":
        lines = ["# Trayectorias de operadores", ""]
        lines.append("| Operador | Microsector | t₀ | t₁ | Δt | ΔNFR pico |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for entry in payload.get("events", []):
            lines.append(
                "| {type} | {microsector} | {start:.2f} | {end:.2f} | {duration:.2f} | {peak:+.3f} |".format(
                    type=entry.get("type"),
                    microsector=entry.get("microsector"),
                    start=_to_float(entry.get("start_time")),
                    end=_to_float(entry.get("end_time")),
                    duration=_to_float(entry.get("duration")),
                    peak=_to_float(entry.get("delta_metrics", {}).get("peak")),
                )
            )
        lines.append("")
        lines.append("## Operator statistics")
        lines.append("| Operator | Events | Mean Δt | Mean peak |")
        lines.append("| --- | --- | --- | --- |")
        for name, stats in sorted(payload.get("summary", {}).items()):
            lines.append(
                "| {name} | {count} | {duration:.2f} | {peak:.3f} |".format(
                    name=name,
                    count=int(stats.get("count", 0)),
                    duration=_to_float(stats.get("mean_duration")),
                    peak=_to_float(stats.get("mean_peak")),
                )
            )
        return "\n".join(lines)
    if fmt == "visual":
        span = payload.get("structural_span", {})
        start = _to_float(span.get("start"))
        end = _to_float(span.get("end"), default=start + 1.0)
        if math.isclose(end, start):
            end = start + 1.0
        width = 42
        lines = ["# Operator timeline", ""]
        for entry in payload.get("events", []):
            start_pos = int((entry["structural_start"] - start) / (end - start) * width)
            end_pos = int((entry["structural_end"] - start) / (end - start) * width)
            start_pos = max(0, min(width - 1, start_pos))
            end_pos = max(start_pos, min(width - 1, end_pos))
            bar = " " * start_pos + "▇" * (end_pos - start_pos + 1)
            lines.append(
                f"{entry['type']:<2} MS{entry['microsector']:02d} |{bar.ljust(width)}|"
            )
        return "\n".join(lines)
    raise ValueError(
        f"Unknown format for operator trajectories: {fmt}"
    )


def render_delta_bifurcation(payload: Mapping[str, Any], fmt: str = "json") -> str:
    fmt = fmt.lower()
    if fmt == "json":
        return json.dumps(payload, indent=2, sort_keys=True)
    if fmt == "markdown":
        lines = ["# ΔNFR bifurcation analysis", ""]
        stats = payload.get("derivative_stats", {})
        lines.append("## Global statistics")
        lines.append(
            f"- Transitions: {len(payload.get('transitions', []))}" \
            f" · Extrema: {len(payload.get('extrema', []))}"
        )
        lines.append(
            f"- Max growth: {_to_float(stats.get('max')):+.3f}"
        )
        lines.append(
            f"- Min growth: {_to_float(stats.get('min')):+.3f}"
        )
        lines.append(
            f"- Mean growth: {_to_float(stats.get('mean')):+.3f}"
        )
        lines.append("")
        if payload.get("transitions"):
            lines.append("## Sign crossings")
            lines.append("| Index | tₛ | ΔNFR₀ | ΔNFR₁ | Slope |")
            lines.append("| --- | --- | --- | --- | --- |")
            for entry in payload["transitions"]:
                lines.append(
                    "| {index:.0f} | {structural:.2f} | {from_val:+.3f} | {to:+.3f} | {slope:+.3f} |".format(
                        index=_to_float(entry.get("index")),
                        structural=_to_float(entry.get("structural")),
                        from_val=_to_float(entry.get("from")),
                        to=_to_float(entry.get("to")),
                        slope=_to_float(entry.get("slope")),
                    )
                )
        if payload.get("microsector_bifurcations"):
            lines.append("")
            lines.append("## Bifurcaciones por microsector")
            lines.append("| Microsector | Eventos | ΔNFR medio |")
            lines.append("| --- | --- | --- |")
            for key, entry in sorted(payload["microsector_bifurcations"].items()):
                lines.append(
                    f"| {key} | {int(entry.get('count', 0.0))} | {_to_float(entry.get('mean_delta')):+.3f} |"
                )
        return "\n".join(lines)
    if fmt == "visual":
        lines = ["# ΔNFR (eje estructural)", ""]
        series = [_to_float(entry.get("delta_nfr")) for entry in payload.get("series", [])]
        sparkline = _sparkline(series)
        if sparkline:
            lines.append(f"Serie: {sparkline}")
        if payload.get("transitions"):
            lines.append("")
            lines.append("Cruces:")
            for entry in payload["transitions"]:
                lines.append(
                    f"· tₛ {entry['structural']:.2f}: {entry['from']:+.3f} → {entry['to']:+.3f}"
                )
        return "\n".join(lines)
    raise ValueError(
        f"Unknown format for the bifurcation analysis: {fmt}"
    )


def coherence_map_exporter(results: Mapping[str, Any]) -> str:
    payload = build_coherence_map_payload(results)
    return render_coherence_map(payload, fmt="json")


def operator_trajectory_exporter(results: Mapping[str, Any]) -> str:
    payload = build_operator_trajectories_payload(results)
    return render_operator_trajectories(payload, fmt="json")


def delta_bifurcation_exporter(results: Mapping[str, Any]) -> str:
    payload = build_delta_bifurcation_payload(results)
    return render_delta_bifurcation(payload, fmt="json")


exporters_registry = {
    "json": json_exporter,
    "csv": csv_exporter,
    "markdown": markdown_exporter,
    "set": lfs_set_exporter,
    "lfs-notes": lfs_notes_exporter,
    "html_ext": html_exporter,
    "coherence-map": coherence_map_exporter,
    "operator-trajectory": operator_trajectory_exporter,
    "delta-bifurcation": delta_bifurcation_exporter,
}

__all__ = [
    "CAR_MODEL_PREFIXES",
    "Exporter",
    "REPORT_ARTIFACT_FORMATS",
    "build_coherence_map_payload",
    "build_operator_trajectories_payload",
    "build_delta_bifurcation_payload",
    "render_coherence_map",
    "render_operator_trajectories",
    "render_delta_bifurcation",
    "coherence_map_exporter",
    "operator_trajectory_exporter",
    "delta_bifurcation_exporter",
    "html_exporter",
    "json_exporter",
    "csv_exporter",
    "markdown_exporter",
    "lfs_notes_exporter",
    "lfs_set_exporter",
    "normalise_set_output_name",
    "resolve_car_prefix",
    "exporters_registry",
]
