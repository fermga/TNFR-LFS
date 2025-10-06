from __future__ import annotations

from dataclasses import asdict, is_dataclass
from html import escape
import math
from typing import Any, Iterable, Mapping, Sequence


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return value
    return None


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)):
        return ()
    if isinstance(value, Sequence):
        return value
    return ()


def _format_float(value: Any, *, decimals: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(numeric):
        return "-"
    template = f"{{:.{decimals}f}}"
    return template.format(numeric)


def _render_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    rendered_rows = list(rows)
    if not rendered_rows:
        return ""
    head_cells = "".join(f"<th>{escape(str(cell))}</th>" for cell in headers)
    body_parts = []
    for row in rendered_rows:
        body_cells = "".join(f"<td>{escape(str(cell))}</td>" for cell in row)
        body_parts.append(f"<tr>{body_cells}</tr>")
    body_html = "".join(body_parts)
    return f"<table><thead><tr>{head_cells}</tr></thead><tbody>{body_html}</tbody></table>"


def _render_list(items: Iterable[str]) -> str:
    entries = [item for item in items if item]
    if not entries:
        return ""
    return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in entries) + "</ul>"


def _render_definition_list(entries: Mapping[str, str]) -> str:
    if not entries:
        return ""
    parts = []
    for key, value in entries.items():
        parts.append(
            f"<dt>{escape(str(key))}</dt><dd>{escape(str(value))}</dd>"
        )
    return "<dl>" + "".join(parts) + "</dl>"


def _session_heading(session: Mapping[str, Any] | None) -> str:
    if not session:
        return "Reporte TNFR × LFS"
    car = session.get("car_model")
    track = session.get("track_profile") or session.get("track_name")
    stint = session.get("session") or session.get("label")
    parts = ["Reporte TNFR × LFS"]
    detail = ", ".join(str(item) for item in (car, track) if item)
    if detail:
        parts.append(detail)
    if stint:
        parts.append(str(stint))
    return " – ".join(parts)


def _render_global_metrics(results: Mapping[str, Any]) -> str:
    metrics: list[tuple[str, Any]] = [
        ("ΔNFR", results.get("delta_nfr")),
        ("Sense Index", results.get("sense_index")),
        ("Disonancia", results.get("dissonance")),
        ("Acoplamiento", results.get("coupling")),
        ("Resonancia", results.get("resonance")),
    ]
    objectives = _as_mapping(results.get("objectives"))
    if objectives:
        metrics.extend(
            (
                f"Objetivo {escape(str(key))}",
                objectives.get(key),
            )
            for key in ("target_delta_nfr", "target_sense_index")
            if key in objectives
        )
    rows = []
    for label, value in metrics:
        formatted = _format_float(value)
        rows.append((label, formatted))
    return _render_table(["Métrica", "Valor"], rows)


def _render_microsector_variability(entries: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        overall = entry.get("overall")
        if not isinstance(overall, Mapping):
            continue
        delta_payload = overall.get("delta_nfr")
        si_payload = overall.get("sense_index")
        if not isinstance(delta_payload, Mapping) or not isinstance(si_payload, Mapping):
            continue
        label = entry.get("label") or f"MS {entry.get('microsector', '?')}"
        samples = overall.get("samples")
        rows.append(
            (
                str(label),
                samples if samples is not None else "-",
                _format_float(delta_payload.get("stdev")),
                _format_float(si_payload.get("stdev")),
                _format_float(si_payload.get("stability_score"), decimals=2),
            )
        )
    if not rows:
        return ""
    headers = [
        "Microsector",
        "Muestras",
        "σ ΔNFR",
        "σ Sense Index",
        "Estabilidad SI",
    ]
    return _render_table(headers, rows)


def _flatten_pairwise(pairwise: Mapping[str, Any]) -> Mapping[str, str]:
    flat: dict[str, str] = {}
    for domain, payload in pairwise.items():
        if not isinstance(payload, Mapping):
            continue
        for pair, value in payload.items():
            if isinstance(pair, str) and pair.endswith("_samples"):
                continue
            flat[f"{domain}: {pair}"] = _format_float(value)
    return flat


def _render_robustness(results: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    variability = _as_sequence(results.get("microsector_variability"))
    variability_rows: Sequence[Mapping[str, Any]] = [
        entry
        for entry in variability
        if isinstance(entry, Mapping)
    ]
    variability_html = _render_microsector_variability(variability_rows)
    if variability_html:
        pieces.append("<h3>Variabilidad por microsector</h3>" + variability_html)
    recursive_trace = _as_sequence(results.get("recursive_trace"))
    if recursive_trace:
        stats = {
            "Mínimo": _format_float(min(recursive_trace)),
            "Máximo": _format_float(max(recursive_trace)),
            "Último": _format_float(recursive_trace[-1]),
        }
        pieces.append("<h3>Memoria recursiva</h3>" + _render_definition_list(stats))
    pairwise = _as_mapping(results.get("pairwise_coupling")) or {}
    flat_pairs = _flatten_pairwise(pairwise)
    if flat_pairs:
        pieces.append("<h3>Acoplamientos nodales</h3>" + _render_definition_list(flat_pairs))
    return "".join(pieces)


def _normalise_pareto_entry(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    if is_dataclass(entry):
        return asdict(entry)
    return entry


def _render_pareto(results: Mapping[str, Any], session: Mapping[str, Any]) -> str:
    pareto_candidates = results.get("pareto_points")
    pareto_entries = []
    if isinstance(pareto_candidates, Sequence) and not isinstance(
        pareto_candidates, (str, bytes, bytearray)
    ):
        pareto_entries.extend(
            [entry for entry in pareto_candidates if isinstance(entry, Mapping)]
        )
    session_pareto = session.get("pareto") if session else None
    if isinstance(session_pareto, Sequence) and not isinstance(
        session_pareto, (str, bytes, bytearray)
    ):
        pareto_entries.extend(
            [entry for entry in session_pareto if isinstance(entry, Mapping)]
        )
    if not pareto_entries:
        return ""
    normalised: list[Mapping[str, Any]] = [
        _normalise_pareto_entry(entry) for entry in pareto_entries
    ]
    breakdown_keys: set[str] = set()
    for entry in normalised:
        breakdown = entry.get("breakdown")
        if isinstance(breakdown, Mapping):
            breakdown_keys.update(str(key) for key in breakdown)
    headers = ["Score"] + sorted(breakdown_keys)
    rows: list[Sequence[str]] = []
    for entry in normalised:
        score = _format_float(entry.get("score"))
        cells = [score]
        breakdown = entry.get("breakdown")
        for key in sorted(breakdown_keys):
            if isinstance(breakdown, Mapping):
                cells.append(_format_float(breakdown.get(key)))
            else:
                cells.append("-")
        rows.append(cells)
    return "<h2>Frente Pareto</h2>" + _render_table(headers, rows)


def _render_abtest(session: Mapping[str, Any], results: Mapping[str, Any]) -> str:
    candidates: list[Mapping[str, Any]] = []
    session_abtest = session.get("abtest") if session else None
    mapping = _as_mapping(session_abtest)
    if mapping:
        candidates.append(mapping)
    standalone = results.get("abtest")
    mapping = _as_mapping(standalone)
    if mapping:
        candidates.append(mapping)
    if not candidates:
        return ""
    payload = candidates[0]
    rows = [
        ("Métrica", payload.get("metric", "-")),
        ("Media baseline", _format_float(payload.get("baseline_mean"))),
        ("Media variante", _format_float(payload.get("variant_mean"))),
        ("Δ media", _format_float(payload.get("mean_difference"))),
        (
            "Intervalo bootstrap",
            f"[{_format_float(payload.get('bootstrap_low'))}, {_format_float(payload.get('bootstrap_high'))}]",
        ),
        (
            "p permutación",
            _format_float(payload.get("permutation_p_value"), decimals=4),
        ),
        ("Potencia", _format_float(payload.get("estimated_power"), decimals=3)),
    ]
    alpha_value = payload.get("alpha")
    if isinstance(alpha_value, (int, float)) and math.isfinite(alpha_value):
        rows.append(("α", _format_float(alpha_value, decimals=3)))
    table = _render_table(["Estadística", "Valor"], rows)
    baseline_laps = _as_sequence(payload.get("baseline_laps"))
    variant_laps = _as_sequence(payload.get("variant_laps"))
    lap_rows = []
    for label, values in (("Baseline", baseline_laps), ("Variante", variant_laps)):
        formatted = ", ".join(_format_float(value) for value in values) if values else "-"
        lap_rows.append((label, formatted))
    laps_table = _render_table(["Grupo", "Laps"], lap_rows)
    section = "<h2>Comparación A/B</h2>" + table
    if laps_table:
        section += laps_table
    return section


def _render_session_messages(messages: Sequence[str]) -> str:
    if not messages:
        return ""
    return "<h2>Perfil de sesión</h2>" + _render_list(messages)


def _render_playbook(session: Mapping[str, Any] | None) -> str:
    if not session:
        return ""
    suggestions = _as_sequence(session.get("playbook_suggestions"))
    text_items = [str(item) for item in suggestions if item]
    if not text_items:
        return ""
    return "<h2>Sugerencias de playbook</h2>" + _render_list(text_items)


def html_exporter(results: Mapping[str, Any]) -> str:
    session = _as_mapping(results.get("session")) or {}
    messages = [str(item) for item in _as_sequence(results.get("session_messages")) if item]
    title = _session_heading(session)
    head = (
        "<!DOCTYPE html><html lang=\"es\"><head><meta charset=\"utf-8\">"
        f"<title>{escape(title)}</title>"
        "<style>body{font-family:system-ui, sans-serif;color:#111;background:#fafafa;margin:0;padding:0;}"
        "main{max-width:960px;margin:0 auto;padding:32px;}h1,h2,h3{color:#123;}table{border-collapse:collapse;width:100%;margin:16px 0;}"
        "th,td{border:1px solid #ccd;padding:8px;text-align:left;}thead{background:#eef;}section{margin-bottom:32px;}"
        "dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;margin:0;}dl dt{font-weight:600;}"
        "ul{margin:0 0 16px 24px;padding:0;}li{margin-bottom:4px;}</style></head>"
    )
    body_parts = [f"<main><h1>{escape(title)}</h1>"]
    global_metrics = _render_global_metrics(results)
    if global_metrics:
        body_parts.append(f"<section><h2>Métricas globales</h2>{global_metrics}</section>")
    robustness = _render_robustness(results)
    if robustness:
        body_parts.append(f"<section><h2>Robustez</h2>{robustness}</section>")
    pareto_section = _render_pareto(results, session)
    if pareto_section:
        body_parts.append(f"<section>{pareto_section}</section>")
    abtest_section = _render_abtest(session, results)
    if abtest_section:
        body_parts.append(f"<section>{abtest_section}</section>")
    playbook_section = _render_playbook(session)
    if playbook_section:
        body_parts.append(f"<section>{playbook_section}</section>")
    session_section = _render_session_messages(messages)
    if session_section:
        body_parts.append(f"<section>{session_section}</section>")
    body_parts.append("</main>")
    body = "".join(body_parts)
    return head + body + "</html>"


__all__ = ["html_exporter"]
