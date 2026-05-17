"""Pure (Qt-free) serializers for :class:`AdvisorResult`.

Both the Studio UI (:mod:`lfs_telemetry.studio.widgets.setup_advisor_tab`)
and the CLI (``python -m lfs_telemetry advise``) render advisor output
through the helpers in this module so the on-screen text, the exported
Markdown report and the JSON payload stay in lock-step.

All textual output is jargon-filtered (see ``docs/TNFR_SETUP_ADVISOR.md``
§10.3): only physical engineering terms reach the user.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict

from .advisor import AdvisorResult, ProposedSetup
from .multi_stint import StintComparison
from .operators import ConsolidatedAdjustment, PhysicalAction, SetupSynthesis, synthesize_actions  # noqa: F401  (synthesize_actions kept for legacy callers)
from .sensitivities import SensitivityEstimate


# ---------------------------------------------------------------------------
# Translation tables (TNFR-free physical language only)
# ---------------------------------------------------------------------------

_ACTION_KIND_LABEL: Dict[str, str] = {
    "damper_rebound": "Damper rebound",
    "damper_bump": "Damper bump",
    "spring": "Spring rate",
    "arb": "Anti-roll bar",
    "tyre_pressure": "Tyre pressure",
    "brake_bias": "Brake bias (% front)",
    "camber": "Camber",
    "toe": "Toe-in",
    "ride_height": "Ride height",
}

_TARGET_LABEL: Dict[str, str] = {
    "FL": "front-left", "FR": "front-right",
    "RL": "rear-left", "RR": "rear-right",
    "front": "front axle", "rear": "rear axle",
    "global": "whole car",
}


def humanize_action(
    act: PhysicalAction,
    sensitivity: SensitivityEstimate | None = None,
) -> tuple[str, str, str]:
    """Return ``(subsystem, change, rationale)`` text for an action.

    If a :class:`SensitivityEstimate` is provided, its sentence is
    appended to the rationale column.
    """
    kind_label = _ACTION_KIND_LABEL.get(act.kind, act.kind)
    tgt_label = _TARGET_LABEL.get(act.target, act.target)
    sub = f"{kind_label} ({tgt_label})"
    sign = "+" if act.delta >= 0 else ""
    change = f"{sign}{act.delta:g} {act.units}"
    rationale = act.rationale_id.replace("_", " ")
    if sensitivity is not None and sensitivity.sentence:
        rationale = f"{rationale} - {sensitivity.sentence}"
    return sub, change, rationale


def format_refusal(reason: str) -> str:
    """Translate an internal refusal tag into a physical sentence."""
    if reason.startswith("insufficient_stint"):
        tail = reason.split(":", 1)[-1].strip()
        return ("Need >= 5 consecutive valid laps from the same stint."
                + (f" {tail}" if tail else ""))
    if reason.startswith("no_rule_fired"):
        return ("Global coherence is already inside the target band;"
                " no setup action is recommended for this stint.")
    if reason.startswith("grammar_U_violation"):
        return ("Proposed setup changes would violate canonical"
                " stability constraints; refusing to recommend.")
    if reason.startswith("no_coherence_positive_rule"):
        return ("No retained change increases the projected"
                " coherence; refusing to recommend.")
    if reason.startswith("sector_decomposition_failed"):
        return ("Could not decompose the lap into sectors; check"
                " distance and lap markers in the telemetry.")
    return reason


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


def result_to_json(result: AdvisorResult) -> str:
    """Serialize an :class:`AdvisorResult` to a deterministic JSON string."""
    diagnostics = [
        {"key": d.key, "message": d.message,
         "value": d.value, "units": d.units}
        for d in result.diagnostics
    ]
    if result.proposed is None:
        payload = {
            "status": "refusal",
            "refusal_reason": result.refusal_reason or "",
            "refusal_message": format_refusal(result.refusal_reason or ""),
            "diagnostics": diagnostics,
        }
    else:
        p: ProposedSetup = result.proposed
        synthesis = p.synthesis if p.synthesis is not None \
            else synthesize_actions(tuple(p.actions))
        payload = {
            "status": "proposal",
            "actions": [asdict(a) for a in p.actions],
            "consolidated_setup": [
                {
                    "kind": adj.kind,
                    "target": adj.target,
                    "net_delta": adj.net_delta,
                    "units": adj.units,
                    "contributing_rules": list(adj.contributing_rules),
                    "confidence": adj.confidence,
                }
                for adj in synthesis.adjustments
            ],
            "conflict_groups": [
                {
                    "kind": k, "target": t,
                    "rules_up": list(u), "rules_down": list(d),
                }
                for (k, t, u, d) in synthesis.conflict_groups
            ],
            "expected_coherence_delta": p.expected_coherence_delta,
            "expected_lap_time_delta_ms": p.expected_lap_time_delta_ms,
            "grammar_passed": p.grammar_passed,
            "seed": p.seed,
            "baseline_hash": p.baseline_hash,
            "stint_signature": p.stint_signature,
            "coherence_before": p.coherence_before,
            "coherence_after": p.coherence_after,
            "diagnostics": diagnostics,
        }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def result_to_markdown(
    result: AdvisorResult,
    *,
    car_key: str = "",
    track_code: str = "",
    n_laps: int | None = None,
    baseline=None,
) -> str:
    """Serialize an :class:`AdvisorResult` to a Markdown report."""
    lines: list[str] = []
    title = "TNFR Setup Advisor — proposal"
    if result.proposed is None:
        title = "TNFR Setup Advisor — no safe recommendation"
    header_tail: list[str] = []
    if car_key:
        header_tail.append(car_key)
    if track_code:
        header_tail.append(f"@ {track_code}")
    if n_laps is not None:
        header_tail.append(f"({n_laps}-lap stint)")
    if header_tail:
        title = f"{title} — {' '.join(header_tail)}"
    lines.append(f"# {title}")
    lines.append("")

    if result.proposed is None:
        lines.append("**No safe recommendation.**")
        lines.append("")
        lines.append(format_refusal(result.refusal_reason or ""))
        lines.append("")
    else:
        p = result.proposed
        lines.append(
            f"- Projected coherence: **+{p.expected_coherence_delta:.3f}**"
            f" ({p.coherence_before:.3f} -> {p.coherence_after:.3f})"
        )
        lines.append(
            f"- Stability constraints: "
            f"{'pass' if p.grammar_passed else 'fail'}"
        )
        lines.append(
            f"- Seed: `{p.seed}` - baseline `{p.baseline_hash}`"
            f" - stint `{p.stint_signature}`"
        )
        lines.append("")
        if p.actions:
            lines.append("## Proposed changes")
            lines.append("")
            lines.append("| # | Subsystem | Change | Why |")
            lines.append("|---|-----------|--------|-----|")
            for i, act in enumerate(p.actions, 1):
                sens = None
                if baseline is not None:
                    try:
                        from .sensitivities import (
                            estimate_action_sensitivity,
                        )
                        sens = estimate_action_sensitivity(act, baseline)
                    except Exception:
                        sens = None
                sub, change, rationale = humanize_action(act, sens)
                lines.append(f"| {i} | {sub} | {change} | {rationale} |")
            lines.append("")

            # ---- Consolidated optimal setup -----------------------
            synthesis = p.synthesis if p.synthesis is not None \
                else synthesize_actions(tuple(p.actions))
            lines.extend(_synthesis_section(synthesis))

    if result.diagnostics:
        lines.append("## Diagnostics")
        lines.append("")
        for d in result.diagnostics:
            v = f" = {d.value:.3f}" if d.value is not None else ""
            u = f" {d.units}" if d.units else ""
            lines.append(f"- {d.message}{v}{u}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _format_consolidated_delta(adj: ConsolidatedAdjustment) -> str:
    sign = "+" if adj.net_delta >= 0 else ""
    return f"{sign}{adj.net_delta:g} {adj.units}"


def _synthesis_section(synthesis: SetupSynthesis) -> list[str]:
    """Render the consolidated optimal-setup block.

    Aggregates every rule action that targets the same channel
    ``(kind, target)`` and reports the net delta + which rules
    contributed + confidence. Conflicts (mixed signs on one channel)
    are flagged in their own subsection — they signal an ambiguous
    structural field where the driver should treat the net delta as
    a weak signal rather than a directive.
    """
    lines: list[str] = []
    if not synthesis.adjustments:
        return lines
    lines.append("## Consolidated optimal setup")
    lines.append("")
    lines.append(
        "Aggregated net delta per channel — every individual "
        "recommendation above contributes to this single coherent "
        "setup proposal:"
    )
    lines.append("")
    lines.append("| Subsystem | Net change | Confidence | Driven by |")
    lines.append("|-----------|-----------:|-----------:|-----------|")
    for adj in synthesis.adjustments:
        kind_label = _ACTION_KIND_LABEL.get(adj.kind, adj.kind)
        tgt_label = _TARGET_LABEL.get(adj.target, adj.target)
        sub = f"{kind_label} ({tgt_label})"
        delta = _format_consolidated_delta(adj)
        n = len(adj.contributing_rules)
        rules_txt = f"{n} signal" if n == 1 else f"{n} signals"
        stars = "*" * adj.confidence + "-" * (5 - adj.confidence)
        lines.append(f"| {sub} | **{delta}** | {stars} | {rules_txt} |")
    lines.append("")
    if synthesis.conflict_groups:
        lines.append("### Conflicting signals")
        lines.append("")
        lines.append(
            "These channels received opposing recommendations — the "
            "net delta is the best compromise but treat it as a weak "
            "signal until the underlying balance is resolved:"
        )
        lines.append("")
        for kind, target, ups, downs in synthesis.conflict_groups:
            kind_label = _ACTION_KIND_LABEL.get(kind, kind)
            tgt_label = _TARGET_LABEL.get(target, target)
            lines.append(
                f"- **{kind_label} ({tgt_label})** — "
                f"{len(ups)} push up vs {len(downs)} push down"
            )
        lines.append("")
    return lines


__all__ = [
    "humanize_action",
    "format_refusal",
    "result_to_json",
    "result_to_markdown",
    "comparison_to_json",
    "comparison_to_markdown",
    "synthesis_to_html",
]


def synthesis_to_html(
    synthesis: SetupSynthesis | None,
    *,
    muted_color: str = "#888",
) -> str:
    """Render a :class:`SetupSynthesis` as a Studio-friendly HTML block.

    Returns ``""`` when the synthesis is missing or empty. Uses the
    same physical-language label tables as :func:`_synthesis_section`
    so the UI and the Markdown export never drift.
    """
    if synthesis is None or not synthesis.adjustments:
        return ""
    th = ("padding:2px 10px;background:#222;color:#cfd;"
          "text-align:left;font-weight:normal;")
    td = "padding:2px 10px;border-top:1px solid #2a2a2a;"
    head = (
        f"<tr><th style='{th}'>Subsystem</th>"
        f"<th style='{th}'>Net change</th>"
        f"<th style='{th}'>Confidence</th>"
        f"<th style='{th}'>Driven by</th></tr>"
    )
    rows: list[str] = []
    for adj in synthesis.adjustments:
        kind_label = _ACTION_KIND_LABEL.get(adj.kind, adj.kind)
        tgt_label = _TARGET_LABEL.get(adj.target, adj.target)
        sub = f"{kind_label} ({tgt_label})"
        delta = _format_consolidated_delta(adj)
        n = len(adj.contributing_rules)
        rules_txt = f"{n} signal" if n == 1 else f"{n} signals"
        stars = "★" * adj.confidence + "·" * (5 - adj.confidence)
        rows.append(
            f"<tr><td style='{td}'>{sub}</td>"
            f"<td style='{td}'><b>{delta}</b></td>"
            f"<td style='{td}'>{stars}</td>"
            f"<td style='{td}'>{rules_txt}</td></tr>"
        )
    blurb = (
        "Aggregated net delta per channel — every individual "
        "recommendation above contributes to this single coherent "
        "setup proposal."
    )
    html = [
        "<h3 style='margin:10px 0 4px 0;'>Consolidated optimal setup</h3>",
        f"<p style='color:{muted_color};margin:0 0 4px 0;'>{blurb}</p>",
        "<table style='border-collapse:collapse;min-width:560px;'>",
        head, *rows, "</table>",
    ]
    if synthesis.conflict_groups:
        html.append(
            "<h4 style='margin:8px 0 4px 0;'>Conflicting signals</h4>"
        )
        html.append(
            f"<p style='color:{muted_color};margin:0 0 4px 0;'>"
            "These channels received opposing recommendations — the "
            "net delta is the best compromise but treat it as a weak "
            "signal until the underlying balance is resolved."
            "</p>"
        )
        items: list[str] = []
        for kind, target, ups, downs in synthesis.conflict_groups:
            kind_label = _ACTION_KIND_LABEL.get(kind, kind)
            tgt_label = _TARGET_LABEL.get(target, target)
            items.append(
                f"<li><b>{kind_label} ({tgt_label})</b> — "
                f"{len(ups)} push up vs {len(downs)} push down</li>"
            )
        html.append(
            "<ul style='margin:0 0 6px 18px;'>" + "".join(items) + "</ul>"
        )
    return "".join(html)


# ---------------------------------------------------------------------------
# Stint-comparison serializers (Phase 9.A)
# ---------------------------------------------------------------------------


def _comparison_payload(c: StintComparison) -> dict:
    return {
        "headline": c.headline,
        "coherence_before": c.coherence_before,
        "coherence_after": c.coherence_after,
        "coherence_change": c.coherence_change,
        "median_lap_time_before_ms": c.median_lap_time_before_ms,
        "median_lap_time_after_ms": c.median_lap_time_after_ms,
        "lap_time_change_ms": c.lap_time_change_ms,
        "proposed_actions_validated": c.proposed_actions_validated,
        "proposed_actions_total": c.proposed_actions_total,
        "diagnostics": [
            {"key": d.key, "message": d.message,
             "value": d.value, "units": d.units}
            for d in c.diagnostics
        ],
        "before": json.loads(result_to_json(c.before)),
        "after": json.loads(result_to_json(c.after)),
    }


def comparison_to_json(c: StintComparison) -> str:
    """Serialize a :class:`StintComparison` to deterministic JSON."""
    return json.dumps(_comparison_payload(c), indent=2, ensure_ascii=False)


def comparison_to_markdown(
    c: StintComparison,
    *,
    car_key: str = "",
    track_code: str = "",
) -> str:
    """Serialize a :class:`StintComparison` to a Markdown report."""
    head: list[str] = ["TNFR Setup Advisor - stint comparison"]
    tail = [s for s in (car_key, f"@ {track_code}" if track_code else "")
            if s]
    if tail:
        head.append("- ".join(tail))
    lines: list[str] = [f"# {' '.join(head)}", "", c.headline, ""]

    lines.append("## Empirical change")
    lines.append("")
    lines.append(
        f"- Global coherence: {c.coherence_before:.3f} -> "
        f"{c.coherence_after:.3f} (**{c.coherence_change:+.3f}**)"
    )
    if (c.median_lap_time_before_ms is not None
            and c.median_lap_time_after_ms is not None):
        lines.append(
            f"- Median lap span: "
            f"{c.median_lap_time_before_ms / 1000.0:.3f}s -> "
            f"{c.median_lap_time_after_ms / 1000.0:.3f}s "
            f"(**{(c.lap_time_change_ms or 0.0) / 1000.0:+.3f}s**)"
        )
    if c.proposed_actions_total > 0:
        lines.append(
            f"- Proposed changes resolved by the new setup: "
            f"**{c.proposed_actions_validated}/"
            f"{c.proposed_actions_total}**"
        )
    lines.append("")
    if c.diagnostics:
        lines.append("## Diagnostics")
        lines.append("")
        for d in c.diagnostics:
            v = f" = {d.value:.3f}" if d.value is not None else ""
            u = f" {d.units}" if d.units else ""
            lines.append(f"- {d.message}{v}{u}")
        lines.append("")
    # Consolidated optimal setup derived from the *after* stint: what
    # the advisor recommends as the next coherent step given the
    # current car state. Rendered here so the comparison report is
    # self-contained without having to also dump the full per-stint
    # advisor outputs.
    after_proposed = c.after.proposed if c.after is not None else None
    after_synthesis = (
        after_proposed.synthesis
        if after_proposed is not None and after_proposed.synthesis is not None
        else None
    )
    if after_synthesis is not None and after_synthesis.adjustments:
        synth_lines = _synthesis_section(after_synthesis)
        # Promote the section heading so it reads as a next-step
        # recommendation in the comparison context.
        if synth_lines and synth_lines[0] == "## Consolidated optimal setup":
            synth_lines[0] = "## Next-step optimal setup"
        lines.extend(synth_lines)
    return "\n".join(lines).rstrip() + "\n"
