"""Phase 9.C — Rule-learning hook (frequentist re-ranking).

Given a history of past :class:`StintComparison` runs, re-rank the
canonical ``TRIGGER_RULES`` so that empirically-successful rules
appear earlier in the advisor's evaluation order. The original rule
set is **never** trimmed: this is a re-ordering only — every
canonical rule remains a hard constraint and is still evaluated.

Determinism: the re-rank is a stable sort keyed on success rate +
mean realised coherence change, with the original index as the
tie-breaker, so the function is bytewise-deterministic for any input
history.

This is an API surface for offline / opt-in usage. The default
:class:`SetupAdvisor` continues to use the canonical ordering until a
caller explicitly substitutes the ranked tuple.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .multi_stint import StintComparison
from .operators import TRIGGER_RULES, TriggerRule


@dataclass(frozen=True)
class RuleOutcome:
    """Aggregated empirical outcome for one rule across many stints."""

    rule_name: str
    fires: int                  # how many before-stints proposed this rule
    positive_coherence: int     # of those, how many had Δcoherence > 0
    mean_coherence_change: float
    mean_lap_time_change_ms: float | None


def aggregate_outcomes(
    history: Sequence[StintComparison],
) -> Mapping[str, RuleOutcome]:
    """Compute per-rule outcome statistics from a comparison history."""
    fires: dict[str, int] = {}
    positives: dict[str, int] = {}
    coh_sum: dict[str, float] = {}
    lap_sum: dict[str, float] = {}
    lap_n: dict[str, int] = {}

    for cmp in history:
        if cmp.before.proposed is None:
            continue
        # Each action carries its originating rule's name in
        # ``rationale_id``; deduplicate per stint to avoid weighting
        # multi-target rules more than single-target ones.
        rule_ids = {a.rationale_id for a in cmp.before.proposed.actions}
        for rid in rule_ids:
            fires[rid] = fires.get(rid, 0) + 1
            coh_sum[rid] = coh_sum.get(rid, 0.0) + cmp.coherence_change
            if cmp.coherence_change > 0.0:
                positives[rid] = positives.get(rid, 0) + 1
            if cmp.lap_time_change_ms is not None:
                lap_sum[rid] = lap_sum.get(rid, 0.0) + cmp.lap_time_change_ms
                lap_n[rid] = lap_n.get(rid, 0) + 1

    outcomes: dict[str, RuleOutcome] = {}
    for rid, n in fires.items():
        mean_lap = lap_sum[rid] / lap_n[rid] if lap_n.get(rid) else None
        outcomes[rid] = RuleOutcome(
            rule_name=rid,
            fires=n,
            positive_coherence=positives.get(rid, 0),
            mean_coherence_change=coh_sum[rid] / n,
            mean_lap_time_change_ms=mean_lap,
        )
    return outcomes


def rank_rules_by_outcome(
    history: Sequence[StintComparison],
    base_rules: Sequence[TriggerRule] = TRIGGER_RULES,
    *,
    smoothing: float = 1.0,
) -> tuple[TriggerRule, ...]:
    """Return ``base_rules`` re-ordered by empirical success rate.

    Rules with no history retain their original position relative to
    each other (stable sort). Rules with history are scored as::

        score = (positive + smoothing) / (fires + 2 * smoothing)
              + mean_coherence_change

    Laplace smoothing keeps small samples conservative. The result is
    sorted by score DESC, then original index ASC.
    """
    outcomes = aggregate_outcomes(history)

    def score(rule: TriggerRule) -> float:
        o = outcomes.get(rule.name)
        if o is None or o.fires == 0:
            return 0.0
        rate = (o.positive_coherence + smoothing) / \
            (o.fires + 2.0 * smoothing)
        return float(rate + o.mean_coherence_change)

    indexed = list(enumerate(base_rules))
    indexed.sort(key=lambda kv: (-score(kv[1]), kv[0]))
    return tuple(r for _, r in indexed)


def try_engine_ranking(
    history: Sequence[StintComparison],
    base_rules: Sequence[TriggerRule] = TRIGGER_RULES,
) -> tuple[TriggerRule, ...] | None:
    """Optional hook into ``tnfr.dynamics.self_optimizing_engine``.

    Returns ``None`` if the symbol is not present in the installed
    ``tnfr`` package. The fallback is :func:`rank_rules_by_outcome`,
    which callers should invoke explicitly when this returns ``None``.
    """
    try:
        from tnfr.dynamics import self_optimizing_engine  # type: ignore
    except Exception:
        return None
    # The current engine API is exploratory; we wrap it conservatively.
    try:
        scores = self_optimizing_engine(  # type: ignore[call-arg]
            [
                {
                    "rule": rid,
                    "delta_c": cmp.coherence_change,
                    "delta_t_ms": cmp.lap_time_change_ms,
                }
                for cmp in history
                if cmp.before.proposed is not None
                for rid in {
                    a.rationale_id
                    for a in cmp.before.proposed.actions
                }
            ],
        )
    except Exception:
        return None
    if not isinstance(scores, Mapping):
        return None
    indexed = list(enumerate(base_rules))
    indexed.sort(
        key=lambda kv: (-float(scores.get(kv[1].name, 0.0)), kv[0]),
    )
    return tuple(r for _, r in indexed)


__all__ = (
    "RuleOutcome",
    "aggregate_outcomes",
    "rank_rules_by_outcome",
    "try_engine_ranking",
)
