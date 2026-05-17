"""Phase 9.C — rule_learning re-ranker tests."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from lfs_telemetry.telemetry.lap import LapTelemetry
from lfs_telemetry.tnfr_racing.advisor import SetupAdvisor
from lfs_telemetry.tnfr_racing.multi_stint import (
    StintComparison, compare_stints,
)
from lfs_telemetry.tnfr_racing.operators import TRIGGER_RULES
from lfs_telemetry.tnfr_racing.rule_learning import (
    aggregate_outcomes, rank_rules_by_outcome,
)

from .test_advisor import _synthetic_baseline

ASSETS = Path(__file__).resolve().parents[2] / "assets"


def _real_comparison() -> StintComparison:
    paths = sorted(ASSETS.glob("synthetic_BL1_FBM_v2_lap*.csv"))
    laps = [LapTelemetry.from_csv(p) for p in paths]
    baseline = _synthetic_baseline()
    return compare_stints(
        before_laps=laps, after_laps=laps,
        baseline_before=baseline, baseline_after=baseline,
        car=laps[0].car, track_code="BL1",
    )


def test_empty_history_preserves_order() -> None:
    ranked = rank_rules_by_outcome([], TRIGGER_RULES)
    assert ranked == TRIGGER_RULES


def test_rank_is_deterministic() -> None:
    cmp = _real_comparison()
    a = rank_rules_by_outcome([cmp, cmp], TRIGGER_RULES)
    b = rank_rules_by_outcome([cmp, cmp], TRIGGER_RULES)
    assert a == b
    # Hard-constraint guarantee: no rule is dropped.
    assert set(a) == set(TRIGGER_RULES)
    assert len(a) == len(TRIGGER_RULES)


def test_aggregate_outcomes_counts_fires() -> None:
    cmp = _real_comparison()
    outcomes = aggregate_outcomes([cmp, cmp, cmp])
    if cmp.before.proposed is None:
        # Refusal -> nothing to aggregate.
        assert outcomes == {}
        return
    rule_ids = {a.rationale_id for a in cmp.before.proposed.actions}
    for rid in rule_ids:
        assert outcomes[rid].fires == 3
        # ΔC = 0 on self-comparison -> not counted as positive.
        assert outcomes[rid].positive_coherence == 0


def test_positive_history_promotes_rule() -> None:
    """A rule with synthetic +ΔC history should rank above one with 0."""
    cmp = _real_comparison()
    if cmp.before.proposed is None or not cmp.before.proposed.actions:
        pytest.skip("self-comparison did not yield a proposal to learn from")
    # Build two fake comparisons: one with positive ΔC, one with zero.
    good = replace(cmp, coherence_change=+0.10)
    bad = replace(cmp, coherence_change=0.0)
    history = [good, good, good, bad]
    ranked = rank_rules_by_outcome(history, TRIGGER_RULES)
    # The rule(s) that fired now appear no later than they did before
    # (re-rank is monotone with positive ΔC, never demoted).
    fired_ids = {a.rationale_id for a in cmp.before.proposed.actions}
    base_index = {r.name: i for i, r in enumerate(TRIGGER_RULES)}
    new_index = {r.name: i for i, r in enumerate(ranked)}
    for rid in fired_ids:
        assert new_index[rid] <= base_index[rid]
