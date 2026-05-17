"""Phase 9.A — Multi-stint comparison.

Compare two stints (typically *before* and *after* a setup change) and
report whether the prior :class:`SetupAdvisor` proposal materialised
empirically: did the recommended changes actually move global
coherence upward and lap-times downward?

Pure / deterministic. No Qt, no I/O. Two calls with identical inputs
always yield bytewise-equal :class:`StintComparison` instances.

Usage::

    from lfs_telemetry.tnfr_racing.advisor import SetupAdvisor
    from lfs_telemetry.tnfr_racing.multi_stint import compare_stints

    cmp = compare_stints(
        before_laps=laps_old_setup,
        after_laps=laps_new_setup,
        baseline_before=car_info_bin_old,
        baseline_after=car_info_bin_new,
        car=car_spec,
        track_code="BL1",
        seed=20260516,
    )
    print(cmp.headline)               # one-line physical summary
    print(cmp.coherence_change)       # +0.041
    print(cmp.lap_time_change_ms)     # -312.0
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from lfs_telemetry.telemetry.car_info_bin import CarInfoBin
from lfs_telemetry.telemetry.lap import LapTelemetry
from lfs_telemetry.telemetry.observables import CarSpec

from .advisor import AdvisorResult, Diagnostic, SetupAdvisor

_DEFAULT_SEED = 20260516


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StintComparison:
    """Outcome of comparing two stints back-to-back."""

    before: AdvisorResult
    after: AdvisorResult

    coherence_before: float
    coherence_after: float
    coherence_change: float

    median_lap_time_before_ms: Optional[float]
    median_lap_time_after_ms: Optional[float]
    lap_time_change_ms: Optional[float]

    proposed_actions_validated: int
    proposed_actions_total: int

    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)
    headline: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _median_lap_time_ms(laps: Sequence[LapTelemetry]) -> Optional[float]:
    """Median lap time across a stint, in milliseconds.

    Reads ``time_ms`` from the raw frame and uses the end-minus-start
    span per lap. Returns ``None`` if no lap exposes the column.
    """
    spans: list[float] = []
    for lap in laps:
        df = lap.raw
        if df is None or df.empty or "time_ms" not in df.columns:
            continue
        t = pd.to_numeric(df["time_ms"], errors="coerce").dropna()
        if t.empty:
            continue
        spans.append(float(t.iloc[-1] - t.iloc[0]))
    if not spans:
        return None
    return float(pd.Series(spans).median())


def _coherence_of(result: AdvisorResult) -> Optional[float]:
    """Use the *before* coherence of the advisor run as the empirical C.

    The advisor measures global coherence *of the stint as captured*
    before applying any hypothetical change. That is exactly the
    measured value for the setup that was on the car during that
    stint, so it is the right observable for an A/B comparison.
    """
    if result.proposed is not None:
        return float(result.proposed.coherence_before)
    # Refusal case: peek into diagnostics for the "coherence_before"
    # entry that the advisor may have emitted (e.g. no_rule_fired
    # refusal carries it).
    for d in result.diagnostics:
        if d.key == "coherence_before" and d.value is not None:
            return float(d.value)
    return None


def _count_validated_actions(
    before: AdvisorResult, after: AdvisorResult,
) -> tuple[int, int]:
    """How many proposed actions appear to have been retained after.

    Heuristic: a proposed action *from before* is considered
    "validated" if the same ``rationale_id`` (or none at all) fires
    again *after*. The intent is to spot rules that keep firing under
    the new setup — i.e. the change did not solve the underlying
    problem.
    """
    if before.proposed is None:
        return (0, 0)
    proposed_ids = {a.rationale_id for a in before.proposed.actions}
    if not proposed_ids:
        return (0, 0)
    after_ids: set[str] = set()
    if after.proposed is not None:
        after_ids = {a.rationale_id for a in after.proposed.actions}
    # "Validated" = no longer firing on the new setup.
    validated = sum(1 for rid in proposed_ids if rid not in after_ids)
    return (validated, len(proposed_ids))


def _headline(
    coh_change: float,
    lap_change_ms: Optional[float],
    validated: int,
    total: int,
) -> str:
    coh_part = (
        f"coherence improved by {coh_change:+.3f}" if coh_change > 0.0
        else f"coherence decreased by {coh_change:+.3f}" if coh_change < 0.0
        else "coherence unchanged"
    )
    if lap_change_ms is None:
        lap_part = "lap time unchanged (no time signal)"
    elif lap_change_ms < 0:
        lap_part = f"median lap {-lap_change_ms / 1000:.3f}s faster"
    elif lap_change_ms > 0:
        lap_part = f"median lap {lap_change_ms / 1000:.3f}s slower"
    else:
        lap_part = "median lap unchanged"
    if total > 0:
        val_part = (
            f"{validated}/{total} proposed change(s) no longer needed"
        )
    else:
        val_part = "no prior proposal to validate"
    return f"{coh_part}; {lap_part}; {val_part}."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_stints(
    *,
    before_laps: Sequence[LapTelemetry],
    after_laps: Sequence[LapTelemetry],
    baseline_before: CarInfoBin,
    baseline_after: CarInfoBin,
    car: CarSpec,
    track_code: str,
    seed: int = _DEFAULT_SEED,
) -> StintComparison:
    """Run the advisor on both stints and aggregate the deltas."""
    advisor = SetupAdvisor(seed=int(seed))
    res_before = advisor.advise(
        list(before_laps), baseline_before, car, track_code,
    )
    res_after = advisor.advise(
        list(after_laps), baseline_after, car, track_code,
    )
    return _build_comparison(
        res_before, res_after, before_laps, after_laps,
    )


def _build_comparison(
    before: AdvisorResult,
    after: AdvisorResult,
    before_laps: Sequence[LapTelemetry],
    after_laps: Sequence[LapTelemetry],
) -> StintComparison:
    c_before = _coherence_of(before)
    c_after = _coherence_of(after)
    if c_before is None or c_after is None:
        # Without a measurable coherence on either side we cannot
        # quantify the comparison; surface a documented diagnostic.
        return StintComparison(
            before=before, after=after,
            coherence_before=c_before or 0.0,
            coherence_after=c_after or 0.0,
            coherence_change=0.0,
            median_lap_time_before_ms=_median_lap_time_ms(before_laps),
            median_lap_time_after_ms=_median_lap_time_ms(after_laps),
            lap_time_change_ms=None,
            proposed_actions_validated=0,
            proposed_actions_total=0,
            diagnostics=(
                Diagnostic(
                    key="comparison_incomplete",
                    message=(
                        "One or both stints did not yield a measurable "
                        "global coherence; comparison is undefined."
                    ),
                ),
            ),
            headline=(
                "Comparison undefined: missing coherence on at least one "
                "stint."
            ),
        )

    coh_change = float(c_after - c_before)
    lap_before = _median_lap_time_ms(before_laps)
    lap_after = _median_lap_time_ms(after_laps)
    lap_change = (
        float(lap_after - lap_before)
        if lap_before is not None and lap_after is not None
        else None
    )
    validated, total = _count_validated_actions(before, after)

    diagnostics: list[Diagnostic] = [
        Diagnostic(
            key="coherence_before",
            message="Global structural coherence of the before stint.",
            value=c_before,
        ),
        Diagnostic(
            key="coherence_after",
            message="Global structural coherence of the after stint.",
            value=c_after,
        ),
        Diagnostic(
            key="coherence_change",
            message=(
                "Empirical change in global coherence attributable to "
                "the setup difference between the two stints."
            ),
            value=coh_change,
        ),
    ]
    if lap_change is not None:
        diagnostics.append(Diagnostic(
            key="lap_time_change_ms",
            message=(
                "Difference of median lap-time spans (after minus "
                "before)."
            ),
            value=lap_change, units="ms",
        ))
    if total > 0:
        diagnostics.append(Diagnostic(
            key="proposed_actions_validated",
            message=(
                "Number of previously-proposed changes that no longer "
                "fire under the new setup."
            ),
            value=float(validated),
        ))

    return StintComparison(
        before=before, after=after,
        coherence_before=c_before,
        coherence_after=c_after,
        coherence_change=coh_change,
        median_lap_time_before_ms=lap_before,
        median_lap_time_after_ms=lap_after,
        lap_time_change_ms=lap_change,
        proposed_actions_validated=validated,
        proposed_actions_total=total,
        diagnostics=tuple(diagnostics),
        headline=_headline(coh_change, lap_change, validated, total),
    )


__all__ = (
    "StintComparison",
    "compare_stints",
)
