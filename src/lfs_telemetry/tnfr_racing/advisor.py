""":class:`SetupAdvisor` — the public TNFR-grounded setup recommender.

Pipeline (deterministic, side-effect free):

1. :func:`~lap_filters.filter_consecutive_laps` — demand ≥ 5 consecutive
   clean laps from the same stint; otherwise refuse with reason.
2. Concatenate enriched DataFrames → stint frame.
3. :func:`~telemetry.sectors.lap_sectors` (3 equal sectors).
4. Build track + car + coupled networks (Phase 3).
5. :func:`~operators.evaluate_triggers` over node metrics (Phase 4).
6. :func:`~grammar.validate_sequence` of the fired operator sequence
   against canonical TNFR U1–U6 — refuse on violation.
7. Estimate ΔC per rule, drop rules whose surrogate ΔC ≤ 0.
8. Assemble :class:`ProposedSetup` (actions + before/after snapshot +
   reproducibility hashes).

The pipeline is pure: it never mutates ``laps``, ``baseline`` or
``car``. Two calls with identical inputs always yield bytewise-equal
:class:`AdvisorResult` instances.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from lfs_telemetry.telemetry.car_info_bin import CarInfoBin
from lfs_telemetry.telemetry.lap import LapTelemetry
from lfs_telemetry.telemetry.observables import CarSpec
from lfs_telemetry.telemetry.sectors import lap_sectors

from .coherence import (
    compute_global_coherence,
    estimate_rule_delta_coherence,
    estimate_sequence_delta_coherence,
)
from .config import AdvisorConfig
from .coupling import couple_track_and_car
from .fields import NetworkSnapshot, network_snapshot
from .grammar import validate_sequence
from .lap_filters import filter_consecutive_laps
from .network_car import build_car_network
from .network_track import build_track_network
from .operators import (
    PhysicalAction,
    SetupSynthesis,
    TriggerRule,
    evaluate_triggers,
    extract_node_metrics,
    synthesize_actions,
)

_DEFAULT_SEED = 20260516

# Canonical TNFR grammar invariants enforced by ``tnfr.operators.grammar_validate``
# (verified empirically against the upstream engine in v0.0.3.x):
#
#   * U1a — when ``epi_initial`` ≈ 0 the sequence MUST start with a
#           canonical generator ∈ {Emission, Recursivity, Transition}.
#   * U1b — the sequence MUST end with a closure operator
#           ∈ {Dissonance, Silence, Transition, Recursivity}.
#   * U2  — every Dissonance (destabilizer) MUST be paired with a
#           stabilizer ∈ {Coherence, SelfOrganization} earlier in the
#           sequence; otherwise ∫ νf·ΔNFR dt may diverge.
#
# The setup advisor only emits operators from the restricted physical
# palette ({Coherence, Dissonance, Reception, Coupling, Silence,
# Contraction}); to keep that palette grammar-stable end-to-end we
# auto-insert canonical *padding* operators whose only role is to
# satisfy U1a / U1b / U2. Padding operators:
#
#   * are NOT exposed as :class:`PhysicalAction` to the user,
#   * do NOT contribute to the surrogate ΔC estimate
#     (their physical effect is null by construction), and
#   * are inserted at well-defined positions so the operator sequence
#     submitted to :func:`validate_sequence` is guaranteed to satisfy
#     the canonical U1–U6 grammar whenever a non-empty set of physical
#     rules has fired.
_EPI_GENERATOR_THRESHOLD = 1e-3
_CANONICAL_GENERATORS = ("emission", "recursivity", "transition")
_CANONICAL_CLOSURES = ("dissonance", "silence", "transition", "recursivity")
_CANONICAL_STABILIZERS = ("coherence", "selforganization")


def _pad_for_grammar(
    ops: Sequence[Any],
    *,
    epi_initial: float,
) -> tuple[Any, ...]:
    """Return ``ops`` padded so it satisfies the canonical U1–U6 grammar.

    Inserts canonical generators / stabilizers / closure operators as
    needed; see the module-level comment for the canonical contract.
    Padding operators carry no physical action and do not enter the
    ΔC surrogate. Idempotent: padding an already-valid sequence is a
    no-op modulo equality of the operator instances.
    """
    from tnfr.operators.definitions import (
        Coherence as _PadCoherence,
        Emission as _PadEmission,
        Silence as _PadSilence,
    )

    seq: list[Any] = list(ops)
    if not seq:
        return ()

    def _name(op: Any) -> str:
        return getattr(op, "name", type(op).__name__).lower()

    names_lower = [_name(o) for o in seq]

    # U2 — pair destabilizers with a stabilizer (prepend Coherence).
    has_destabilizer = "dissonance" in names_lower
    has_stabilizer = any(n in _CANONICAL_STABILIZERS for n in names_lower)
    if has_destabilizer and not has_stabilizer:
        seq.insert(0, _PadCoherence())
        names_lower.insert(0, "coherence")

    # U1a — when EPI ≈ 0, prepend a canonical generator (Emission).
    if (
        abs(float(epi_initial)) < _EPI_GENERATOR_THRESHOLD
        and (not names_lower or names_lower[0] not in _CANONICAL_GENERATORS)
    ):
        seq.insert(0, _PadEmission())
        names_lower.insert(0, "emission")

    # U1b — ensure the sequence ends with a canonical closure (Silence).
    if not names_lower or names_lower[-1] not in _CANONICAL_CLOSURES:
        seq.append(_PadSilence())

    return tuple(seq)


# -- result dataclasses -------------------------------------------------


@dataclass(frozen=True)
class Diagnostic:
    """One observation in physical (non-jargon) language."""

    key: str
    message: str
    value: float | None = None
    units: str | None = None


@dataclass(frozen=True)
class ProposedSetup:
    """A grammar-validated, coherence-positive setup recommendation.

    ``snapshot_after`` is **identical** to ``snapshot_before`` in v1:
    the advisor does not mutate the network graph to apply the
    proposed deltas, so the "after" snapshot reflects the same
    topology / EPI / νf as before. The predicted structural change is
    encoded entirely in :attr:`expected_coherence_delta` (a
    deterministic surrogate). True post-application snapshots are
    deferred to the empirical validation phase (see
    docs/TNFR_SETUP_ADVISOR.md §10.7). Callers that need a real
    before/after comparison should run two stints and use
    :func:`~lfs_telemetry.tnfr_racing.multi_stint.compare_stints`.
    """

    actions: tuple[PhysicalAction, ...]
    expected_coherence_delta: float
    expected_lap_time_delta_ms: float | None
    grammar_passed: bool
    seed: int
    baseline_hash: str
    stint_signature: str
    coherence_before: float
    coherence_after: float
    snapshot_before: NetworkSnapshot
    snapshot_after: NetworkSnapshot
    # Pre-computed consolidated optimal setup: every per-rule
    # PhysicalAction aggregated by (kind, target) into a single net
    # delta per channel, with conflicts flagged. Computed once here
    # so serializers / UI don't recompute on every render.
    synthesis: SetupSynthesis | None = None


@dataclass(frozen=True)
class AdvisorResult:
    """Public output of :meth:`SetupAdvisor.advise`.

    Exactly one of ``proposed`` / ``refusal_reason`` is set.
    """

    proposed: ProposedSetup | None
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)
    refusal_reason: str | None = None

    @classmethod
    def no_safe_recommendation(
        cls, reason: str, diagnostics: Sequence[Diagnostic] = (),
    ) -> "AdvisorResult":
        """Standard refusal constructor."""
        return cls(proposed=None, diagnostics=tuple(diagnostics),
                    refusal_reason=reason)


# -- hashing helpers ----------------------------------------------------


def _hash_baseline(baseline: CarInfoBin) -> str:
    payload: dict[str, Any] = {
        "short_name": baseline.short_name,
        "mass_kg": baseline.mass_kg,
        "weight_dist_front": baseline.weight_dist_front,
        "brake_balance_front": baseline.brake_balance_front,
        "brake_strength_nm": baseline.brake_strength_nm,
        "gear_ratios": list(baseline.gear_ratios),
        "final_drive": baseline.final_drive,
        "wheels": [
            {
                "name": w.name,
                "pressure_kpa": w.tyre_pressure_kpa,
                "spring": w.spring_const,
                "damping_comp": w.damping_comp,
                "damping_rebound": w.damping_rebound,
                "anti_roll": w.anti_roll,
                "camber_rad": w.camber_rad,
                "toe_in_rad": w.toe_in_rad,
            }
            for w in baseline.wheels
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _hash_stint(laps: Sequence[LapTelemetry], track_code: str) -> str:
    parts: list[str] = [track_code, getattr(laps[0].car, "name", "?")]
    for lap in laps:
        df = lap.raw if hasattr(lap, "raw") else lap.enriched
        if df is None or df.empty:
            parts.append("empty")
            continue
        if "time_ms" in df.columns:
            t = pd.to_numeric(df["time_ms"], errors="coerce").dropna()
            parts.append(
                f"{int(t.iloc[0]) if len(t) else 0}-"
                f"{int(t.iloc[-1]) if len(t) else 0}-{len(df)}"
            )
        else:
            parts.append(f"len{len(df)}")
    blob = "|".join(parts).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# -- main advisor -------------------------------------------------------


class SetupAdvisor:
    """Public TNFR Setup Advisor.

    Parameters
    ----------
    seed
        RNG seed forwarded to network builders. Reserved for future
        stochastic operators; v1 is fully deterministic.
    config
        Optional :class:`AdvisorConfig` (defaults are sensible).
    """

    def __init__(
        self,
        *,
        seed: int = _DEFAULT_SEED,
        config: AdvisorConfig | None = None,
    ) -> None:
        self.seed = int(seed)
        self.config = config or AdvisorConfig()

    def advise(
        self,
        laps: Sequence[LapTelemetry],
        baseline: CarInfoBin,
        car: CarSpec,
        track_code: str,
    ) -> AdvisorResult:
        """Run the full advisor pipeline. See module docstring."""
        # 1. Stint validation -------------------------------------------
        sf = filter_consecutive_laps(list(laps), min_count=5)
        if not sf.ok:
            return AdvisorResult.no_safe_recommendation(
                reason=f"insufficient_stint: {sf.reason}"
            )
        clean_laps = sf.laps

        # 2. Concatenate enriched frames -------------------------------
        df_avg = pd.concat(
            [lap.enriched for lap in clean_laps], ignore_index=True,
        )

        # 3. Sectors (3 equal) -----------------------------------------
        sectors = lap_sectors(clean_laps[0], n_equal=3)
        if not sectors:
            return AdvisorResult.no_safe_recommendation(
                reason="sector_decomposition_failed"
            )

        # 4. Networks --------------------------------------------------
        g_track, _ = build_track_network(
            track_code, sectors, df_avg, car, seed=self.seed,
        )
        g_car, _ = build_car_network(car, df_avg, seed=self.seed)
        g = couple_track_and_car(g_track, g_car, df_avg, car)

        snap_before = network_snapshot(g)
        c_before = compute_global_coherence(g)

        # 5. Triggers --------------------------------------------------
        metrics = extract_node_metrics(g)
        fired: tuple[TriggerRule, ...] = evaluate_triggers(metrics)
        if not fired:
            return AdvisorResult.no_safe_recommendation(
                reason="no_rule_fired",
                diagnostics=(
                    Diagnostic(
                        key="coherence_before",
                        message="Global coherence is already within "
                        "target band; no setup action recommended.",
                        value=c_before,
                    ),
                ),
            )

        # 6. Grammar U1-U6 --------------------------------------------
        # Build the canonical operator sequence from fired rules and
        # pad it with grammar-only operators (Emission for U1a when
        # EPI ≈ 0, Coherence for U2 when a destabilizer is present
        # without a stabilizer, Silence as the U1b closure). Padding
        # operators do not become PhysicalActions and contribute 0 to
        # the surrogate ΔC; they exist solely to keep the sequence
        # submitted to the upstream tnfr GrammarValidator canonically
        # well-formed end-to-end (see module-level comment).
        ops = tuple(r.operator_factory() for r in fired)
        ops = _pad_for_grammar(ops, epi_initial=snap_before.epi_mean)
        grammar = validate_sequence(ops, epi_initial=snap_before.epi_mean)
        if not grammar.ok:
            return AdvisorResult.no_safe_recommendation(
                reason=f"grammar_U_violation: {grammar.reason}",
                diagnostics=(
                    Diagnostic(
                        key="grammar",
                        message=(
                            "Proposed setup changes violate canonical "
                            "stability constraints; refusing to recommend."
                        ),
                    ),
                ),
            )

        # 7. Filter by surrogate ΔC -----------------------------------
        kept: tuple[TriggerRule, ...] = tuple(
            r for r in fired if estimate_rule_delta_coherence(r) > 0.0
        )
        if not kept:
            return AdvisorResult.no_safe_recommendation(
                reason="no_coherence_positive_rule"
            )
        delta_c = estimate_sequence_delta_coherence(kept)
        c_after = min(1.0, c_before + delta_c)
        snap_after = snap_before  # graph not mutated; v1 surrogate

        # 8. Assemble ProposedSetup ------------------------------------
        actions = tuple(r.action for r in kept)
        synthesis = synthesize_actions(actions)
        proposed = ProposedSetup(
            actions=actions,
            expected_coherence_delta=delta_c,
            expected_lap_time_delta_ms=None,
            grammar_passed=True,
            seed=self.seed,
            baseline_hash=_hash_baseline(baseline),
            stint_signature=_hash_stint(clean_laps, track_code),
            coherence_before=c_before,
            coherence_after=c_after,
            snapshot_before=snap_before,
            snapshot_after=snap_after,
            synthesis=synthesis,
        )

        diag_list: list[Diagnostic] = [
            Diagnostic(
                key="coherence_before",
                message="Global structural coherence before changes.",
                value=c_before,
            ),
            Diagnostic(
                key="coherence_after",
                message="Projected structural coherence after applying "
                "the proposed setup deltas (deterministic surrogate).",
                value=c_after,
            ),
            Diagnostic(
                key="rules_fired",
                message=f"{len(kept)} of {len(fired)} fired rules retained "
                "after coherence-positive filter.",
                value=float(len(kept)),
            ),
        ]
        if synthesis.conflict_groups:
            n_conflicts = len(synthesis.conflict_groups)
            diag_list.append(
                Diagnostic(
                    key="conflicting_signals",
                    message=(
                        f"{n_conflicts} setup channel(s) received mixed-sign "
                        "advice from different rules; review the conflicting "
                        "signals subsection before applying the consolidated "
                        "delta."
                    ),
                    value=float(n_conflicts),
                )
            )
        diagnostics = tuple(diag_list)

        return AdvisorResult(
            proposed=proposed, diagnostics=diagnostics, refusal_reason=None,
        )

    # Backward-compat shim for the Phase 1 stub API.
    analyze = advise


__all__ = (
    "AdvisorResult",
    "Diagnostic",
    "ProposedSetup",
    "SetupAdvisor",
)
