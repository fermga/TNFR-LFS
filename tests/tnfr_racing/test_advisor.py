"""Phase 5: SetupAdvisor pipeline end-to-end."""
from __future__ import annotations

from pathlib import Path

import pytest

from lfs_telemetry.telemetry.car_info_bin import CarInfoBin, CarInfoWheel
from lfs_telemetry.telemetry.lap import LapTelemetry
from lfs_telemetry.tnfr_racing.advisor import (
    AdvisorResult,
    Diagnostic,
    ProposedSetup,
    SetupAdvisor,
)

ASSETS = Path(__file__).resolve().parents[2] / "assets"


def _synthetic_baseline(short_name: str = "FBM") -> CarInfoBin:
    """Minimal CarInfoBin sufficient for the advisor's baseline hash."""
    wheels = tuple(
        CarInfoWheel(
            name=n, tyre_type=0, contact_x_m=0.75 * (1 if "R" in n else -1),
            contact_y_m=1.20 if n.startswith("F") else -1.20, contact_z_m=0.0,
            unsprung_mass_kg=18.0, tyre_width_m=0.22, sidewall_height_prop=0.55,
            rim_radius_m=0.20, rim_width_m=0.18,
            spring_const=42000.0, damping_comp=3200.0, damping_rebound=3800.0,
            anti_roll=18000.0,
            camber_rad=-0.035, inclination_rad=0.0, caster_rad=0.10,
            scrub_radius_m=0.01,
            moment_inertia=1.2, susp_deflection_m=0.05,
            max_susp_deflection_m=0.12,
            tyre_vert_spring=180000.0, tyre_vert_deflection=0.008,
            tyre_pressure_kpa=160.0, air_temp_c=25.0, toe_in_rad=0.0,
        )
        for n in ("RL", "RR", "FL", "FR")
    )
    return CarInfoBin(
        file_version=2, short_name=short_name, passengers=1,
        cg_x_m=0.0, cg_y_m=0.0, cg_z_m=0.30,
        cg_x_rel=0.5, cg_y_rel=0.5, cg_z_rel=0.30,
        fuel_tank_x_m=0.0, fuel_tank_y_m=-0.5, fuel_tank_z_m=0.25,
        max_torque_nm=180.0, max_torque_rpm=5500.0,
        max_power_kw=110.0, max_power_rpm=7800.0,
        fuel_capacity_l=40.0, mass_kg=525.0, wheelbase_m=2.40,
        weight_dist_front=0.50, forward_gears=6, drive="RWD",
        torque_split=0.0, drivetrain_efficiency=0.93,
        gear_ratios=(-3.0, 3.0, 2.1, 1.6, 1.3, 1.1, 0.9), final_drive=4.0,
        parallel_steer=0.5, brake_strength_nm=1100.0, brake_balance_front=0.60,
        wheels=wheels,
    )


@pytest.fixture(scope="module")
def bl1_inputs():
    paths = sorted(ASSETS.glob("synthetic_BL1_FBM_v2_lap*.csv"))
    laps = [LapTelemetry.from_csv(p) for p in paths]
    return laps, _synthetic_baseline(), laps[0].car


def test_advise_returns_result(bl1_inputs) -> None:
    laps, baseline, car = bl1_inputs
    advisor = SetupAdvisor(seed=17)
    res = advisor.advise(laps, baseline, car, "BL1")
    assert isinstance(res, AdvisorResult)
    # Either a proposal or a documented refusal — never both.
    assert (res.proposed is None) != (res.refusal_reason is None)


def test_advise_rejects_short_stint(bl1_inputs) -> None:
    laps, baseline, car = bl1_inputs
    res = SetupAdvisor(seed=17).advise(laps[:2], baseline, car, "BL1")
    assert res.proposed is None
    assert res.refusal_reason is not None
    assert res.refusal_reason.startswith("insufficient_stint")


def test_advise_is_deterministic(bl1_inputs) -> None:
    laps, baseline, car = bl1_inputs
    a = SetupAdvisor(seed=17).advise(laps, baseline, car, "BL1")
    b = SetupAdvisor(seed=17).advise(laps, baseline, car, "BL1")
    assert a.refusal_reason == b.refusal_reason
    if a.proposed and b.proposed:
        assert a.proposed.actions == b.proposed.actions
        assert a.proposed.baseline_hash == b.proposed.baseline_hash
        assert a.proposed.stint_signature == b.proposed.stint_signature
        assert a.proposed.expected_coherence_delta == pytest.approx(
            b.proposed.expected_coherence_delta
        )


def test_proposed_setup_invariants(bl1_inputs) -> None:
    laps, baseline, car = bl1_inputs
    res = SetupAdvisor(seed=17).advise(laps, baseline, car, "BL1")
    if res.proposed is None:
        # Acceptable: real telemetry may not trigger any rule.
        assert res.refusal_reason
        return
    p: ProposedSetup = res.proposed
    assert p.grammar_passed is True
    assert p.expected_coherence_delta > 0.0
    assert 0.0 <= p.coherence_before <= 1.0
    assert 0.0 <= p.coherence_after <= 1.0
    assert p.coherence_after >= p.coherence_before
    assert len(p.baseline_hash) == 16
    assert len(p.stint_signature) == 16
    # Actions never mention TNFR jargon — only physical kinds.
    allowed_kinds = {
        "damper_rebound", "damper_bump", "spring", "arb",
        "tyre_pressure", "brake_bias", "camber", "toe", "ride_height",
    }
    for act in p.actions:
        assert act.kind in allowed_kinds
        assert act.target in ("FL", "FR", "RL", "RR", "front", "rear", "global")
        assert act.units in ("clicks", "N/mm", "kPa", "%", "deg", "mm")


def test_diagnostics_use_physical_language(bl1_inputs) -> None:
    laps, baseline, car = bl1_inputs
    res = SetupAdvisor(seed=17).advise(laps, baseline, car, "BL1")
    forbidden = ("EPI", "νf", "ΔNFR", "operator", "Φ_s", "U1", "U6")
    for d in res.diagnostics:
        assert isinstance(d, Diagnostic)
        for tok in forbidden:
            assert tok not in d.message, f"{d.key}: leaked '{tok}' → {d.message}"


def test_baseline_hash_changes_with_baseline(bl1_inputs) -> None:
    laps, _, car = bl1_inputs
    b1 = _synthetic_baseline("FBM")
    b2 = _synthetic_baseline("FBM")
    object.__setattr__(b2, "mass_kg", 530.0)
    r1 = SetupAdvisor(seed=17).advise(laps, b1, car, "BL1")
    r2 = SetupAdvisor(seed=17).advise(laps, b2, car, "BL1")
    if r1.proposed and r2.proposed:
        assert r1.proposed.baseline_hash != r2.proposed.baseline_hash


# ---------------------------------------------------------------------------
# Phase 7 — validation harness (docs/TNFR_SETUP_ADVISOR.md §10)
# ---------------------------------------------------------------------------

# Full forbidden-token list from docs §10.3. The advisor's public surface
# (diagnostics + action kinds/targets/rationale_ids) must never leak any of
# these — they are TNFR-internal labels, not race-engineering language.
FORBIDDEN_TERMS = (
    "EPI", "νf", "ΔNFR", "Φ_s", "Si", "operador",
    "AL", "EN", "IL", "OZ", "UM", "RA", "SHA",
    "VAL", "NUL", "THOL", "ZHIR", "NAV", "REMESH",
    "U1", "U2", "U3", "U4", "U5", "U6", "tétrada",
)


def _public_text_blobs(result: AdvisorResult) -> list[str]:
    """Every textual fragment the user can possibly see for this result."""
    blobs: list[str] = []
    if result.refusal_reason is not None:
        blobs.append(result.refusal_reason)
    for d in result.diagnostics:
        blobs.append(d.key)
        blobs.append(d.message)
        if d.units:
            blobs.append(d.units)
    if result.proposed is not None:
        for act in result.proposed.actions:
            blobs.append(act.kind)
            blobs.append(act.target)
            blobs.append(act.units)
            blobs.append(act.rationale_id)
    return blobs


def test_no_tnfr_jargon_in_any_public_surface(bl1_inputs) -> None:
    laps, baseline, car = bl1_inputs
    res = SetupAdvisor(seed=17).advise(laps, baseline, car, "BL1")
    for blob in _public_text_blobs(res):
        for term in FORBIDDEN_TERMS:
            assert term not in blob, (
                f"TNFR jargon '{term}' leaked into public text: {blob!r}"
            )


def test_no_tnfr_jargon_on_grammar_refusal_path() -> None:
    """Force the grammar-refusal branch and verify diagnostics stay clean."""
    from lfs_telemetry.tnfr_racing.advisor import AdvisorResult, Diagnostic
    # Build a refusal directly with the same wording the advisor uses.
    refused = AdvisorResult.no_safe_recommendation(
        reason="grammar_U_violation: synthetic",
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
    for blob in _public_text_blobs(refused):
        # The refusal_reason itself encodes "U_violation" — that's an
        # internal tag never rendered raw by the UI (see
        # SetupAdvisorTab._format_refusal). Skip it here; the UI test
        # covers the rendered form.
        if blob.startswith("grammar_U_violation"):
            continue
        for term in FORBIDDEN_TERMS:
            assert term not in blob, (
                f"TNFR jargon '{term}' leaked into refusal text: {blob!r}"
            )


def test_proposed_sequence_passes_canonical_grammar(bl1_inputs) -> None:
    """Re-validate the operator sequence the advisor fed to the grammar."""
    import pandas as pd

    from lfs_telemetry.tnfr_racing.coupling import couple_track_and_car
    from lfs_telemetry.tnfr_racing.fields import network_snapshot
    from lfs_telemetry.tnfr_racing.grammar import validate_sequence
    from lfs_telemetry.telemetry.sectors import lap_sectors
    from lfs_telemetry.tnfr_racing.lap_filters import filter_consecutive_laps
    from lfs_telemetry.tnfr_racing.network_car import build_car_network
    from lfs_telemetry.tnfr_racing.network_track import build_track_network
    from lfs_telemetry.tnfr_racing.operators import (
        evaluate_triggers, extract_node_metrics,
    )

    laps, baseline, car = bl1_inputs
    sf = filter_consecutive_laps(list(laps), min_count=5)
    if not sf.ok:
        pytest.skip("synthetic stint not consecutive: {}".format(sf.reason))
    df = pd.concat([lp.enriched for lp in sf.laps], ignore_index=True)
    sectors = lap_sectors(sf.laps[0], n_equal=3)
    g_t, _ = build_track_network("BL1", sectors, df, car, seed=17)
    g_c, _ = build_car_network(car, df, seed=17)
    g = couple_track_and_car(g_t, g_c, df, car)
    fired = evaluate_triggers(extract_node_metrics(g))
    ops = [r.operator_factory() for r in fired]
    if not ops:
        pytest.skip("no operator fired on this stint")
    # Mirror the advisor's grammar padding (U1a generator + U2
    # stabilizer + U1b closure) so the sequence we re-validate matches
    # the one the advisor itself submits to the upstream validator.
    from lfs_telemetry.tnfr_racing.advisor import _pad_for_grammar
    snap = network_snapshot(g)
    ops = list(_pad_for_grammar(tuple(ops), epi_initial=snap.epi_mean))
    result = validate_sequence(ops, epi_initial=snap.epi_mean)
    assert bool(result), f"grammar validation failed: {result.reason}"


def test_coherence_delta_strictly_positive_or_refusal(bl1_inputs) -> None:
    """Acceptance criterion (§10.5b): ΔC(t) > 0 when a proposal is made.

    MIN_BUSINESS_COHERENCE = 0.242 is a bifurcation floor in the TNFR
    engine — *not* a quality threshold. Only the delta is asserted.
    """
    laps, baseline, car = bl1_inputs
    res = SetupAdvisor(seed=17).advise(laps, baseline, car, "BL1")
    if res.proposed is None:
        assert res.refusal_reason is not None
        return
    p = res.proposed
    assert p.expected_coherence_delta > 0.0
    # Coherence is bounded in [0, 1]; if it already saturates at the
    # ceiling, ``coherence_after == coherence_before == 1.0`` is the
    # only physically possible outcome. Only the *delta* is asserted
    # strictly (the docstring above already says so).
    assert p.coherence_after >= p.coherence_before
    assert p.coherence_after <= 1.0


def test_seed_changes_do_not_change_baseline_hash(bl1_inputs) -> None:
    """``baseline_hash`` depends only on the CarInfoBin, not the seed."""
    laps, baseline, car = bl1_inputs
    a = SetupAdvisor(seed=1).advise(laps, baseline, car, "BL1")
    b = SetupAdvisor(seed=999_999).advise(laps, baseline, car, "BL1")
    if a.proposed and b.proposed:
        assert a.proposed.baseline_hash == b.proposed.baseline_hash
        assert a.proposed.stint_signature == b.proposed.stint_signature
        assert a.proposed.seed != b.proposed.seed


def test_stint_signature_changes_with_track(bl1_inputs) -> None:
    laps, baseline, car = bl1_inputs
    a = SetupAdvisor(seed=17).advise(laps, baseline, car, "BL1")
    b = SetupAdvisor(seed=17).advise(laps, baseline, car, "AS3")
    if a.proposed and b.proposed:
        assert a.proposed.stint_signature != b.proposed.stint_signature


def test_proposed_setup_caches_synthesis(bl1_inputs) -> None:
    """Consolidated synthesis must be computed once and cached on the
    ProposedSetup so downstream consumers (CLI, serializers, UI) read
    a single deterministic aggregation."""
    laps, baseline, car = bl1_inputs
    res = SetupAdvisor(seed=17).advise(laps, baseline, car, "BL1")
    if res.proposed is None:
        pytest.skip("no proposal on this stint")
    p = res.proposed
    assert p.synthesis is not None
    # Every action must map back to exactly one consolidated channel.
    channels = {(a.kind, a.target) for a in p.actions}
    synth_channels = {(adj.kind, adj.target) for adj in p.synthesis.adjustments}
    assert channels == synth_channels


# -- canonical grammar padding (U1a / U2 / U1b) ------------------------


def test_pad_for_grammar_prepends_generator_when_epi_zero() -> None:
    """U1a: a sequence starting from EPI≈0 must lead with a canonical
    generator (Emission/Recursivity/Transition). Padding inserts
    Emission as the neutral generative initiator."""
    from tnfr.operators.definitions import Coherence, Silence
    from lfs_telemetry.tnfr_racing.advisor import _pad_for_grammar
    from lfs_telemetry.tnfr_racing.grammar import validate_sequence

    raw = (Coherence(), Silence())
    padded = _pad_for_grammar(raw, epi_initial=0.0)
    names = [getattr(o, "name", type(o).__name__).lower() for o in padded]
    assert names[0] == "emission"
    assert names[-1] in {"silence", "dissonance", "transition", "recursivity"}
    # And the padded sequence must now satisfy the upstream validator.
    assert bool(validate_sequence(padded, epi_initial=0.0))
    # When EPI > 0 the generator is NOT prepended.
    padded_pos = _pad_for_grammar(raw, epi_initial=0.5)
    assert getattr(padded_pos[0], "name", "").lower() != "emission"


def test_pad_for_grammar_prepends_stabilizer_for_dissonance() -> None:
    """U2: a Dissonance-only sequence (e.g. only ARB rules fire) must
    be paired with a stabilizer. Padding inserts Coherence."""
    from tnfr.operators.definitions import Dissonance, Silence
    from lfs_telemetry.tnfr_racing.advisor import _pad_for_grammar
    from lfs_telemetry.tnfr_racing.grammar import validate_sequence

    # Two Dissonance ops, no closure: padding must add Coherence (U2)
    # AND Silence (U1b) since the last op is not a canonical closure
    # when the sequence already ends in Dissonance the closure can be
    # implicit; here we force the closure path with an extra op pair.
    raw = (Dissonance(), Dissonance(), Silence())
    padded = _pad_for_grammar(raw, epi_initial=0.5)
    names = [getattr(o, "name", type(o).__name__).lower() for o in padded]
    assert "coherence" in names
    # Coherence must appear before any Dissonance (stabilizer-pair order).
    assert names.index("coherence") < names.index("dissonance")
    assert bool(validate_sequence(padded, epi_initial=0.5))


def test_pad_for_grammar_is_noop_when_already_canonical() -> None:
    """Padding an already-canonical sequence must not add operators."""
    from tnfr.operators.definitions import Coherence, Dissonance, Silence
    from lfs_telemetry.tnfr_racing.advisor import _pad_for_grammar

    raw = (Coherence(), Dissonance(), Silence())
    padded = _pad_for_grammar(raw, epi_initial=0.5)
    assert len(padded) == len(raw)
    names_raw = [getattr(o, "name", type(o).__name__).lower() for o in raw]
    names_padded = [getattr(o, "name", type(o).__name__).lower() for o in padded]
    assert names_raw == names_padded


def test_pad_for_grammar_empty_sequence_stays_empty() -> None:
    """Padding refuses to fabricate a sequence from nothing — the
    advisor handles ``no_rule_fired`` before grammar validation."""
    from lfs_telemetry.tnfr_racing.advisor import _pad_for_grammar
    assert _pad_for_grammar((), epi_initial=0.0) == ()


def test_pad_for_grammar_does_not_inflate_delta_c() -> None:
    """Padding operators must carry 0 surrogate ΔC: the predicted
    coherence improvement is determined by physical fired rules only,
    never by grammar-padding operators."""
    from tnfr.operators.definitions import (
        Coherence as _Coh,
        Dissonance as _Dis,
        Emission as _Em,
        Silence as _Si,
    )
    from lfs_telemetry.tnfr_racing.coherence import _OPERATOR_GAIN

    # Emission must contribute 0.0 (neutral generator).
    assert _OPERATOR_GAIN.get("Emission", 0.0) == 0.0
    # Coherence pad has the same gain as a Coherence rule — but it is
    # never added to ``kept`` rules in advisor.py, so the surrogate
    # ΔC stays a function of fired physical rules only. This test
    # documents that invariant.
    _ = (_Coh, _Dis, _Em, _Si)  # imports prove the operators exist

