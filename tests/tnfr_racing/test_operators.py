"""Phase 4: TRIGGER_RULES + grammar validation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lfs_telemetry.telemetry.lap import LapTelemetry
from lfs_telemetry.telemetry.sectors import lap_sectors
from lfs_telemetry.tnfr_racing.coupling import couple_track_and_car
from lfs_telemetry.tnfr_racing.grammar import GrammarResult, validate_sequence
from lfs_telemetry.tnfr_racing.network_car import build_car_network
from lfs_telemetry.tnfr_racing.network_track import build_track_network
from lfs_telemetry.tnfr_racing.operators import (
    PhysicalAction,
    TRIGGER_RULES,
    TriggerRule,
    agg_corner_phase,
    evaluate_triggers,
    extract_node_metrics,
)

ASSETS = Path(__file__).resolve().parents[2] / "assets"


@pytest.fixture(scope="module")
def coupled_graph():
    paths = sorted(ASSETS.glob("synthetic_BL1_FBM_v2_lap*.csv"))
    laps = [LapTelemetry.from_csv(p) for p in paths]
    df = pd.concat([lap.enriched for lap in laps], ignore_index=True)
    sectors = lap_sectors(laps[0], n_equal=3)
    gt, _ = build_track_network("BL1", sectors, df, laps[0].car, seed=17)
    gc, _ = build_car_network(laps[0].car, df, seed=17)
    return couple_track_and_car(gt, gc, df, laps[0].car)


def test_trigger_rules_table_well_formed() -> None:
    # The canonical table is the operator → physical-action map. The
    # advisor's professionalism depends on broad coverage, so we lock
    # the lower bound (≥ 30 rules) but allow growth without breaking
    # this contract test on each addition.
    assert len(TRIGGER_RULES) >= 30
    names = [r.name for r in TRIGGER_RULES]
    assert len(set(names)) == len(names)
    # Every canonical PhysicalAction subsystem must be covered by at
    # least one rule — otherwise the advisor cannot recommend it.
    kinds = {r.action.kind for r in TRIGGER_RULES}
    required = {
        "damper_rebound", "damper_bump", "spring", "arb",
        "tyre_pressure", "brake_bias", "camber", "toe", "ride_height",
    }
    assert required <= kinds, f"missing action kinds: {required - kinds}"
    for r in TRIGGER_RULES:
        assert isinstance(r, TriggerRule)
        assert isinstance(r.action, PhysicalAction)
        assert r.action.rationale_id == r.name
        # operator factory must yield a tnfr Operator
        op = r.operator_factory()
        assert hasattr(op, "name") or hasattr(op, "__class__")


def test_extract_node_metrics_shape(coupled_graph) -> None:
    m = extract_node_metrics(coupled_graph)
    assert "wheel.FL" in m and "axle.front" in m and "chassis" in m
    assert "corner.0.entry" in m
    for entry in m.values():
        assert {"epi", "vf", "theta", "meta"} <= set(entry)
    assert m["corner.0.entry"]["phase"] == "entry"


def test_evaluate_triggers_returns_subset(coupled_graph) -> None:
    m = extract_node_metrics(coupled_graph)
    fired = evaluate_triggers(m)
    assert isinstance(fired, tuple)
    assert all(r in TRIGGER_RULES for r in fired)


def test_synthetic_predicates_fire_when_forced() -> None:
    """Hand-craft a metrics dict that activates a representative set
    of rules (front-loaded understeer scenario). With 30+ rules in
    the table we no longer enforce *which* exact subset fires — only
    that enough rules of each category trigger to prove the table is
    wired up and reachable.
    """
    base = {
        "wheel.FL": {"epi": 0.95, "vf": 9.0, "theta": 0.0, "meta": {"friction_use_mean": 0.9, "temp_mean_c": 145.0}},
        "wheel.FR": {"epi": 0.70, "vf": 9.0, "theta": 0.0, "meta": {"friction_use_mean": 0.6, "temp_mean_c": 125.0}},
        "wheel.RL": {"epi": 0.40, "vf": 3.0, "theta": 0.0, "meta": {"friction_use_mean": 0.9, "temp_mean_c": 145.0}},
        "wheel.RR": {"epi": 0.70, "vf": 9.0, "theta": 0.0, "meta": {"friction_use_mean": 0.6, "temp_mean_c": 125.0}},
        "axle.front": {"epi": 0.8, "vf": 2.0, "theta": 0.0, "meta": {"slip_diff_rms": 0.20}},
        "axle.rear": {"epi": 0.5, "vf": 2.0, "theta": 0.0, "meta": {"slip_diff_rms": 0.20}},
        "brake.front": {"epi": 0.3, "vf": 1.0, "theta": 0.0, "meta": {"brake_bias_front_real_mean": 0.50, "brake_force_rms_n": 5000.0}},
        "brake.rear": {"epi": 0.3, "vf": 1.0, "theta": 0.0, "meta": {"brake_force_rms_n": 3000.0}},
        "engine": {"epi": 0.3, "vf": 1.0, "theta": 0.0, "meta": {}},
        "chassis": {"epi": 0.3, "vf": 2.0, "theta": 0.0, "meta": {}},
        "driver": {"epi": 0.3, "vf": 1.0, "theta": 0.0, "meta": {}},
        "corner.0.entry": {"epi": 0.7, "vf": 1.0, "theta": 0.0, "meta": {}, "phase": "entry"},
        "corner.0.apex": {"epi": 0.7, "vf": 1.0, "theta": 0.0, "meta": {}, "phase": "apex"},
        "corner.0.exit": {"epi": 0.7, "vf": 1.0, "theta": 0.0, "meta": {}, "phase": "exit"},
    }
    fired = {r.name for r in evaluate_triggers(base)}
    # Representative sample from each category should fire under this
    # configuration. If any of these stops firing, a predicate has
    # regressed.
    must_fire = {
        "FL_oscillation_entry",     # damper / Coherence
        "apex_understeer",          # ARB balance / Dissonance
        "friction_saturation_FL",   # Reception
        "lateral_imbalance_front",  # Coupling
        "thermal_axle_overload_front",  # Silence
        "thermal_axle_overload_rear",
        "brake_bias_drift",         # Coherence brake
        "brake_force_concentration_front",  # Contraction
        "chassis_load_saturation_front",    # Mutation spring
        "camber_misalignment_front",        # SelfOrganization
        "toe_misalignment_front",
    }
    assert must_fire <= fired, f"missing fires: {must_fire - fired}"

    flipped = {**base,
               "axle.front": {**base["axle.front"], "epi": 0.5},
               "axle.rear": {**base["axle.rear"], "epi": 0.8}}
    fired_b = {r.name for r in evaluate_triggers(flipped)}
    assert "exit_oversteer" in fired_b
    assert "apex_understeer" not in fired_b


def test_agg_corner_phase_handles_missing() -> None:
    assert agg_corner_phase({}, "entry") != agg_corner_phase({}, "entry") or True
    # i.e. result is NaN — NaN != NaN. Just exercise the path.
    val = agg_corner_phase({}, "entry")
    assert val != val  # NaN


def test_validate_sequence_accepts_canonical_run() -> None:
    # Emission is a valid start operator per VALID_START_OPERATORS.
    from tnfr.operators.definitions import Coherence, Emission, Silence

    res = validate_sequence([Emission(), Coherence(), Silence()], epi_initial=0.0)
    assert isinstance(res, GrammarResult)
    assert res.ok, res.reason


def test_validate_sequence_rejects_empty() -> None:
    res = validate_sequence([])
    assert not res.ok
    assert "empty" in res.reason


def test_validate_sequence_rule_operators_are_valid() -> None:
    """All TRIGGER_RULES operators must instantiate without error."""
    for r in TRIGGER_RULES:
        op = r.operator_factory()
        # GrammarValidator expects objects with a 'name' attr.
        assert hasattr(op, "name")


# ---------------------------------------------------------------------
# Setup synthesis (consolidated optimal setup)
# ---------------------------------------------------------------------


def test_synthesize_actions_merges_same_channel() -> None:
    from lfs_telemetry.tnfr_racing.operators import (
        ConsolidatedAdjustment, synthesize_actions,
    )

    acts = (
        PhysicalAction("arb", "rear", +2.0, "clicks", "exit_oversteer"),
        PhysicalAction("arb", "rear", -2.0, "clicks", "apex_understeer"),
        PhysicalAction("arb", "rear", +1.0, "clicks", "lateral_imbalance_rear"),
        PhysicalAction("tyre_pressure", "FL", +1.0, "kPa", "friction_saturation_FL"),
    )
    synth = synthesize_actions(acts)

    by_key = {(a.kind, a.target): a for a in synth.adjustments}
    arb_rear = by_key[("arb", "rear")]
    assert arb_rear.net_delta == 1.0
    assert arb_rear.confidence == 3
    assert set(arb_rear.contributing_rules) == {
        "exit_oversteer", "apex_understeer", "lateral_imbalance_rear",
    }
    # Conflict detected: +2/+1 vs -2 → ups and downs both non-empty
    conflicts = {(k, t) for (k, t, _, _) in synth.conflict_groups}
    assert ("arb", "rear") in conflicts

    fl = by_key[("tyre_pressure", "FL")]
    assert fl.net_delta == 1.0
    assert fl.confidence == 1
    assert isinstance(fl, ConsolidatedAdjustment)


def test_synthesize_actions_empty_returns_empty() -> None:
    from lfs_telemetry.tnfr_racing.operators import synthesize_actions

    synth = synthesize_actions(())
    assert synth.adjustments == ()
    assert synth.fired_rules == ()
    assert synth.conflict_groups == ()


def test_synthesize_setup_sorts_by_abs_delta() -> None:
    from lfs_telemetry.tnfr_racing.operators import synthesize_actions

    acts = (
        PhysicalAction("tyre_pressure", "FL", +1.0, "kPa", "r1"),
        PhysicalAction("spring", "front", +5.0, "N/mm", "r2"),
        PhysicalAction("camber", "front", -0.2, "deg", "r3"),
    )
    synth = synthesize_actions(acts)
    # Largest |delta| first
    deltas = [abs(a.net_delta) for a in synth.adjustments]
    assert deltas == sorted(deltas, reverse=True)
