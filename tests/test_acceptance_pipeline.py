from __future__ import annotations

from statistics import mean

import pytest

from tnfr_lfs.core.coherence import sense_index
from tnfr_lfs.core.contextual_delta import (
    apply_contextual_delta,
    load_context_matrix,
    resolve_context_from_bundle,
)
from tnfr_lfs.core.operators import (
    coherence_operator,
    coupling_operator,
    orchestrate_delta_metrics,
    mutation_operator,
    recursive_filter_operator,
    recursivity_operator,
    resonance_operator,
)
from tnfr_lfs.core.phases import expand_phase_alias


def _is_monotonic(series: list[float]) -> bool:
    return all(curr >= prev - 1e-9 for prev, curr in zip(series, series[1:]))


def test_acceptance_pipeline_monotonicity_and_coupling(
    monkeypatch,
    acceptance_bundle_series,
    acceptance_records,
    acceptance_microsectors,
) -> None:
    """Validate monotonic Si, nodal coupling and resonance via orchestration."""

    expected_bundles = list(acceptance_bundle_series)

    def fake_reception(records, extractor=None):  # type: ignore[override]
        assert len(records) == len(expected_bundles)
        return expected_bundles

    monkeypatch.setattr("tnfr_lfs.core.operators.reception_operator", fake_reception)

    result = orchestrate_delta_metrics(
        [acceptance_records],
        target_delta_nfr=0.6,
        target_sense_index=0.7,
        microsectors=acceptance_microsectors,
        coherence_window=3,
        recursion_decay=0.35,
    )

    coherence_stage = result["stages"]["coherence"]
    smoothed_si = coherence_stage["smoothed_sense_index"]
    smoothed_delta = coherence_stage["smoothed_delta"]
    assert _is_monotonic(smoothed_si)

    context_matrix = load_context_matrix()
    bundle_context = [
        resolve_context_from_bundle(context_matrix, bundle) for bundle in expected_bundles
    ]
    adjusted_series = [
        apply_contextual_delta(
            bundle.delta_nfr,
            factors,
            context_matrix=context_matrix,
        )
        for bundle, factors in zip(expected_bundles, bundle_context)
    ]
    expected_delta = coherence_operator(adjusted_series, window=3)
    assert smoothed_delta == pytest.approx(expected_delta)

    expected_coupling = coupling_operator(smoothed_delta, smoothed_si)
    assert result["coupling"] == pytest.approx(expected_coupling)

    expected_resonance = resonance_operator(smoothed_si)
    assert result["resonance"] == pytest.approx(expected_resonance)

    nodal_metrics = result["nodal_metrics"]
    multipliers = [
        max(
            context_matrix.min_multiplier,
            min(context_matrix.max_multiplier, factors.multiplier),
        )
        for factors in bundle_context
    ]
    tyres_delta = [
        bundle.tyres.delta_nfr * multipliers[idx]
        for idx, bundle in enumerate(expected_bundles)
    ]
    suspension_delta = [
        bundle.suspension.delta_nfr * multipliers[idx]
        for idx, bundle in enumerate(expected_bundles)
    ]
    chassis_delta = [
        bundle.chassis.delta_nfr * multipliers[idx]
        for idx, bundle in enumerate(expected_bundles)
    ]

    assert nodal_metrics["delta_by_node"]["tyres"] == pytest.approx(tyres_delta)

    pairwise_delta = nodal_metrics["pairwise_coupling"]["delta_nfr"]
    assert pairwise_delta["tyres↔suspension"] == pytest.approx(
        coupling_operator(tyres_delta, suspension_delta)
    )
    assert pairwise_delta["tyres↔chassis"] == pytest.approx(
        coupling_operator(tyres_delta, chassis_delta)
    )

    variability = result["microsector_variability"]
    assert variability and variability[0]["overall"]["sense_index"]["variance"] >= 0.0

    occupancy = acceptance_microsectors[0].window_occupancy
    apex_aliases = expand_phase_alias("apex")
    apex_values: list[float] = []
    for phase in apex_aliases:
        phase_data = occupancy.get(phase)
        if phase_data:
            apex_values.extend(phase_data.values())
    assert apex_values
    assert pytest.approx(mean(apex_values), rel=1e-3) == 64.86666666666667


def test_acceptance_nu_f_weighting_penalises_fast_nodes(acceptance_microsectors) -> None:
    """ν_f scaling should reduce Si when the tyre node speeds up."""

    deltas = {"tyres": 0.6, "suspension": 0.2, "chassis": 0.0}
    baseline_nfr = 500.0
    phase_weights = acceptance_microsectors[0].phase_weights
    slow_map = {"tyres": 0.18, "suspension": 0.14, "chassis": 0.12}
    fast_map = dict(slow_map)
    fast_map["tyres"] = slow_map["tyres"] * 2.2

    slow_si = sense_index(
        sum(deltas.values()),
        deltas,
        baseline_nfr,
        nu_f_by_node=slow_map,
        active_phase="apex",
        w_phase=phase_weights,
        nu_f_targets=None,
    )
    fast_si = sense_index(
        sum(deltas.values()),
        deltas,
        baseline_nfr,
        nu_f_by_node=fast_map,
        active_phase="apex",
        w_phase=phase_weights,
        nu_f_targets=None,
    )

    assert fast_si < slow_si


def test_acceptance_memory_and_mutation_converge() -> None:
    """Recursive memory approaches steady state and mutation stabilises."""

    rec_state: dict[str, dict[str, object]] = {}
    session = {"car_model": "gt3", "track_name": "kyoto", "tyre_compound": "soft"}
    sequence = [
        {"thermal_load": 5050.0, "style_index": 0.58, "phase": "entry"},
        {"thermal_load": 5125.0, "style_index": 0.6, "phase": "apex"},
        {"thermal_load": 5180.0, "style_index": 0.62, "phase": "apex"},
    ]
    for measures in sequence:
        rec_output = recursivity_operator(
            rec_state,
            session,
            "0",
            measures,
            decay=0.45,
            history=10,
        )

    steady = {"thermal_load": 5200.0, "style_index": 0.63, "phase": "apex"}
    for _ in range(5):
        rec_output = recursivity_operator(
            rec_state,
            session,
            "0",
            steady,
            decay=0.45,
            history=10,
        )

    assert abs(rec_output["filtered"]["thermal_load"] - steady["thermal_load"]) < 1.0
    assert abs(rec_output["filtered"]["style_index"] - steady["style_index"]) < 1e-2

    series = [0.45, 0.55, 0.6, 0.63, 0.63, 0.63]
    recursive_trace = recursive_filter_operator(series, seed=0.45, decay=0.45)
    assert recursive_trace[-1] > recursive_trace[-2]
    assert abs(recursive_trace[-1] - 0.63) < 0.02

    mutation_state: dict[str, dict[str, object]] = {}
    initial = mutation_operator(
        mutation_state,
        {
            "microsector_id": "0",
            "current_archetype": "medium",
            "candidate_archetype": "hairpin",
            "fallback_archetype": "medium",
            "entropy": 0.72,
            "style_index": 0.88,
            "style_reference": 0.7,
            "phase": "apex",
        },
        entropy_threshold=0.7,
        entropy_increase=0.05,
        style_threshold=0.12,
    )
    assert initial["mutated"]
    assert initial["archetype"] == "hairpin"

    steady_triggers = {
        "microsector_id": "0",
        "current_archetype": initial["archetype"],
        "candidate_archetype": initial["archetype"],
        "fallback_archetype": "medium",
        "entropy": 0.6,
        "style_index": 0.71,
        "style_reference": 0.7,
        "phase": "apex",
    }
    for _ in range(3):
        final_state = mutation_operator(
            mutation_state,
            steady_triggers,
            entropy_threshold=0.7,
            entropy_increase=0.05,
            style_threshold=0.12,
        )

    assert not final_state["mutated"]


def test_orchestrate_delta_metrics_exposes_network_memory() -> None:
    operator_state = {
        "recursivity": {
            "active_session": "FZR|AS5|soft",
            "sessions": {
                "FZR|AS5|soft": {
                    "components": ("FZR", "AS5", "soft"),
                    "stint_index": 1,
                    "samples": 2,
                    "active": {
                        "0": {
                            "phase": "entry",
                            "samples": 1,
                            "filtered": {"style_index": 0.71},
                            "trace": ({"phase": "entry", "style_index": 0.71},),
                            "converged": False,
                            "last_measures": {"style_index": 0.71},
                        }
                    },
                    "history": (
                        {
                            "stint": 0,
                            "ended_at": 12.0,
                            "reason": "max_samples",
                            "samples": 2,
                            "microsectors": {
                                "0": {
                                    "phase": "entry",
                                    "samples": 2,
                                    "filtered": {"style_index": 0.68},
                                    "trace": (),
                                    "converged": True,
                                    "last_measures": {},
                                }
                            },
                        },
                    ),
                }
            },
        }
    }

    result = orchestrate_delta_metrics(
        [],
        target_delta_nfr=0.0,
        target_sense_index=0.0,
        operator_state=operator_state,
    )

    memory = result["network_memory"]
    assert memory["sessions"]
    session_snapshot = memory["sessions"]["FZR|AS5|soft"]
    assert session_snapshot["history"]
    assert session_snapshot["active"]["0"]["filtered"]["style_index"] == pytest.approx(0.71)
