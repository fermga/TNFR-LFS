"""Unit tests for the high-level TNFR × LFS operators."""

from __future__ import annotations

from dataclasses import replace
from math import sqrt
from statistics import mean, pstdev, pvariance
from typing import List

import pytest

from tnfr_lfs.core import Goal, Microsector, TelemetryRecord
from tnfr_lfs.core.coherence import sense_index
from tnfr_lfs.core.epi import (
    DEFAULT_PHASE_WEIGHTS,
    delta_nfr_by_node,
    resolve_nu_f_by_node,
)
from tnfr_lfs.core.epi_models import (
    BrakesNode,
    ChassisNode,
    DriverNode,
    EPIBundle,
    SuspensionNode,
    TrackNode,
    TransmissionNode,
    TyresNode,
)
from tnfr_lfs.core.operators import (
    DissonanceBreakdown,
    acoplamiento_operator,
    coherence_operator,
    dissonance_breakdown_operator,
    dissonance_operator,
    evolve_epi,
    emission_operator,
    mutation_operator,
    orchestrate_delta_metrics,
    pairwise_coupling_operator,
    recepcion_operator,
    recursivity_operator,
    recursividad_operator,
    resonance_operator,
)


def _build_record(
    timestamp: float,
    vertical_load: float,
    slip_ratio: float,
    lateral_accel: float,
    longitudinal_accel: float,
    nfr: float,
    si: float,
    *,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    brake_pressure: float = 0.0,
    locking: float = 0.0,
    speed: float = 0.0,
    yaw_rate: float = 0.0,
    slip_angle: float = 0.0,
    steer: float = 0.0,
    throttle: float = 0.0,
    gear: int = 0,
    vertical_load_front: float = 0.0,
    vertical_load_rear: float = 0.0,
    mu_eff_front: float = 0.0,
    mu_eff_rear: float = 0.0,
    suspension_travel_front: float = 0.0,
    suspension_travel_rear: float = 0.0,
    suspension_velocity_front: float = 0.0,
    suspension_velocity_rear: float = 0.0,
) -> TelemetryRecord:
    return TelemetryRecord(
        timestamp=timestamp,
        vertical_load=vertical_load,
        slip_ratio=slip_ratio,
        lateral_accel=lateral_accel,
        longitudinal_accel=longitudinal_accel,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        brake_pressure=brake_pressure,
        locking=locking,
        nfr=nfr,
        si=si,
        speed=speed,
        yaw_rate=yaw_rate,
        slip_angle=slip_angle,
        steer=steer,
        throttle=throttle,
        gear=gear,
        vertical_load_front=vertical_load_front,
        vertical_load_rear=vertical_load_rear,
        mu_eff_front=mu_eff_front,
        mu_eff_rear=mu_eff_rear,
        suspension_travel_front=suspension_travel_front,
        suspension_travel_rear=suspension_travel_rear,
        suspension_velocity_front=suspension_velocity_front,
        suspension_velocity_rear=suspension_velocity_rear,
    )


def _build_goal(phase: str, target_delta: float, *, archetype: str = "equilibrio") -> Goal:
    return Goal(
        phase=phase,
        archetype=archetype,
        description=f"Meta sintética para {phase}",
        target_delta_nfr=target_delta,
        target_sense_index=0.9,
        nu_f_target=0.0,
        slip_lat_window=(-0.5, 0.5),
        slip_long_window=(-0.5, 0.5),
        yaw_rate_window=(-0.5, 0.5),
        dominant_nodes=("tyres",),
    )


def _build_microsector(
    index: int,
    entry_idx: int,
    apex_idx: int,
    exit_idx: int,
    *,
    apex_target: float,
    support_event: bool = True,
    archetype: str = "apoyo",
) -> Microsector:
    goals = (
        _build_goal("entry", 0.0, archetype=archetype),
        _build_goal("apex", apex_target, archetype=archetype),
        _build_goal("exit", 0.0, archetype=archetype),
    )
    phase_boundaries = {
        "entry": (entry_idx, entry_idx + 1),
        "apex": (apex_idx, apex_idx + 1),
        "exit": (exit_idx, exit_idx + 1),
    }
    phase_samples = {
        "entry": (entry_idx,),
        "apex": (apex_idx,),
        "exit": (exit_idx,),
    }
    dominant_nodes = {phase: ("tyres",) for phase in ("entry", "apex", "exit")}
    phase_weights = {phase: {} for phase in ("entry", "apex", "exit")}
    filtered_measures = {
        "thermal_load": 5000.0,
        "style_index": 0.9,
        "grip_rel": 1.0,
    }
    window_occupancy = {
        "entry": {"slip_lat": 100.0, "slip_long": 100.0, "yaw_rate": 100.0},
        "apex": {"slip_lat": 100.0, "slip_long": 100.0, "yaw_rate": 100.0},
        "exit": {"slip_lat": 100.0, "slip_long": 100.0, "yaw_rate": 100.0},
    }
    return Microsector(
        index=index,
        start_time=float(entry_idx),
        end_time=float(exit_idx),
        curvature=1.0,
        brake_event=False,
        support_event=support_event,
        delta_nfr_signature=0.0,
        goals=goals,
        phase_boundaries=phase_boundaries,
        phase_samples=phase_samples,
        active_phase="apex",
        dominant_nodes=dominant_nodes,
        phase_weights=phase_weights,
        grip_rel=1.0,
        filtered_measures=filtered_measures,
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy=window_occupancy,
    )


def _build_bundle(timestamp: float, tyre_delta: float, *, delta_nfr: float | None = None) -> EPIBundle:
    delta_value = tyre_delta if delta_nfr is None else delta_nfr
    return EPIBundle(
        timestamp=timestamp,
        epi=0.0,
        delta_nfr=delta_value,
        sense_index=0.9,
        tyres=TyresNode(delta_nfr=tyre_delta, sense_index=0.9),
        suspension=SuspensionNode(delta_nfr=delta_value, sense_index=0.9),
        chassis=ChassisNode(delta_nfr=delta_value, sense_index=0.9),
        brakes=BrakesNode(delta_nfr=0.0, sense_index=0.9),
        transmission=TransmissionNode(delta_nfr=0.0, sense_index=0.9),
        track=TrackNode(delta_nfr=0.0, sense_index=0.9),
        driver=DriverNode(delta_nfr=0.0, sense_index=0.9),
    )


def test_coherence_operator_reduces_jitter_without_bias():
    raw_series = [0.2, 0.8, 0.1, 0.9, 0.2]
    smoothed = coherence_operator(raw_series, window=3)

    assert mean(smoothed) == pytest.approx(mean(raw_series), rel=1e-9)
    assert pstdev(smoothed) < pstdev(raw_series)


def test_evolve_epi_runs_euler_step():
    prev = 0.45
    deltas = {"tyres": 1.2, "suspension": -0.6, "driver": 0.4}
    nu_map = {"tyres": 0.18, "suspension": 0.14, "driver": 0.05}
    new_epi, derivative, nodal = evolve_epi(prev, deltas, 0.1, nu_map)

    expected_derivative = 0.18 * 1.2 + 0.14 * -0.6 + 0.05 * 0.4
    assert derivative == pytest.approx(expected_derivative, rel=1e-9)
    assert new_epi == pytest.approx(prev + expected_derivative * 0.1, rel=1e-9)
    assert {"tyres", "suspension", "driver"} <= set(nodal)
    nodal_derivative = sum(component[1] for component in nodal.values())
    nodal_integral = sum(component[0] for component in nodal.values())
    assert nodal_derivative == pytest.approx(derivative, rel=1e-9)
    assert nodal_integral == pytest.approx(expected_derivative * 0.1, rel=1e-9)


def test_sense_index_penalises_active_phase_weights():
    baseline = _build_record(0.0, 5000.0, 0.02, 0.9, 0.3, 0.8, 0.92)
    sample = _build_record(0.1, 5450.0, 0.18, 1.4, -0.6, 0.86, 0.78)
    node_deltas = delta_nfr_by_node(replace(sample, reference=baseline))
    weights = {
        "entry": {"__default__": 1.0},
        "apex": {"__default__": 1.0, "tyres": 2.0, "chassis": 1.6},
        "exit": {"__default__": 1.0},
    }
    nu_entry = resolve_nu_f_by_node(sample, phase="entry", phase_weights=weights)
    nu_apex = resolve_nu_f_by_node(sample, phase="apex", phase_weights=weights)

    assert nu_apex["tyres"] > nu_entry["tyres"]
    assert nu_apex["chassis"] > nu_entry["chassis"]

    entry_index = sense_index(
        sample.nfr - baseline.nfr,
        node_deltas,
        baseline.nfr,
        nu_f_by_node=nu_entry,
        active_phase="entry",
        w_phase=weights,
    )
    apex_index = sense_index(
        sample.nfr - baseline.nfr,
        node_deltas,
        baseline.nfr,
        nu_f_by_node=nu_apex,
        active_phase="apex",
        w_phase=weights,
    )

    neutral_index = sense_index(
        sample.nfr - baseline.nfr,
        node_deltas,
        baseline.nfr,
        nu_f_by_node=resolve_nu_f_by_node(sample),
        active_phase="apex",
        w_phase=DEFAULT_PHASE_WEIGHTS,
    )

    assert apex_index < entry_index
    assert apex_index <= neutral_index


def test_orchestrator_pipeline_builds_consistent_metrics():
    segment_a = [
        _build_record(0.0, 5200.0, 0.05, 1.2, 0.6, 0.82, 0.91),
        _build_record(1.0, 5100.0, 0.04, 1.1, 0.5, 0.81, 0.92),
    ]
    segment_b = [
        _build_record(2.0, 5000.0, 0.03, 1.0, 0.4, 0.80, 0.90),
        _build_record(3.0, 4950.0, 0.02, 0.9, 0.35, 0.79, 0.88),
    ]

    results = orchestrate_delta_metrics(
        [segment_a, segment_b],
        target_delta_nfr=0.0,
        target_sense_index=0.9,
    )

    assert results["objectives"]["sense_index"] == pytest.approx(0.9)
    assert len(results["bundles"]) == 4
    assert len(results["delta_nfr_series"]) == 4
    assert len(results["sense_index_series"]) == 4
    assert results["dissonance"] >= 0.0
    assert isinstance(results["dissonance_breakdown"], DissonanceBreakdown)
    assert results["dissonance_breakdown"].value == pytest.approx(results["dissonance"])
    assert -1.0 <= results["coupling"] <= 1.0
    assert 0.0 <= results["resonance"] <= 1.0
    assert len(results["recursive_trace"]) == 4
    assert set(results["pairwise_coupling"]) == {"delta_nfr", "sense_index"}
    for metrics in results["pairwise_coupling"].values():
        assert {"tyres↔suspension", "tyres↔chassis", "suspension↔chassis"} <= set(metrics)
        for value in metrics.values():
            assert -1.0 <= value <= 1.0


def test_orchestrator_consumes_fixture_segments(synthetic_records):
    segments = [synthetic_records[:9], synthetic_records[9:]]

    report = orchestrate_delta_metrics(
        segments,
        target_delta_nfr=0.5,
        target_sense_index=0.82,
    )

    assert report["objectives"]["delta_nfr"] == pytest.approx(0.5)
    assert len(report["bundles"]) == len(synthetic_records)
    assert pytest.approx(report["sense_index"], rel=1e-6) == mean(report["sense_index_series"])
    assert len(report["recursive_trace"]) == len(synthetic_records)
    assert "pairwise_coupling" in report
    assert isinstance(report["dissonance_breakdown"], DissonanceBreakdown)
    assert report["dissonance_breakdown"].value == pytest.approx(report["dissonance"])


def test_orchestrator_reports_microsector_variability(monkeypatch):
    segment_a = [_build_record(0.0, 5000.0, 0.1, 0.5, 0.2, 0.5, 0.8)] * 2
    segment_b = [_build_record(2.0, 4800.0, 0.2, 0.6, 0.25, 0.4, 0.78)] * 2
    delta_values = [[0.1, 0.3], [0.2, 0.4]]
    si_values = [[0.8, 0.82], [0.78, 0.76]]

    def _fake_recepcion(segment):
        index = _fake_recepcion.call_count
        _fake_recepcion.call_count += 1
        bundles: List[EPIBundle] = []
        timestamp = 0.0
        for delta, si in zip(delta_values[index], si_values[index]):
            bundles.append(
                EPIBundle(
                    timestamp=timestamp,
                    epi=0.0,
                    delta_nfr=delta,
                    sense_index=si,
                    tyres=TyresNode(delta_nfr=delta, sense_index=si),
                    suspension=SuspensionNode(delta_nfr=delta, sense_index=si),
                    chassis=ChassisNode(delta_nfr=delta, sense_index=si),
                    brakes=BrakesNode(delta_nfr=0.0, sense_index=si),
                    transmission=TransmissionNode(delta_nfr=0.0, sense_index=si),
                    track=TrackNode(delta_nfr=0.0, sense_index=si),
                    driver=DriverNode(delta_nfr=0.0, sense_index=si),
                )
            )
            timestamp += 1.0
        return bundles

    _fake_recepcion.call_count = 0
    monkeypatch.setattr("tnfr_lfs.core.operators.recepcion_operator", _fake_recepcion)

    microsectors = [
        _build_microsector(0, 0, 1, 2, apex_target=0.0),
        _build_microsector(1, 1, 2, 3, apex_target=0.0),
    ]

    results = orchestrate_delta_metrics(
        [segment_a, segment_b],
        target_delta_nfr=0.0,
        target_sense_index=0.8,
        coherence_window=1,
        microsectors=microsectors,
    )

    variability = results["microsector_variability"]
    assert len(variability) == 2
    first = variability[0]
    assert first["overall"]["samples"] == 3
    assert first["overall"]["delta_nfr"]["variance"] == pytest.approx(
        pvariance([0.1, 0.3, 0.2])
    )
    assert first["overall"]["sense_index"]["variance"] == pytest.approx(
        pvariance([0.8, 0.82, 0.78])
    )
    assert set(first["laps"]) == {"Vuelta 1", "Vuelta 2"}
    assert first["laps"]["Vuelta 1"]["samples"] == 2
    assert first["laps"]["Vuelta 1"]["delta_nfr"]["variance"] == pytest.approx(0.01)
    assert first["laps"]["Vuelta 2"]["samples"] == 1
    assert first["laps"]["Vuelta 2"]["delta_nfr"]["variance"] == pytest.approx(0.0)

def test_dissonance_breakdown_identifies_useful_and_parasitic_events():
    bundles = [
        _build_bundle(0.0, 0.1),
        _build_bundle(0.1, 0.6),
        _build_bundle(0.2, 0.2),
        _build_bundle(0.3, -0.4),
        _build_bundle(0.4, -0.1),
    ]
    microsectors = [
        _build_microsector(0, 0, 1, 2, apex_target=0.5),
        _build_microsector(1, 2, 3, 4, apex_target=-0.1),
    ]
    series = [bundle.delta_nfr for bundle in bundles]

    breakdown = dissonance_breakdown_operator(
        series,
        target=0.0,
        microsectors=microsectors,
        bundles=bundles,
    )

    assert isinstance(breakdown, DissonanceBreakdown)
    assert breakdown.value == pytest.approx(dissonance_operator(series, target=0.0))
    assert breakdown.total_events == 2
    assert breakdown.useful_events == 1
    assert breakdown.parasitic_events == 1
    assert breakdown.useful_magnitude == pytest.approx(0.1)
    assert breakdown.parasitic_magnitude == pytest.approx(0.3)
    assert breakdown.useful_percentage == pytest.approx(25.0)
    assert breakdown.parasitic_percentage == pytest.approx(75.0)


def test_emission_operator_clamps_sense_index():
    objectives = emission_operator(target_delta_nfr=0.5, target_sense_index=1.5)

    assert objectives["delta_nfr"] == pytest.approx(0.5)
    assert objectives["sense_index"] == 1.0


def test_recepcion_operator_wraps_epi_extractor():
    records = [
        _build_record(0.0, 5000.0, 0.1, 1.0, 0.5, 0.8, 0.9),
        _build_record(1.0, 5050.0, 0.09, 1.1, 0.6, 0.81, 0.91),
    ]

    bundles = recepcion_operator(records)

    assert len(bundles) == 2
    assert isinstance(bundles[0].epi, float)


def test_recursividad_operator_requires_decay_in_range():
    with pytest.raises(ValueError):
        recursividad_operator([0.1, 0.2], decay=1.0)


def test_recursivity_operator_tracks_state_and_phase_changes():
    state: dict[str, dict[str, object]] = {}

    first = recursivity_operator(
        state,
        "ms-1",
        {"thermal_load": 420.0, "style_index": 0.82, "phase": "entry"},
        decay=0.5,
    )
    assert first["filtered"]["thermal_load"] == pytest.approx(420.0)
    assert not first["phase_changed"]

    second = recursivity_operator(
        state,
        "ms-1",
        {"thermal_load": 520.0, "style_index": 0.72, "phase": "entry"},
        decay=0.5,
    )
    assert second["filtered"]["thermal_load"] == pytest.approx(470.0)
    assert not second["phase_changed"]

    third = recursivity_operator(
        state,
        "ms-1",
        {"thermal_load": 360.0, "style_index": 0.9, "phase": "apex"},
        decay=0.5,
    )
    assert third["phase_changed"]
    assert third["filtered"]["thermal_load"] == pytest.approx(360.0)
    assert state["ms-1"]["trace"][-1]["phase"] == "apex"

    other = recursivity_operator(
        state,
        "ms-2",
        {"thermal_load": 300.0, "style_index": 0.95, "phase": "entry"},
        decay=0.5,
    )
    assert other["filtered"]["thermal_load"] == pytest.approx(300.0)
    assert len(state) == 2


def test_mutation_operator_detects_style_and_entropy_mutations():
    state: dict[str, dict[str, object]] = {}
    base_triggers = {
        "microsector_id": "ms-5",
        "current_archetype": "equilibrio",
        "candidate_archetype": "apoyo",
        "fallback_archetype": "recuperacion",
        "entropy": 0.3,
        "style_index": 0.82,
        "style_reference": 0.82,
        "phase": "entry",
    }

    initial = mutation_operator(state, base_triggers)
    assert not initial["mutated"]
    assert initial["archetype"] == "equilibrio"

    style_shift = mutation_operator(
        state,
        {**base_triggers, "style_index": 0.55, "dynamic_conditions": True},
    )
    assert style_shift["mutated"]
    assert style_shift["archetype"] == "apoyo"

    entropy_spike = mutation_operator(
        state,
        {
            **base_triggers,
            "current_archetype": style_shift["archetype"],
            "entropy": 0.92,
            "style_index": 0.58,
        },
    )
    assert entropy_spike["mutated"]
    assert entropy_spike["archetype"] == "recuperacion"

    phase_adjustment = mutation_operator(
        state,
        {
            **base_triggers,
            "current_archetype": entropy_spike["archetype"],
            "style_index": 0.7,
            "phase": "apex",
        },
    )
    assert phase_adjustment["mutated"]
    assert phase_adjustment["archetype"] == "apoyo"
    assert state["ms-5"]["phase"] == "apex"


def test_acoplamiento_and_resonance_behaviour():
    series_a = [0.1, 0.2, 0.3, 0.4]
    series_b = [0.1, 0.15, 0.25, 0.35]

    coupling = acoplamiento_operator(series_a, series_b)
    resonance = resonance_operator(series_b)
    dissonance = dissonance_operator(series_a, target=0.25)

    assert coupling > 0
    expected_resonance = sqrt(mean(value * value for value in series_b))
    assert resonance == pytest.approx(expected_resonance, rel=1e-9)
    assert dissonance == pytest.approx(mean(abs(value - 0.25) for value in series_a))


def test_pairwise_coupling_operator_builds_expected_matrix():
    series = {
        "tyres": [1.0, 2.0, 3.0, 4.0],
        "suspension": [2.0, 4.0, 6.0, 8.0],
        "chassis": [4.0, 3.0, 2.0, 1.0],
    }

    pairwise = pairwise_coupling_operator(series)

    assert pytest.approx(pairwise["tyres↔suspension"], rel=1e-9) == 1.0
    assert pytest.approx(pairwise["tyres↔chassis"], rel=1e-9) == -1.0
    assert pytest.approx(pairwise["suspension↔chassis"], rel=1e-9) == -1.0


def test_pairwise_coupling_operator_allows_unbalanced_lengths():
    series = {
        "tyres": [1.0, 2.0, 3.0],
        "suspension": [1.5, 2.5],
    }

    pairwise = pairwise_coupling_operator(series)

    expected = acoplamiento_operator(series["tyres"], series["suspension"], strict_length=False)
    assert pairwise["tyres↔suspension"] == pytest.approx(expected, rel=1e-9)

