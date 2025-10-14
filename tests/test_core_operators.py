"""Unit tests for the high-level TNFR × LFS operators."""

from __future__ import annotations

from dataclasses import replace
from math import cos, pi, sin, sqrt
from statistics import mean, pstdev, pvariance
from typing import List, Mapping, Sequence

import numpy as np
import pytest

from tests.helpers import (
    build_dynamic_record,
    build_goal,
    build_microsector,
    build_operator_bundle,
    clone_protocol_series,
)

from tnfr_core import Microsector, TelemetryRecord, phase_synchrony_index
from tnfr_core.spectrum import phase_to_latency_ms
from tnfr_core.coherence import sense_index
from tnfr_core.contextual_delta import (
    apply_contextual_delta,
    load_context_matrix,
    resolve_series_context,
)
from tnfr_core.epi import (
    DEFAULT_PHASE_WEIGHTS,
    DeltaCalculator,
    delta_nfr_by_node,
    resolve_nu_f_by_node,
)
from tnfr_core.epi_models import (
    BrakesNode,
    ChassisNode,
    DriverNode,
    EPIBundle,
    SuspensionNode,
    TrackNode,
    TransmissionNode,
    TyresNode,
)
from tnfr_core.phases import PHASE_SEQUENCE, expand_phase_alias
from tnfr_core.constants import WHEEL_SUFFIXES
from tnfr_core.operators import (
    DissonanceBreakdown,
    acoplamiento_operator,
    coupling_operator,
    coherence_operator,
    dissonance_breakdown_operator,
    dissonance_operator,
    _aggregate_operator_events,
    _stage_coherence,
    _stage_epi_evolution,
    evolve_epi,
    emission_operator,
    mutation_operator,
    orchestrate_delta_metrics,
    pairwise_coupling_operator,
    reception_operator,
    recursivity_operator,
    recursive_filter_operator,
    recursividad_operator,
    resonance_operator,
    TyreBalanceControlOutput,
    tyre_balance_controller,
)
from tnfr_core.metrics import compute_window_metrics
from tnfr_core.operator_detection import canonical_operator_label
from tnfr_core.interfaces import SupportsEPIBundle


CASES = [
    {
        "id": "test_delta_calculator_decomposes_longitudinal_component",
        "baseline": {
            "timestamp": 0.0,
            "vertical_load": 4000.0,
            "slip_ratio": 0.0,
            "lateral_accel": 1.0,
            "longitudinal_accel": 0.2,
            "nfr": 5.0,
            "si": 0.8,
            "brake_pressure": 20.0,
            "speed": 40.0,
        },
        "sample": {
            "timestamp": 0.1,
            "vertical_load": 4200.0,
            "slip_ratio": 0.05,
            "lateral_accel": 1.1,
            "longitudinal_accel": 1.2,
            "nfr": 6.5,
            "si": 0.78,
            "brake_pressure": 60.0,
            "speed": 42.0,
        },
        "dominant": "longitudinal",
        "delta_nfr": 1.5,
    },
    {
        "id": "test_delta_calculator_decomposes_lateral_component",
        "baseline": {
            "timestamp": 0.0,
            "vertical_load": 4100.0,
            "slip_ratio": 0.02,
            "lateral_accel": 0.8,
            "longitudinal_accel": 0.3,
            "nfr": 4.0,
            "si": 0.82,
            "steer": 0.02,
            "yaw_rate": 0.1,
        },
        "sample": {
            "timestamp": 0.1,
            "vertical_load": 4100.0,
            "slip_ratio": 0.02,
            "lateral_accel": 1.8,
            "longitudinal_accel": 0.3,
            "nfr": 5.2,
            "si": 0.81,
            "steer": 0.4,
            "yaw_rate": 0.4,
            "slip_angle": 0.3,
        },
        "dominant": "lateral",
        "delta_nfr": 1.2,
    },
]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_delta_calculator_decomposes_components(case):
    baseline = build_dynamic_record(**case["baseline"])
    sample = replace(baseline, **case["sample"], reference=baseline)
    bundle = DeltaCalculator.compute_bundle(sample, baseline, epi_value=0.0)

    assert bundle.delta_nfr == pytest.approx(case["delta_nfr"], rel=1e-3)
    assert bundle.delta_nfr_proj_longitudinal + bundle.delta_nfr_proj_lateral == pytest.approx(
        bundle.delta_nfr,
        rel=1e-6,
    )

    if case["dominant"] == "longitudinal":
        assert abs(bundle.delta_nfr_proj_longitudinal) > abs(bundle.delta_nfr_proj_lateral)
    else:
        assert abs(bundle.delta_nfr_proj_lateral) > abs(bundle.delta_nfr_proj_longitudinal)


def test_stage_epi_evolution_accepts_protocol_samples():
    first = build_dynamic_record(
        timestamp=0.0,
        vertical_load=4050.0,
        slip_ratio=0.03,
        lateral_accel=0.6,
        longitudinal_accel=0.25,
        nfr=5.0,
        si=0.78,
        throttle=0.4,
        steer=0.1,
        yaw_rate=0.12,
    )
    second = replace(
        first,
        timestamp=0.1,
        slip_ratio=0.06,
        lateral_accel=0.95,
        longitudinal_accel=0.35,
        nfr=6.2,
        si=0.8,
        throttle=0.45,
        steer=0.18,
        yaw_rate=0.2,
        reference=first,
    )
    third = replace(
        second,
        timestamp=0.2,
        slip_ratio=0.08,
        lateral_accel=1.1,
        longitudinal_accel=0.5,
        nfr=7.1,
        si=0.82,
        throttle=0.48,
        steer=0.22,
        yaw_rate=0.28,
        reference=second,
    )

    records = [first, second, third]

    expected = _stage_epi_evolution(records)
    protocol_records = clone_protocol_series(records)
    result = _stage_epi_evolution(protocol_records)

    assert result.keys() == expected.keys()

    for series_key in ("integrated", "derivative"):
        assert len(result[series_key]) == len(expected[series_key])
        for actual, reference in zip(result[series_key], expected[series_key]):
            assert actual == pytest.approx(reference)

    for node_map_key in ("per_node_integrated", "per_node_derivative"):
        assert result[node_map_key].keys() == expected[node_map_key].keys()
        for node, reference_series in expected[node_map_key].items():
            actual_series = result[node_map_key][node]
            assert len(actual_series) == len(reference_series)
            for actual, reference in zip(actual_series, reference_series):
                assert actual == pytest.approx(reference)


def _build_microsector(
    index: int,
    entry_idx: int,
    apex_idx: int,
    exit_idx: int,
    *,
    apex_target: float,
    support_event: bool = True,
    archetype: str = "hairpin",
    phase_synchrony_map: Mapping[str, float] | None = None,
    cphi_values: Mapping[str, float] | None = None,
) -> Microsector:
    synchrony_payload = None
    if phase_synchrony_map is not None:
        synchrony_payload = {
            phase: float(phase_synchrony_map.get(phase, 1.0))
            for phase in PHASE_SEQUENCE
        }

    kwargs: dict[str, object] = {
        "index": index,
        "entry_index": entry_idx,
        "apex_index": apex_idx,
        "exit_index": exit_idx,
        "support_event": support_event,
        "archetype": archetype,
        "apex_target": apex_target,
    }
    if synchrony_payload is not None:
        kwargs["phase_synchrony"] = synchrony_payload
    if cphi_values is not None:
        kwargs["cphi_values"] = cphi_values
    return build_microsector(**kwargs)


class _ProtocolBundleStub:
    """Minimal stub implementing :class:`SupportsEPIBundle`."""

    def __init__(self, timestamp: float, delta_nfr: float, sense_index: float) -> None:
        self.timestamp = timestamp
        self.epi = 0.0
        self.delta_nfr = delta_nfr
        self.sense_index = sense_index
        self.tyres = TyresNode(delta_nfr=delta_nfr, sense_index=sense_index)
        self.suspension = SuspensionNode(delta_nfr=delta_nfr, sense_index=sense_index)
        self.chassis = ChassisNode(delta_nfr=delta_nfr, sense_index=sense_index)
        self.brakes = BrakesNode(delta_nfr=0.0, sense_index=sense_index)
        self.transmission = TransmissionNode(delta_nfr=0.0, sense_index=sense_index)
        self.track = TrackNode(delta_nfr=0.0, sense_index=sense_index)
        self.driver = DriverNode(delta_nfr=0.0, sense_index=sense_index)
        self.structural_timestamp = None
        self.delta_breakdown: Mapping[str, Mapping[str, float]] = {}
        self.dEPI_dt = 0.0
        self.integrated_epi = 0.0
        self.node_evolution: Mapping[str, tuple[float, float]] = {}
        self.delta_nfr_proj_longitudinal = 0.0
        self.delta_nfr_proj_lateral = 0.0
        self.nu_f_classification = ""
        self.nu_f_category = ""
        self.nu_f_label = ""
        self.nu_f_dominant = 0.0
        self.coherence_index = 0.0
        self.ackermann_parallel_index = 0.0


def _reference_coherence(series: Sequence[float], window: int) -> List[float]:
    half_window = window // 2
    smoothed: List[float] = []
    for index in range(len(series)):
        start = max(0, index - half_window)
        end = min(len(series), index + half_window + 1)
        window_slice = series[start:end]
        smoothed.append(sum(window_slice) / len(window_slice))
    original_mean = sum(series) / len(series)
    smoothed_mean = sum(smoothed) / len(smoothed)
    bias = original_mean - smoothed_mean
    if abs(bias) < 1e-12:
        return smoothed
    return [value + bias for value in smoothed]


def test_coherence_operator_window_of_one_preserves_series() -> None:
    series = [0.1, -0.4, 0.9, 0.0, -0.2]

    result = coherence_operator(series, window=1)

    assert isinstance(result, list)
    assert result == pytest.approx(series, abs=1e-12)



def test_coherence_operator_reduces_jitter_without_bias():
    raw_series = [0.2, 0.8, 0.1, 0.9, 0.2]
    smoothed = coherence_operator(raw_series, window=3)

    assert mean(smoothed) == pytest.approx(mean(raw_series), rel=1e-9)
    assert pstdev(smoothed) < pstdev(raw_series)



def test_coherence_operator_matches_reference_for_large_series() -> None:
    rng = np.random.default_rng(42)
    series = rng.normal(size=5000).tolist()

    result = coherence_operator(series, window=5)
    expected = _reference_coherence(series, window=5)

    assert len(result) == len(series)
    assert result == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_stage_coherence_accepts_protocol_bundles() -> None:
    bundles: List[SupportsEPIBundle] = [
        _ProtocolBundleStub(0.0, 0.12, 0.62),
        _ProtocolBundleStub(0.5, -0.08, 0.58),
        _ProtocolBundleStub(1.0, 0.04, 0.61),
    ]
    objectives = emission_operator(0.0, 0.6)

    stage = _stage_coherence(bundles, objectives, coherence_window=1)

    assert stage["bundles"], "expected updated bundles returned"
    assert all(isinstance(bundle, EPIBundle) for bundle in stage["bundles"])
    assert len(stage["smoothed_delta"]) == len(bundles)
    assert len(stage["smoothed_sense_index"]) == len(bundles)
    for updated, original in zip(stage["smoothed_delta"], bundles):
        assert updated == pytest.approx(original.delta_nfr, rel=0.15)
    for updated, original in zip(stage["smoothed_sense_index"], bundles):
        assert 0.0 <= updated <= 1.0
        assert updated == pytest.approx(original.sense_index, rel=0.05)
    # ensure original protocol bundle remains unchanged
    assert isinstance(bundles[0], _ProtocolBundleStub)
    assert bundles[0].delta_nfr == pytest.approx(0.12)


def test_dissonance_breakdown_handles_protocol_bundles() -> None:
    bundles: List[SupportsEPIBundle] = [
        _ProtocolBundleStub(0.0, 0.2, 0.5),
        _ProtocolBundleStub(0.5, -0.1, 0.55),
        _ProtocolBundleStub(1.0, 0.0, 0.52),
    ]
    series = [bundle.delta_nfr for bundle in bundles]

    breakdown = dissonance_breakdown_operator(
        series,
        0.0,
        microsectors=None,
        bundles=bundles,
    )

    assert isinstance(breakdown, DissonanceBreakdown)
    assert breakdown.value == pytest.approx(mean(abs(value) for value in series))


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
    assert getattr(nodal, "metadata", {}) == {}


def test_evolve_epi_applies_phase_metadata():
    prev = 0.0
    deltas = {
        "tyres": 1.0,
        "driver": -0.5,
        "__theta__": "apex",
        "__w_phase__": {
            "apex": {"__default__": 0.75, "tyres": 2.0},
            "__default__": {"__default__": 1.0},
        },
        "nu_f_objectives": {"driver": 0.2},
    }
    nu_map = {"tyres": 0.2, "driver": 0.1}

    new_epi, derivative, nodal = evolve_epi(prev, deltas, 0.1, nu_map)

    tyres_weight = 0.2 * 2.0
    driver_weight = 0.2 * 0.75
    expected_derivative = tyres_weight * 1.0 + driver_weight * -0.5

    assert derivative == pytest.approx(expected_derivative, rel=1e-9)
    assert new_epi == pytest.approx(expected_derivative * 0.1, rel=1e-9)
    assert nodal["tyres"][1] == pytest.approx(tyres_weight * 1.0, rel=1e-9)
    assert nodal["driver"][1] == pytest.approx(driver_weight * -0.5, rel=1e-9)

    metadata = getattr(nodal, "metadata", {})
    assert metadata.get("theta") == "apex"
    assert metadata.get("theta_effect", {}).get("tyres") == pytest.approx(2.0)
    assert metadata.get("theta_effect", {}).get("driver") == pytest.approx(0.75)
    assert metadata.get("nu_f_objectives", {}).get("driver") == pytest.approx(0.2)


def test_sense_index_penalises_active_phase_weights():
    baseline = build_dynamic_record(0.0, 5000.0, 0.02, 0.9, 0.3, 0.8, 0.92)
    sample = build_dynamic_record(0.1, 5450.0, 0.18, 1.4, -0.6, 0.86, 0.78)
    node_deltas = delta_nfr_by_node(replace(sample, reference=baseline))
    weights = {
        "entry": {"__default__": 1.0},
        "apex": {"__default__": 1.0, "tyres": 2.0, "chassis": 1.6},
        "exit": {"__default__": 1.0},
    }
    nu_entry = resolve_nu_f_by_node(sample, phase="entry", phase_weights=weights).by_node
    nu_apex = resolve_nu_f_by_node(sample, phase="apex", phase_weights=weights).by_node

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
        nu_f_by_node=resolve_nu_f_by_node(sample).by_node,
        active_phase="apex",
        w_phase=DEFAULT_PHASE_WEIGHTS,
    )

    assert apex_index < entry_index
    assert apex_index <= neutral_index


def test_orchestrator_pipeline_builds_consistent_metrics():
    segment_a = [
        build_dynamic_record(0.0, 5200.0, 0.05, 1.2, 0.6, 0.82, 0.91),
        build_dynamic_record(1.0, 5100.0, 0.04, 1.1, 0.5, 0.81, 0.92),
    ]
    segment_b = [
        build_dynamic_record(2.0, 5000.0, 0.03, 1.0, 0.4, 0.80, 0.90),
        build_dynamic_record(3.0, 4950.0, 0.02, 0.9, 0.35, 0.79, 0.88),
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
    assert results["support_effective"] >= 0.0
    assert results["load_support_ratio"] >= 0.0
    assert results["structural_expansion_longitudinal"] >= 0.0
    assert results["structural_contraction_longitudinal"] >= 0.0
    assert results["structural_expansion_lateral"] >= 0.0
    assert results["structural_contraction_lateral"] >= 0.0
    stages = results["stages"]
    assert set(stages) == {"reception", "coherence", "nodal", "epi", "sense"}
    reception_stage = stages["reception"]
    assert reception_stage["sample_count"] == 4
    assert len(reception_stage["lap_indices"]) == 4
    coherence_stage = stages["coherence"]
    assert len(coherence_stage["raw_delta"]) == reception_stage["sample_count"]
    assert coherence_stage["bundles"] == results["bundles"]
    epi_stage = stages["epi"]
    assert results["epi_evolution"] == epi_stage
    assert len(epi_stage["integrated"]) == reception_stage["sample_count"]
    nodal_stage = stages["nodal"]
    assert results["nodal_metrics"] == nodal_stage
    assert set(nodal_stage["delta_by_node"]) == {"tyres", "suspension", "chassis"}
    sense_stage = stages["sense"]
    assert results["sense_memory"] == sense_stage
    assert len(sense_stage["series"]) == reception_stage["sample_count"]


def test_window_metrics_phase_alignment_tracks_cross_spectrum():
    frequency = 1.0
    phase_offset = pi / 6
    records: List[TelemetryRecord] = []
    for index in range(64):
        timestamp = index * 0.05
        steer_value = sin(2.0 * pi * frequency * timestamp)
        response_value = sin(2.0 * pi * frequency * timestamp - phase_offset)
        records.append(
            build_dynamic_record(
                timestamp,
                5000.0,
                0.02,
                response_value,
                0.0,
                100.0,
                0.82,
                yaw_rate=response_value,
                steer=steer_value,
            )
        )

    metrics = compute_window_metrics(records, phase_indices=range(len(records)))

    assert metrics.nu_f == pytest.approx(frequency, abs=0.1)
    assert abs(metrics.phase_alignment - cos(phase_offset)) < 0.15
    assert abs(abs(metrics.phase_lag) - phase_offset) < 0.25
    expected_sync = phase_synchrony_index(phase_offset, cos(phase_offset))
    assert metrics.phase_synchrony_index == pytest.approx(expected_sync, abs=0.05)
    expected_latency_ms = abs(phase_to_latency_ms(frequency, phase_offset))
    assert metrics.motor_latency_ms == pytest.approx(expected_latency_ms, abs=6.0)
    assert metrics.phase_motor_latency_ms
    first_phase_latency = next(iter(metrics.phase_motor_latency_ms.values()))
    assert first_phase_latency == pytest.approx(expected_latency_ms, abs=6.0)


def test_orchestrator_respects_phase_weight_overrides():
    raw_records = [
        build_dynamic_record(0.0, 5200.0, 0.06, 1.2, 0.4, 500.0, 0.88),
        build_dynamic_record(0.1, 5300.0, 0.08, 1.3, 0.35, 502.0, 0.86),
    ]
    baseline = build_dynamic_record(0.0, 5100.0, 0.02, 1.0, 0.2, 498.0, 0.9)
    records = [replace(sample, reference=baseline) for sample in raw_records]
    base_weights = {
        "entry": {"__default__": 1.0, "tyres": 1.0},
        "apex": {"__default__": 1.0},
        "exit": {"__default__": 1.0},
    }
    boosted_weights = {
        "entry": {"__default__": 1.0, "tyres": 2.0},
        "apex": {"__default__": 1.0},
        "exit": {"__default__": 1.0},
    }
    phases = ("entry", "apex", "exit")
    goals = tuple(build_goal(phase, 0.0) for phase in phases)
    base_microsector = build_microsector(
        index=0,
        start_time=records[0].timestamp,
        end_time=records[-1].timestamp,
        curvature=1.5,
        brake_event=False,
        support_event=False,
        delta_nfr_signature=0.0,
        goals=goals,
        phases=phases,
        phase_boundaries={"entry": (0, 2), "apex": (2, 2), "exit": (2, 2)},
        phase_samples={"entry": (0, 1), "apex": (), "exit": ()},
        active_phase="entry",
        dominant_nodes={phase: ("tyres",) for phase in phases},
        phase_weights=base_weights,
        filtered_measures={"thermal_load": 5200.0, "style_index": 0.9, "grip_rel": 1.0},
        window_occupancy={
            phase: {"slip_lat": 0.0, "slip_long": 0.0, "yaw_rate": 0.0}
            for phase in phases
        },
        phase_lag={phase: 0.0 for phase in phases},
        phase_alignment={phase: 1.0 for phase in phases},
        phase_synchrony={phase: 1.0 for phase in phases},
        include_cphi=False,
    )
    boosted_microsector = replace(base_microsector, phase_weights=boosted_weights)

    base_metrics = orchestrate_delta_metrics(
        [records],
        target_delta_nfr=0.0,
        target_sense_index=0.9,
        microsectors=[base_microsector],
        phase_weights=base_weights,
    )
    boosted_metrics = orchestrate_delta_metrics(
        [records],
        target_delta_nfr=0.0,
        target_sense_index=0.9,
        microsectors=[boosted_microsector],
        phase_weights=boosted_weights,
    )

    base_series = base_metrics["epi_evolution"]["per_node_derivative"]["tyres"]
    boosted_series = boosted_metrics["epi_evolution"]["per_node_derivative"]["tyres"]
    assert boosted_series[0] > base_series[0]


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
    epi_stage = report["epi_evolution"]
    assert len(epi_stage["integrated"]) == len(synthetic_records)
    assert len(epi_stage["derivative"]) == len(synthetic_records)
    nodal_stage = report["nodal_metrics"]
    assert all(len(series) == len(synthetic_records) for series in nodal_stage["delta_by_node"].values())
    assert report["sense_memory"]["memory"] == report["recursive_trace"]


def test_orchestrator_reports_microsector_variability(monkeypatch):
    segment_a = [build_dynamic_record(0.0, 5000.0, 0.1, 0.5, 0.2, 0.5, 0.8)] * 2
    segment_b = [build_dynamic_record(2.0, 4800.0, 0.2, 0.6, 0.25, 0.4, 0.78)] * 2
    delta_values = [[0.1, 0.3], [0.2, 0.4]]
    si_values = [[0.8, 0.82], [0.78, 0.76]]

    def _fake_reception(segment):
        index = _fake_reception.call_count
        _fake_reception.call_count += 1
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

    _fake_reception.call_count = 0
    monkeypatch.setattr(
        "tnfr_core.operators.operators.reception_operator",
        _fake_reception,
    )

    phase_sync_a = {
        phase: 0.85 + index * 0.02 for index, phase in enumerate(PHASE_SEQUENCE)
    }
    cphi_a = {suffix: 0.65 + i * 0.01 for i, suffix in enumerate(WHEEL_SUFFIXES)}
    phase_sync_b = {
        phase: 0.8 + index * 0.015 for index, phase in enumerate(PHASE_SEQUENCE)
    }
    cphi_b = {suffix: 0.6 + i * 0.02 for i, suffix in enumerate(WHEEL_SUFFIXES)}
    microsectors = [
        _build_microsector(
            0,
            0,
            1,
            2,
            apex_target=0.0,
            phase_synchrony_map=phase_sync_a,
            cphi_values=cphi_a,
        ),
        _build_microsector(
            1,
            1,
            2,
            3,
            apex_target=0.0,
            phase_synchrony_map=phase_sync_b,
            cphi_values=cphi_b,
        ),
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
    coherence_stage = results["stages"]["coherence"]
    coherence_bundles = coherence_stage["bundles"]
    sample_indices = sorted(
        {
            idx
            for samples in microsectors[0].phase_samples.values()
            for idx in samples
        }
    )
    delta_samples = [coherence_bundles[idx].delta_nfr for idx in sample_indices]
    expected_delta_variance = pvariance(delta_samples)
    assert first["overall"]["delta_nfr"]["variance"] == pytest.approx(
        expected_delta_variance
    )
    sense_samples = [0.8, 0.82, 0.78]
    assert first["overall"]["sense_index"]["variance"] == pytest.approx(
        pvariance(sense_samples)
    )
    assert first["overall"]["sense_index"]["mean"] == pytest.approx(
        mean(sense_samples)
    )
    expected_si_stdev = sqrt(pvariance(sense_samples))
    expected_si_cov = expected_si_stdev / max(abs(mean(sense_samples)), 1e-9)
    assert first["overall"]["sense_index"]["coefficient_of_variation"] == pytest.approx(
        expected_si_cov
    )
    expected_stability = 1.0 - min(1.0, expected_si_cov / 0.15)
    assert first["overall"]["sense_index"]["stability_score"] == pytest.approx(
        max(0.0, expected_stability)
    )
    integral_samples = []
    timestamps = [coherence_bundles[idx].timestamp for idx in sample_indices]
    for pos, idx in enumerate(sample_indices):
        if pos + 1 < len(sample_indices):
            dt = timestamps[pos + 1] - timestamps[pos]
        elif pos > 0:
            dt = timestamps[pos] - timestamps[pos - 1]
        else:
            dt = 1.0
        if dt <= 0.0:
            dt = 1.0
        integral_samples.append(abs(coherence_bundles[idx].delta_nfr) * dt)
    assert first["overall"]["delta_nfr_integral"]["mean"] == pytest.approx(
        mean(integral_samples)
    )
    assert first["overall"]["phase_synchrony"]["mean"] == pytest.approx(
        mean(phase_sync_a.values())
    )
    assert first["overall"]["cphi"]["mean"] == pytest.approx(mean(cphi_a.values()))
    assert set(first["laps"]) == {"Lap 1", "Lap 2"}
    assert first["laps"]["Lap 1"]["samples"] == 2
    lap_metrics = first["laps"]["Lap 1"]
    assert "delta_nfr_integral" in lap_metrics
    assert lap_metrics["cphi"]["mean"] == pytest.approx(
        first["overall"]["cphi"]["mean"]
    )
    reception_stage = results["stages"]["reception"]
    lap_indices = reception_stage.get("lap_indices", [])
    lap_sequence = results["lap_sequence"]
    for lap_entry in lap_sequence:
        lap_label = str(lap_entry.get("label", lap_entry.get("index")))
        lap_index = int(lap_entry.get("index", 0))
        if lap_label not in first["laps"]:
            continue
        lap_specific_indices = [
            idx
            for idx in sample_indices
            if idx < len(lap_indices) and lap_indices[idx] == lap_index
        ]
        expected_variance = pvariance(
            [coherence_bundles[idx].delta_nfr for idx in lap_specific_indices]
        )
        assert first["laps"][lap_label]["delta_nfr"]["variance"] == pytest.approx(
            expected_variance
        )
        assert first["laps"]["Lap 2"]["samples"] == 1

def test_dissonance_breakdown_identifies_useful_and_parasitic_events():
    bundles = [
        build_operator_bundle(timestamp=0.0, tyre_delta=0.1, yaw_rate=0.0),
        build_operator_bundle(timestamp=0.1, tyre_delta=0.6, yaw_rate=0.1),
        build_operator_bundle(timestamp=0.2, tyre_delta=0.2, yaw_rate=0.25),
        build_operator_bundle(timestamp=0.3, tyre_delta=-0.4, yaw_rate=0.3),
        build_operator_bundle(timestamp=0.4, tyre_delta=-0.1, yaw_rate=0.31),
    ]
    microsectors = [
        _build_microsector(0, 0, 1, 2, apex_target=0.5),
        _build_microsector(1, 2, 3, 4, apex_target=-0.1),
    ]
    matrix = load_context_matrix()
    contexts = resolve_series_context(bundles, matrix=matrix)
    adjusted_series = [
        apply_contextual_delta(
            bundle.delta_nfr,
            contexts[idx],
            context_matrix=matrix,
        )
        for idx, bundle in enumerate(bundles)
    ]

    breakdown = dissonance_breakdown_operator(
        adjusted_series,
        target=0.0,
        microsectors=microsectors,
        bundles=bundles,
    )

    assert isinstance(breakdown, DissonanceBreakdown)
    assert breakdown.value == pytest.approx(
        dissonance_operator(adjusted_series, target=0.0)
    )
    assert breakdown.total_events == 2
    assert breakdown.useful_events == 1
    assert breakdown.parasitic_events == 1
    multipliers = [
        max(
            matrix.min_multiplier,
            min(matrix.max_multiplier, contexts[idx].multiplier),
        )
        for idx in range(len(bundles))
    ]
    expected_useful = 0.0
    expected_parasitic = 0.0
    for microsector in microsectors:
        apex_goal = None
        for alias in expand_phase_alias("apex"):
            apex_goal = next(
                (goal for goal in microsector.goals if goal.phase == alias),
                None,
            )
            if apex_goal is not None:
                break
        if apex_goal is None:
            continue
        apex_indices: list[int] = []
        for alias in expand_phase_alias("apex"):
            apex_indices.extend(
                idx
                for idx in microsector.phase_samples.get(alias, ())
                if idx < len(bundles)
            )
        if not apex_indices:
            continue
        tyre_values = [
            bundles[idx].tyres.delta_nfr * multipliers[idx]
            for idx in apex_indices
        ]
        deviation = mean(tyre_values) - apex_goal.target_delta_nfr
        contribution = abs(deviation)
        if contribution <= 1e-12:
            continue
        if deviation >= 0.0:
            expected_useful += contribution
        else:
            expected_parasitic += contribution
    assert breakdown.useful_magnitude == pytest.approx(expected_useful)
    assert breakdown.parasitic_magnitude == pytest.approx(expected_parasitic)
    assert breakdown.useful_percentage == pytest.approx(61.53846153846154)
    assert breakdown.parasitic_percentage == pytest.approx(38.46153846153846)
    assert breakdown.high_yaw_acc_samples == 3
    assert breakdown.useful_dissonance_samples == 2
    assert breakdown.useful_dissonance_ratio == pytest.approx(2 / 3)
    assert breakdown.useful_dissonance_percentage == pytest.approx(66.66666666666666)


def test_emission_operator_clamps_sense_index():
    objectives = emission_operator(target_delta_nfr=0.5, target_sense_index=1.5)

    assert objectives["delta_nfr"] == pytest.approx(0.5)
    assert objectives["sense_index"] == 1.0


def test_reception_operator_wraps_epi_extractor():
    records = [
        build_dynamic_record(0.0, 5000.0, 0.1, 1.0, 0.5, 0.8, 0.9),
        build_dynamic_record(1.0, 5050.0, 0.09, 1.1, 0.6, 0.81, 0.91),
    ]

    bundles = reception_operator(records)

    assert len(bundles) == 2
    assert isinstance(bundles[0].epi, float)


def test_recursive_filter_operator_requires_decay_in_range():
    with pytest.raises(ValueError):
        recursive_filter_operator([0.1, 0.2], decay=1.0)


def test_recursividad_operator_alias_emits_deprecation_warning():
    series = [0.0, 0.5, 1.0]

    with pytest.deprecated_call():
        legacy = recursividad_operator(series, decay=0.3)

    expected = recursive_filter_operator(series, decay=0.3)
    assert legacy == expected


def test_recursivity_operator_tracks_state_and_phase_changes():
    state: dict[str, dict[str, object]] = {}
    session = ("FZR", "AS5", "soft")

    first = recursivity_operator(
        state,
        session,
        "ms-1",
        {"thermal_load": 420.0, "style_index": 0.82, "phase": "entry"},
        decay=0.5,
    )
    assert first["filtered"]["thermal_load"] == pytest.approx(420.0)
    assert not first["phase_changed"]

    second = recursivity_operator(
        state,
        session,
        "ms-1",
        {"thermal_load": 520.0, "style_index": 0.72, "phase": "entry"},
        decay=0.5,
    )
    assert second["filtered"]["thermal_load"] == pytest.approx(470.0)
    assert not second["phase_changed"]

    third = recursivity_operator(
        state,
        session,
        "ms-1",
        {"thermal_load": 360.0, "style_index": 0.9, "phase": "apex"},
        decay=0.5,
    )
    assert third["phase_changed"]
    assert third["filtered"]["thermal_load"] == pytest.approx(360.0)
    session_key = "|".join(session)
    active_state = state["sessions"][session_key]["active"]
    assert active_state["ms-1"]["trace"][-1]["phase"] == "apex"

    other = recursivity_operator(
        state,
        session,
        "ms-2",
        {"thermal_load": 300.0, "style_index": 0.95, "phase": "entry"},
        decay=0.5,
    )
    assert other["filtered"]["thermal_load"] == pytest.approx(300.0)
    assert len(active_state) == 2
def test_recursivity_operator_separates_sessions_and_tracks_history():
    state: dict[str, dict[str, object]] = {}
    session_a = ("FZR", "aston", "soft")
    session_b = ("XRR", "aston", "soft")

    recursivity_operator(
        state,
        session_a,
        "ms-1",
        {"thermal_load": 5000.0, "style_index": 0.65, "phase": "entry", "timestamp": 0.0},
        decay=0.4,
    )
    recursivity_operator(
        state,
        session_b,
        "ms-1",
        {"thermal_load": 5100.0, "style_index": 0.7, "phase": "entry", "timestamp": 0.0},
        decay=0.4,
    )

    assert set(state["sessions"]) == {"|".join(session_a), "|".join(session_b)}
    for session_key, entry in state["sessions"].items():
        assert entry["active"]
        assert not entry.get("history")


def test_recursivity_operator_rolls_over_on_limits():
    state: dict[str, dict[str, object]] = {}
    session = ("FZR", "aston", "soft")

    for step in range(3):
        recursivity_operator(
            state,
            session,
            "ms-1",
            {
                "thermal_load": 5000.0 + step,
                "style_index": 0.6 + (step * 0.005),
                "phase": "entry",
                "timestamp": float(step),
            },
            decay=0.3,
            max_samples=2,
            convergence_window=2,
            convergence_threshold=0.01,
        )

    session_entry = state["sessions"]["|".join(session)]
    assert session_entry["history"]
    last_history = session_entry["history"][0]
    assert last_history["reason"] in {"max_samples", "convergence"}
    assert last_history["samples"] >= 2
    assert "ms-1" in last_history["microsectors"]
    assert session_entry["active"]["ms-1"]["samples"] == 1
    assert session_entry["stint_index"] >= 1


def test_recursivity_operator_detects_time_gap_rollover():
    state: dict[str, dict[str, object]] = {}
    session = ("FZR", "aston", "soft")

    recursivity_operator(
        state,
        session,
        "ms-1",
        {
            "thermal_load": 5000.0,
            "style_index": 0.6,
            "phase": "entry",
            "timestamp": 1.0,
        },
        decay=0.3,
        max_time_gap=0.5,
    )
    recursivity_operator(
        state,
        session,
        "ms-1",
        {
            "thermal_load": 5010.0,
            "style_index": 0.61,
            "phase": "entry",
            "timestamp": 2.0,
        },
        decay=0.3,
        max_time_gap=0.5,
    )

    session_entry = state["sessions"]["|".join(session)]
    assert session_entry["history"]
    history_entry = session_entry["history"][0]
    assert history_entry["reason"] == "time_gap"


def test_tyre_balance_controller_computes_clamped_deltas():
    metrics = {
        "cphi_fl": 0.68,
        "cphi_fr": 0.74,
        "cphi_rl": 0.70,
        "cphi_rr": 0.76,
        "cphi_fl_temperature": 0.42,
        "cphi_fr_temperature": 0.35,
        "cphi_rl_temperature": 0.39,
        "cphi_rr_temperature": 0.33,
        "cphi_fl_gradient": 0.48,
        "cphi_fr_gradient": 0.36,
        "cphi_rl_gradient": 0.41,
        "cphi_rr_gradient": 0.34,
        "cphi_fl_mu": 0.55,
        "cphi_fr_mu": 0.50,
        "cphi_rl_mu": 0.52,
        "cphi_rr_mu": 0.47,
        "cphi_fl_temp_delta": 0.12,
        "cphi_fr_temp_delta": -0.05,
        "cphi_rl_temp_delta": 0.08,
        "cphi_rr_temp_delta": -0.02,
        "cphi_fl_gradient_rate": 0.58,
        "cphi_fr_gradient_rate": 0.44,
        "cphi_rl_gradient_rate": 0.52,
        "cphi_rr_gradient_rate": 0.40,
        "d_nfr_flat": -0.4,
    }

    control = tyre_balance_controller(metrics)
    assert isinstance(control, TyreBalanceControlOutput)
    assert control.pressure_delta_front == pytest.approx(-0.045, abs=1e-6)
    assert control.pressure_delta_rear == pytest.approx(-0.055, abs=1e-6)
    assert control.camber_delta_front == pytest.approx(-0.0756, abs=1e-6)
    assert control.camber_delta_rear == pytest.approx(-0.0675, abs=1e-6)
    assert control.per_wheel_pressure["fl"] == pytest.approx(-0.0402, abs=1e-6)
    assert control.per_wheel_pressure["fr"] == pytest.approx(-0.0470, abs=1e-6)
    assert control.per_wheel_pressure["rl"] == pytest.approx(-0.0518, abs=1e-6)
    assert control.per_wheel_pressure["rr"] == pytest.approx(-0.0558, abs=1e-6)

    with_offsets = tyre_balance_controller(
        metrics, offsets={"pressure_front": 0.05, "camber_rear": 0.05}
    )
    assert with_offsets.pressure_delta_front == pytest.approx(0.005, abs=1e-6)
    assert with_offsets.camber_delta_rear == pytest.approx(-0.0175, abs=1e-6)


def test_tyre_balance_controller_clips_per_wheel_biases():
    metrics = {
        "cphi_fl": 0.82,
        "cphi_fr": 0.82,
        "cphi_rl": 0.80,
        "cphi_rr": 0.80,
        "cphi_fl_gradient": 0.0,
        "cphi_fr_gradient": 0.0,
        "cphi_rl_gradient": 0.0,
        "cphi_rr_gradient": 0.0,
        "cphi_fl_temp_delta": 0.6,
        "cphi_fr_temp_delta": -0.8,
        "cphi_rl_temp_delta": 0.4,
        "cphi_rr_temp_delta": -0.7,
    }

    control = tyre_balance_controller(
        metrics,
        pressure_max_step=0.1,
        bias_gain=1.0,
        pressure_gain=0.0,
        camber_gain=0.0,
        nfr_gain=0.0,
    )

    assert control.pressure_delta_front == 0.0
    assert control.pressure_delta_rear == 0.0
    assert control.camber_delta_front == 0.0
    assert control.camber_delta_rear == 0.0
    assert control.per_wheel_pressure == {
        "fl": pytest.approx(0.1, abs=1e-6),
        "fr": pytest.approx(-0.1, abs=1e-6),
        "rl": pytest.approx(0.1, abs=1e-6),
        "rr": pytest.approx(-0.1, abs=1e-6),
    }


def test_tyre_balance_controller_returns_zero_without_cphi():
    control = tyre_balance_controller({})
    assert control.pressure_delta_front == 0.0
    assert control.pressure_delta_rear == 0.0
    assert control.camber_delta_front == 0.0
    assert control.camber_delta_rear == 0.0
    assert control.per_wheel_pressure == {"fl": 0.0, "fr": 0.0, "rl": 0.0, "rr": 0.0}


def test_tyre_balance_controller_neutral_when_cphi_missing():
    metrics = {
        "cphi_fl": float("nan"),
        "cphi_fr": None,
        "cphi_rl": float("nan"),
        "cphi_rr": None,
        "cphi_fl_temperature": None,
        "cphi_fr_temperature": float("nan"),
        "cphi_rl_temperature": None,
        "cphi_rr_temperature": float("nan"),
        "cphi_fl_gradient": None,
        "cphi_fr_gradient": float("nan"),
        "cphi_rl_gradient": None,
        "cphi_rr_gradient": float("nan"),
        "cphi_fl_mu": None,
        "cphi_fr_mu": float("nan"),
        "cphi_rl_mu": None,
        "cphi_rr_mu": float("nan"),
        "cphi_fl_temp_delta": None,
        "cphi_fr_temp_delta": float("nan"),
        "cphi_rl_temp_delta": None,
        "cphi_rr_temp_delta": float("nan"),
        "cphi_fl_gradient_rate": None,
        "cphi_fr_gradient_rate": float("nan"),
        "cphi_rl_gradient_rate": None,
        "cphi_rr_gradient_rate": float("nan"),
        "d_nfr_flat": 0.0,
    }

    control = tyre_balance_controller(metrics)
    assert control.pressure_delta_front == 0.0
    assert control.pressure_delta_rear == 0.0
    assert control.camber_delta_front == 0.0
    assert control.camber_delta_rear == 0.0
    assert control.per_wheel_pressure == {"fl": 0.0, "fr": 0.0, "rl": 0.0, "rr": 0.0}


def test_mutation_operator_detects_style_and_entropy_mutations():
    state: dict[str, dict[str, object]] = {}
    base_triggers = {
        "microsector_id": "ms-5",
        "current_archetype": "medium",
        "candidate_archetype": "hairpin",
        "fallback_archetype": "medium",
        "entropy": 0.3,
        "style_index": 0.82,
        "style_reference": 0.82,
        "phase": "entry",
    }

    initial = mutation_operator(state, base_triggers)
    assert not initial["mutated"]
    assert initial["archetype"] == "medium"

    style_shift = mutation_operator(
        state,
        {**base_triggers, "style_index": 0.55, "dynamic_conditions": True},
    )
    assert style_shift["mutated"]
    assert style_shift["archetype"] == "hairpin"

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
    assert entropy_spike["archetype"] == "medium"

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
    assert phase_adjustment["archetype"] == "hairpin"
    assert state["ms-5"]["phase"] == "apex"


def test_acoplamiento_and_resonance_behaviour():
    series_a = [0.1, 0.2, 0.3, 0.4]
    series_b = [0.1, 0.15, 0.25, 0.35]

    coupling = coupling_operator(series_a, series_b)
    resonance = resonance_operator(series_b)
    dissonance = dissonance_operator(series_a, target=0.25)

    assert coupling > 0
    expected_resonance = np.sqrt(np.mean(np.square(series_b)))
    assert resonance == pytest.approx(expected_resonance, rel=1e-9)
    assert dissonance == pytest.approx(mean(abs(value - 0.25) for value in series_a))


def _legacy_dissonance(series: Sequence[float], target: float) -> float:
    if not series:
        return 0.0
    return mean(abs(value - target) for value in series)


@pytest.mark.parametrize(
    ("series", "target"),
    [
        ([0.25, 0.5, 0.75], 0.5),
        ((-1.2, 3.4, -5.6, 7.8), 0.0),
        ([10.0], 8.0),
    ],
)
def test_dissonance_operator_matches_legacy_implementation(
    series: Sequence[float], target: float
) -> None:
    expected = _legacy_dissonance(series, target)
    result = dissonance_operator(series, target)

    assert result == pytest.approx(expected, rel=1e-12)
    assert isinstance(result, float)


def test_dissonance_operator_matches_legacy_for_empty_series() -> None:
    series: Sequence[float] = []
    expected = _legacy_dissonance(series, target=1.0)

    result = dissonance_operator(series, target=1.0)

    assert result == expected == 0.0


def test_acoplamiento_operator_alias_emits_deprecation_warning():
    series_a = [0.0, 0.5, 1.0]
    series_b = [0.0, 0.25, 0.5]

    with pytest.deprecated_call():
        legacy = acoplamiento_operator(series_a, series_b)

    expected = coupling_operator(series_a, series_b)
    assert legacy == expected


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

    expected = coupling_operator(series["tyres"], series["suspension"], strict_length=False)
    assert pairwise["tyres↔suspension"] == pytest.approx(expected, rel=1e-9)


def test_pairwise_coupling_operator_matches_manual_computation_for_multiple_pairs():
    series = {
        "tyres": [1.0, 2.0, 3.0, 4.0],
        "suspension": [2.5, 3.5, 4.5],
        "chassis": [0.5, 0.5, 0.5, 0.5],
    }
    pairs = [
        ("tyres", "suspension"),
        ("suspension", "chassis"),
        ("tyres", "chassis"),
        ("tyres", "wings"),
    ]

    pairwise = pairwise_coupling_operator(series, pairs=pairs)

    for first, second in pairs:
        expected = coupling_operator(
            series.get(first, ()), series.get(second, ()), strict_length=False
        )
        assert pairwise[f"{first}↔{second}"] == pytest.approx(expected, rel=1e-9)


def test_aggregate_operator_events_returns_latent_state_summary() -> None:
    microsector = _build_microsector(1, 0, 2, 4, apex_target=0.3)
    silence_payload = {
        "name": canonical_operator_label("SILENCE"),
        "start_index": 0,
        "end_index": 4,
        "start_time": 0.0,
        "end_time": 4.0,
        "duration": 4.0,
        "load_span": 120.0,
        "structural_density_mean": 0.04,
        "slack": 0.6,
    }
    enriched = replace(microsector, operator_events={"SILENCE": (silence_payload,)})
    aggregated = _aggregate_operator_events([enriched])
    events = aggregated.get("events", {})
    assert "SILENCE" in events
    assert events["SILENCE"][0]["name"] == canonical_operator_label("SILENCE")
    assert events["SILENCE"][0]["microsector"] == enriched.index
    latent = aggregated.get("latent_states", {})
    assert "SILENCE" in latent
    summary = latent["SILENCE"][enriched.index]
    assert pytest.approx(summary["coverage"], rel=1e-6) == 1.0
    assert pytest.approx(summary["duration"], rel=1e-6) == 4.0

