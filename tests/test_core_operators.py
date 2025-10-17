"""Unit tests for the high-level TNFR × LFS operators."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from math import cos, pi, sin, sqrt
from statistics import mean, pstdev, pvariance
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pytest

from tests.helpers import (
    build_dynamic_record,
    build_epi_bundle,
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
    resolve_context_from_bundle,
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
from tnfr_core.operators.al_operator import (
    DissonanceBreakdown,
    _zero_dissonance_breakdown,
    acoplamiento_operator,
    coherence_operator,
    coupling_operator,
    dissonance_breakdown_operator,
    dissonance_operator,
    pairwise_coupling_operator,
    resonance_operator,
)
from tnfr_core.operators.en_operator import (
    _ensure_bundle,
    _normalise_node_evolution,
    _update_bundles,
    emission_operator,
    reception_operator,
)
from tnfr_core.operators.il_operator import (
    TyreBalanceControlOutput,
    _STABILITY_COV_THRESHOLD,
    _delta_integral_series,
    _variance_payload,
    mutation_operator,
    recursive_filter_operator,
    recursividad_operator,
    tyre_balance_controller,
)
from tnfr_core.operators.entry.recursivity import recursivity_operator
from tnfr_core.operators.pipeline import orchestrate_delta_metrics
from tnfr_core.operators.structural.coherence_il import coherence_operator_il
from tnfr_core.operators.structural.epi import (
    compute_nodal_contributions,
    extract_phase_context,
    resolve_nu_targets,
)
from tnfr_core.operators.structural.epi_evolution import evolve_epi
from tnfr_core.operators._shared import _HAS_JAX, jnp
from tnfr_core.operators.pipeline.coherence import _stage_coherence as pipeline_stage_coherence
from tnfr_core.operators.pipeline.epi import _stage_epi_evolution as pipeline_stage_epi
from tnfr_core.operators.pipeline.events import (
    _aggregate_operator_events as pipeline_aggregate_operator_events,
)
from tnfr_core.operators.pipeline.nodal import (
    StructuralDeltaComponent as pipeline_structural_component,
    _stage_nodal_metrics as pipeline_stage_nodal,
)
from tnfr_core.operators.pipeline.reception import (
    _stage_reception as pipeline_stage_reception,
)
from tnfr_core.operators.pipeline.variability import (
    _microsector_cphi_values,
    _microsector_phase_synchrony_values,
    _microsector_variability as pipeline_microsector_variability,
    compute_window_metrics as pipeline_compute_window_metrics,
)
from tnfr_core.operators.pipeline.sense import _stage_sense as pipeline_stage_sense
from tnfr_core.operator_detection import canonical_operator_label
from tnfr_core.interfaces import SupportsEPIBundle, SupportsMicrosector


@dataclass
class _DummyBundle:
    timestamp: float
    delta_nfr: float


def _delta_integral_series_reference(
    bundles: Sequence[SupportsEPIBundle], sample_indices: Sequence[int]
) -> List[float]:
    integrals: List[float] = []
    if not bundles or not sample_indices:
        return integrals
    timestamps = [float(bundles[idx].timestamp) for idx in sample_indices]
    for pos, idx in enumerate(sample_indices):
        dt = 0.0
        if pos + 1 < len(sample_indices):
            dt = max(0.0, timestamps[pos + 1] - timestamps[pos])
        elif pos > 0:
            dt = max(0.0, timestamps[pos] - timestamps[pos - 1])
        if dt <= 0.0:
            dt = 1.0
        integrals.append(abs(float(bundles[idx].delta_nfr)) * dt)
    return integrals


def _variance_payload_reference(values: Sequence[float]) -> Mapping[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "variance": 0.0,
            "stdev": 0.0,
            "coefficient_of_variation": 0.0,
            "stability_score": 1.0,
        }

    average = float(mean(values))
    variance = float(pvariance(values))
    if variance < 0.0 and abs(variance) < 1e-12:
        variance = 0.0
    stdev = sqrt(variance) if variance > 0.0 else 0.0
    baseline = max(abs(average), 1e-9)
    coefficient = stdev / baseline
    stability = 1.0 - min(1.0, coefficient / _STABILITY_COV_THRESHOLD)
    if stability < 0.0:
        stability = 0.0
    return {
        "mean": average,
        "variance": variance,
        "stdev": stdev,
        "coefficient_of_variation": coefficient,
        "stability_score": stability,
    }


def _run_pipeline_stage_epi(
    records: Sequence[TelemetryRecord],
    **kwargs: object,
) -> Mapping[str, object]:
    bundles = kwargs.pop("bundles", None)
    phase_assignments = kwargs.pop("phase_assignments", None)
    phase_weight_lookup = kwargs.pop("phase_weight_lookup", None)
    global_phase_weights = kwargs.pop("global_phase_weights", None)
    return pipeline_stage_epi(
        records,
        bundles=bundles,
        phase_assignments=phase_assignments,
        phase_weight_lookup=phase_weight_lookup,
        global_phase_weights=global_phase_weights,
        ensure_bundle=_ensure_bundle,
        normalise_node_evolution=_normalise_node_evolution,
        **kwargs,
    )


def _run_pipeline_stage_coherence(
    bundles: Sequence[SupportsEPIBundle],
    objectives: Mapping[str, float],
    *,
    coherence_window: int,
    microsectors: Sequence[SupportsMicrosector] | None = None,
) -> Mapping[str, object]:
    return pipeline_stage_coherence(
        bundles,
        objectives,
        coherence_window=coherence_window,
        microsectors=microsectors,
        load_context_matrix=load_context_matrix,
        resolve_context_from_bundle=resolve_context_from_bundle,
        apply_contextual_delta=apply_contextual_delta,
        update_bundles=_update_bundles,
        coherence_operator=coherence_operator,
        dissonance_operator=dissonance_breakdown_operator,
        coupling_operator=coupling_operator,
        resonance_operator=resonance_operator,
        empty_breakdown_factory=_zero_dissonance_breakdown,
    )


def _run_pipeline_stage_sense(
    series: Sequence[float], *, recursion_decay: float
) -> Mapping[str, object]:
    return pipeline_stage_sense(
        series,
        recursion_decay=recursion_decay,
        recursive_filter=lambda values, seed, decay: recursive_filter_operator(
            values, seed=seed, decay=decay
        ),
    )


def _run_pipeline_stage_variability(
    microsectors: Sequence[SupportsMicrosector] | None,
    bundles: Sequence[SupportsEPIBundle],
    lap_indices: Sequence[int],
    lap_metadata: Sequence[Mapping[str, object]],
) -> Sequence[Mapping[str, object]]:
    xp_module = jnp if _HAS_JAX and jnp is not None else np
    return pipeline_microsector_variability(
        microsectors,
        bundles,
        lap_indices,
        lap_metadata,
        xp=xp_module,
        has_jax=_HAS_JAX,
        delta_integral=_delta_integral_series,
        variance_payload=_variance_payload,
    )


def test_stage_reception_flattens_segments_and_builds_lap_metadata() -> None:
    first = build_dynamic_record(
        0.0,
        5000.0,
        0.02,
        0.8,
        0.1,
        100.0,
        0.75,
    )
    first = replace(first, lap=12)
    second = replace(
        first,
        timestamp=0.1,
        lateral_accel=0.82,
        longitudinal_accel=0.12,
        nfr=101.0,
        si=0.76,
    )
    second = replace(second, lap=12)
    third = replace(
        first,
        timestamp=0.25,
        lap=None,
        lateral_accel=0.6,
        longitudinal_accel=0.05,
        nfr=98.0,
        si=0.73,
    )

    segments = ([first, second], [third])

    def fake_reception(segment: Sequence[TelemetryRecord]):
        bundles: list[EPIBundle] = []
        for sample in segment:
            fake_reception.counter += 1
            magnitude = 0.1 * fake_reception.counter
            bundles.append(
                build_operator_bundle(
                    timestamp=sample.timestamp,
                    tyre_delta=magnitude,
                    delta_nfr=magnitude,
                    sense_index=0.65 + 0.01 * fake_reception.counter,
                )
            )
        return bundles

    fake_reception.counter = 0  # type: ignore[attr-defined]

    stage_payload, flattened_records = pipeline_stage_reception(
        segments,
        reception_fn=fake_reception,
    )

    assert stage_payload["sample_count"] == 3
    assert [record.timestamp for record in flattened_records] == [
        record.timestamp for segment in segments for record in segment
    ]
    assert len(stage_payload["bundles"]) == stage_payload["sample_count"]
    assert stage_payload["lap_indices"] == [0, 0, 1]
    lap_sequence = stage_payload["lap_sequence"]
    assert lap_sequence[0]["label"] == "12"
    assert lap_sequence[0]["explicit"] is True
    assert lap_sequence[1]["label"] == "Lap 2"
    assert lap_sequence[1]["explicit"] is False


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([0.1, -0.2, 0.3, -0.4, 0.5], id="small_series"),
        pytest.param(
            [float(x) for x in np.linspace(-5.0, 5.0, 1024)],
            id="large_series",
        ),
    ],
)
def test_variance_payload_vectorized_matches_reference(values: Sequence[float]):
    reference = _variance_payload_reference(values)
    result = _variance_payload(values)

    assert set(result.keys()) == set(reference.keys())
    for key, expected in reference.items():
        assert isinstance(result[key], float)
        assert result[key] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_delta_integral_series_vectorized_matches_reference():
    regular_bundles = [
        _DummyBundle(timestamp=t, delta_nfr=d)
        for t, d in zip((0.0, 1.0, 2.0, 3.0), (0.5, -1.5, 2.0, -0.1))
    ]

    gapped_bundles = [
        _DummyBundle(timestamp=t, delta_nfr=d)
        for t, d in zip((0.0, 0.4, 1.8, 3.6, 7.2), (1.0, -0.2, 0.8, -1.2, 0.4))
    ]

    repeated_bundles = [
        _DummyBundle(timestamp=t, delta_nfr=d)
        for t, d in zip((0.0, 1.0, 1.0, 4.0), (1.5, -0.5, 0.5, 2.5))
    ]

    scenarios = [
        (regular_bundles, list(range(len(regular_bundles)))),
        (gapped_bundles, [0, 2, 4]),
        (repeated_bundles, list(range(len(repeated_bundles)))),
    ]

    for bundles, indices in scenarios:
        expected = _delta_integral_series_reference(bundles, indices)
        result = _delta_integral_series(bundles, indices)
        assert result == pytest.approx(expected)


def test_coherence_operator_il_preserves_mean():
    series = [0.2, -0.1, 0.5, 1.2, -0.4, 0.3]

    smoothed = coherence_operator_il(series, window=5)

    assert len(smoothed) == len(series)
    assert mean(smoothed) == pytest.approx(mean(series), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "window",
    [
        pytest.param(-1, id="negative"),
        pytest.param(0, id="zero"),
        pytest.param(4, id="even"),
    ],
)
def test_coherence_operator_il_rejects_invalid_windows(window: int):
    with pytest.raises(ValueError):
        coherence_operator_il([1.0, 2.0, 3.0], window=window)


def test_coherence_operator_il_handles_edge_cases():
    assert coherence_operator_il([], window=5) == []

    short_series = [1.0, 2.0, 3.0]
    smoothed_short = coherence_operator_il(short_series, window=7)

    assert len(smoothed_short) == len(short_series)
    assert mean(smoothed_short) == pytest.approx(mean(short_series), rel=1e-12, abs=1e-12)


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

    expected = _run_pipeline_stage_epi(records)
    protocol_records = clone_protocol_series(records)
    result = _run_pipeline_stage_epi(protocol_records)

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


def _build_epi_stage_records() -> List[TelemetryRecord]:
    base = build_dynamic_record(
        timestamp=0.0,
        vertical_load=5050.0,
        slip_ratio=0.02,
        lateral_accel=0.7,
        longitudinal_accel=0.3,
        nfr=5.2,
        si=0.8,
        throttle=0.42,
        steer=0.12,
        yaw_rate=0.15,
    )
    first = replace(base, reference=base)
    second = replace(
        first,
        timestamp=0.1,
        slip_ratio=0.05,
        lateral_accel=0.9,
        longitudinal_accel=0.4,
        nfr=5.7,
        si=0.82,
        throttle=0.47,
        steer=0.16,
        yaw_rate=0.22,
        reference=first,
    )
    third = replace(
        second,
        timestamp=0.2,
        slip_ratio=0.07,
        lateral_accel=1.05,
        longitudinal_accel=0.55,
        nfr=6.0,
        si=0.85,
        throttle=0.52,
        steer=0.2,
        yaw_rate=0.28,
        reference=second,
    )
    return [first, second, third]


def _build_epi_stage_bundles(
    timestamps: Sequence[float],
    integrated: Sequence[float],
    derivatives: Sequence[float],
    node_series: Sequence[Mapping[str, tuple[float, float]]],
) -> List[EPIBundle]:
    bundles: List[EPIBundle] = []
    sense_index = 0.85
    for idx, timestamp in enumerate(timestamps):
        node_evolution = dict(node_series[idx])
        node_derivative_sum = sum(values[1] for values in node_evolution.values())
        tyres = TyresNode(delta_nfr=node_derivative_sum, sense_index=sense_index)
        suspension = SuspensionNode(
            delta_nfr=node_derivative_sum * 0.5,
            sense_index=sense_index,
        )
        chassis = ChassisNode(
            delta_nfr=node_derivative_sum * 0.25,
            sense_index=sense_index,
        )
        bundles.append(
            EPIBundle(
                timestamp=timestamp,
                epi=integrated[idx],
                delta_nfr=node_derivative_sum,
                sense_index=sense_index,
                tyres=tyres,
                suspension=suspension,
                chassis=chassis,
                brakes=BrakesNode(delta_nfr=0.0, sense_index=sense_index),
                transmission=TransmissionNode(delta_nfr=0.0, sense_index=sense_index),
                track=TrackNode(delta_nfr=0.0, sense_index=sense_index),
                driver=DriverNode(delta_nfr=0.0, sense_index=sense_index),
                dEPI_dt=derivatives[idx],
                integrated_epi=integrated[idx],
                node_evolution=node_evolution,
            )
        )
    return bundles


def test_stage_epi_evolution_reuses_bundle_history_when_no_phase_overrides(monkeypatch):
    records = _build_epi_stage_records()
    node_evolution_series = [
        {"tyres": (0.05, 0.5), "suspension": (0.02, 0.2), "chassis": (0.01, 0.1)},
        {"tyres": (0.04, 0.4), "suspension": (0.03, 0.3), "chassis": (0.015, 0.15)},
        {"tyres": (0.06, 0.6), "suspension": (0.025, 0.25), "chassis": (0.02, 0.2)},
    ]
    integrated_series = [0.08, 0.165, 0.27]
    derivative_series = [0.8, 0.85, 1.05]
    timestamps = [record.timestamp for record in records]
    bundles = _build_epi_stage_bundles(
        timestamps, integrated_series, derivative_series, node_evolution_series
    )

    def _fail(*_args: object, **_kwargs: object) -> Mapping[str, float]:
        raise AssertionError("EPI evolution should reuse existing bundle data")

    monkeypatch.setattr("tnfr_core.epi.delta_nfr_by_node", _fail)
    monkeypatch.setattr("tnfr_core.epi.resolve_nu_f_by_node", _fail)

    result = _run_pipeline_stage_epi(records, bundles=bundles)

    assert result["integrated"] == pytest.approx(integrated_series)
    assert result["derivative"] == pytest.approx(derivative_series)

    nodes = {name for series in node_evolution_series for name in series}
    expected_node_integrated: Dict[str, List[float]] = {name: [] for name in nodes}
    expected_node_derivative: Dict[str, List[float]] = {name: [] for name in nodes}
    cumulative: Dict[str, float] = {name: 0.0 for name in nodes}
    for series in node_evolution_series:
        for name in nodes:
            integral, derivative = series.get(name, (0.0, 0.0))
            cumulative[name] += integral
            expected_node_integrated[name].append(cumulative[name])
            expected_node_derivative[name].append(derivative)

    assert set(result["per_node_integrated"]) == nodes
    assert set(result["per_node_derivative"]) == nodes

    for name in nodes:
        assert result["per_node_integrated"][name] == pytest.approx(
            expected_node_integrated[name]
        )
        assert result["per_node_derivative"][name] == pytest.approx(
            expected_node_derivative[name]
        )


def test_stage_epi_evolution_recomputes_when_phase_customisation_requested(monkeypatch):
    records = _build_epi_stage_records()
    expected = _run_pipeline_stage_epi(records, phase_assignments={1: "apex"})

    node_evolution_series = [
        {"tyres": (0.05, 0.5), "suspension": (0.02, 0.2), "chassis": (0.01, 0.1)},
        {"tyres": (0.04, 0.4), "suspension": (0.03, 0.3), "chassis": (0.015, 0.15)},
        {"tyres": (0.06, 0.6), "suspension": (0.025, 0.25), "chassis": (0.02, 0.2)},
    ]
    integrated_series = [0.08, 0.165, 0.27]
    derivative_series = [0.8, 0.85, 1.05]
    timestamps = [record.timestamp for record in records]
    bundles = _build_epi_stage_bundles(
        timestamps, integrated_series, derivative_series, node_evolution_series
    )

    delta_calls = 0
    original_delta = delta_nfr_by_node

    def _spy_delta(record: TelemetryRecord) -> Mapping[str, float]:
        nonlocal delta_calls
        delta_calls += 1
        return original_delta(record)

    nu_calls = 0
    original_nu = resolve_nu_f_by_node

    def _spy_nu(*args: object, **kwargs: object):
        nonlocal nu_calls
        nu_calls += 1
        return original_nu(*args, **kwargs)

    monkeypatch.setattr("tnfr_core.epi.delta_nfr_by_node", _spy_delta)
    monkeypatch.setattr("tnfr_core.epi.resolve_nu_f_by_node", _spy_nu)
    monkeypatch.setattr(
        "tnfr_core.operators.pipeline.epi.delta_nfr_by_node",
        _spy_delta,
    )
    monkeypatch.setattr(
        "tnfr_core.operators.pipeline.epi.resolve_nu_f_by_node",
        _spy_nu,
    )

    result = _run_pipeline_stage_epi(
        records,
        bundles=bundles,
        phase_assignments={1: "apex"},
    )

    assert result == expected
    assert delta_calls == len(records)
    assert nu_calls == len(records)


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

    stage = _run_pipeline_stage_coherence(bundles, objectives, coherence_window=1)

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


def test_stage_nodal_metrics_scales_structural_delta_with_context() -> None:
    bundles = [
        build_epi_bundle(
            timestamp=0.0,
            delta_nfr=0.42,
            sense_index=0.82,
            tyres={"delta_nfr": 0.18, "sense_index": 0.81},
            suspension={"delta_nfr": 0.16, "sense_index": 0.79},
            chassis={"delta_nfr": 0.08, "sense_index": 0.76},
        ),
        build_epi_bundle(
            timestamp=0.2,
            delta_nfr=0.36,
            sense_index=0.8,
            tyres={"delta_nfr": 0.14, "sense_index": 0.78},
            suspension={"delta_nfr": 0.12, "sense_index": 0.77},
            chassis={"delta_nfr": 0.1, "sense_index": 0.75},
        ),
    ]

    structural = pipeline_structural_component()

    def _pairwise(series_by_node: Mapping[str, Sequence[float]], pairs: Sequence[tuple[str, str]]):
        return {
            f"{first}↔{second}": coupling_operator(
                series_by_node.get(first, ()),
                series_by_node.get(second, ()),
                strict_length=False,
            )
            for first, second in pairs
        }

    stage = pipeline_stage_nodal(
        bundles,
        load_context_matrix=load_context_matrix,
        xp=np,
        structural_component=structural,
        pairwise_coupling=_pairwise,
    )

    matrix = load_context_matrix()
    contexts = resolve_series_context(bundles, matrix=matrix)
    multipliers = [
        max(
            matrix.min_multiplier,
            min(matrix.max_multiplier, float(contexts[idx].multiplier)),
        )
        for idx in range(len(bundles))
    ]

    expected_structural = structural.component_series(bundles)
    assert stage["structural_delta"] == expected_structural

    for node_name, series in stage["delta_by_node"].items():
        node_values = [getattr(bundle, node_name).delta_nfr for bundle in bundles]
        expected = [value * multipliers[idx] for idx, value in enumerate(node_values)]
        assert series == pytest.approx(expected, rel=1e-12, abs=1e-12)

    for node_name, series in stage["sense_index_by_node"].items():
        expected = [getattr(bundle, node_name).sense_index for bundle in bundles]
        assert series == pytest.approx(expected, rel=1e-12, abs=1e-12)

    derivatives = stage["dEPI_dt_by_operator"]
    for component, values in expected_structural.items():
        expected = [value * multipliers[idx] for idx, value in enumerate(values)]
        assert derivatives[component] == pytest.approx(expected, rel=1e-12, abs=1e-12)

    pairwise_delta = stage["pairwise_coupling"]["delta_nfr"]
    pairwise_si = stage["pairwise_coupling"]["sense_index"]
    assert pairwise_delta["tyres↔suspension"] == pytest.approx(
        coupling_operator(
            stage["delta_by_node"]["tyres"],
            stage["delta_by_node"]["suspension"],
            strict_length=False,
        )
    )
    assert pairwise_si["tyres↔suspension"] == pytest.approx(
        coupling_operator(
            stage["sense_index_by_node"]["tyres"],
            stage["sense_index_by_node"]["suspension"],
            strict_length=False,
        )
    )


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
    phase_context = extract_phase_context(deltas)
    nu_targets = resolve_nu_targets(deltas)
    expected_contrib, theta_effects, expected_derivative = compute_nodal_contributions(
        deltas, nu_map, nu_targets, phase_context, 0.1
    )

    new_epi, derivative, nodal = evolve_epi(prev, deltas, 0.1, nu_map)

    assert derivative == pytest.approx(expected_derivative, rel=1e-9)
    assert new_epi == pytest.approx(prev + expected_derivative * 0.1, rel=1e-9)
    assert set(expected_contrib) == set(nodal)
    for node, expected in expected_contrib.items():
        assert nodal[node][0] == pytest.approx(expected[0], rel=1e-9)
        assert nodal[node][1] == pytest.approx(expected[1], rel=1e-9)
    assert sum(component[1] for component in nodal.values()) == pytest.approx(
        derivative, rel=1e-9
    )
    assert theta_effects == {}
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
    phase_context = extract_phase_context(deltas)
    nu_targets = resolve_nu_targets(deltas)
    expected_contrib, theta_effects, expected_derivative = compute_nodal_contributions(
        deltas, nu_map, nu_targets, phase_context, 0.1
    )

    new_epi, derivative, nodal = evolve_epi(prev, deltas, 0.1, nu_map)

    assert derivative == pytest.approx(expected_derivative, rel=1e-9)
    assert new_epi == pytest.approx(expected_derivative * 0.1, rel=1e-9)
    for node, expected in expected_contrib.items():
        assert nodal[node][1] == pytest.approx(expected[1], rel=1e-9)

    metadata = getattr(nodal, "metadata", {})
    assert metadata.get("theta") == phase_context.identifier
    assert metadata.get("theta_effect") == theta_effects
    assert metadata.get("nu_f_objectives") == nu_targets
    if phase_context.weights is not None:
        assert metadata.get("w_phase") == dict(phase_context.weights)


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

    metrics = pipeline_compute_window_metrics(
        records, phase_indices=range(len(records))
    )

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

    assignments = {index: "entry" for index in range(len(records))}
    base_expected = _run_pipeline_stage_epi(
        records,
        phase_assignments=assignments,
        phase_weight_lookup={index: base_weights for index in assignments},
        global_phase_weights=base_weights,
    )
    boosted_expected = _run_pipeline_stage_epi(
        records,
        phase_assignments=assignments,
        phase_weight_lookup={index: boosted_weights for index in assignments},
        global_phase_weights=boosted_weights,
    )

    base_series = base_metrics["epi_evolution"]["per_node_derivative"]["tyres"]
    boosted_series = boosted_metrics["epi_evolution"]["per_node_derivative"]["tyres"]
    assert base_series == pytest.approx(base_expected["per_node_derivative"]["tyres"])
    assert boosted_series == pytest.approx(
        boosted_expected["per_node_derivative"]["tyres"]
    )
    assert (
        boosted_expected["per_node_derivative"]["tyres"][-1]
        > base_expected["per_node_derivative"]["tyres"][-1]
    )


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
        "tnfr_core.operators.en_operator.reception_operator",
        _fake_reception,
    )
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
    expected_phase_sync_values = _microsector_phase_synchrony_values(microsectors[0])
    assert first["overall"]["phase_synchrony"]["mean"] == pytest.approx(
        mean(expected_phase_sync_values)
    )
    expected_cphi_values = _microsector_cphi_values(microsectors[0])
    expected_cphi_mean = mean(expected_cphi_values) if expected_cphi_values else 0.0
    assert first["overall"]["cphi"]["mean"] == pytest.approx(expected_cphi_mean)
    assert set(first["laps"]) == {"Lap 1", "Lap 2"}
    assert first["laps"]["Lap 1"]["samples"] == 2
    lap_metrics = first["laps"]["Lap 1"]
    assert "delta_nfr_integral" in lap_metrics
    assert "cphi" not in lap_metrics
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


def test_pipeline_stage_variability_returns_empty_for_missing_microsectors() -> None:
    result = _run_pipeline_stage_variability(None, (), (), ())

    assert result == []


def test_pipeline_stage_variability_computes_lap_breakdown_with_samples() -> None:
    bundles = [
        build_operator_bundle(timestamp=0.0, tyre_delta=0.18, delta_nfr=0.2, sense_index=0.74),
        build_operator_bundle(timestamp=0.12, tyre_delta=0.22, delta_nfr=0.28, sense_index=0.76),
        build_operator_bundle(timestamp=0.3, tyre_delta=-0.1, delta_nfr=-0.12, sense_index=0.72),
    ]
    microsector = build_microsector(
        index=0,
        phases=("entry", "apex", "exit"),
        phase_samples={"entry": (0,), "apex": (1,), "exit": (2,)},
        filtered_measures={
            "front": {"coherence_phi": 0.66},
            "rear": {"coherence_phi": 0.72},
        },
        phase_synchrony={"entry": 0.91, "apex": 0.87, "exit": 0.89},
    )
    lap_indices = [0, 0, 1]
    lap_metadata = [
        {"index": 0, "label": "Lap 1"},
        {"index": 1, "label": "Lap 2"},
    ]

    results = _run_pipeline_stage_variability(
        [microsector],
        bundles,
        lap_indices,
        lap_metadata,
    )

    assert len(results) == 1
    entry = results[0]
    assert entry["microsector"] == microsector.index
    assert entry["overall"]["samples"] == len(bundles)

    delta_series = [bundle.delta_nfr for bundle in bundles]
    sense_series = [bundle.sense_index for bundle in bundles]
    integral_series = _delta_integral_series(bundles, list(range(len(bundles))))
    cphi_values = _microsector_cphi_values(microsector)
    synchrony_values = _microsector_phase_synchrony_values(microsector)

    delta_stats = _variance_payload_reference(delta_series)
    sense_stats = _variance_payload_reference(sense_series)
    integral_stats = _variance_payload_reference(integral_series)
    cphi_stats = _variance_payload_reference(cphi_values)
    synchrony_stats = _variance_payload_reference(synchrony_values)

    assert entry["overall"]["delta_nfr"] == {
        key: pytest.approx(value, rel=1e-12, abs=1e-12)
        for key, value in delta_stats.items()
    }
    assert entry["overall"]["sense_index"] == {
        key: pytest.approx(value, rel=1e-12, abs=1e-12)
        for key, value in sense_stats.items()
    }
    assert entry["overall"]["delta_nfr_integral"] == {
        key: pytest.approx(value, rel=1e-12, abs=1e-12)
        for key, value in integral_stats.items()
    }
    assert entry["overall"]["cphi"] == {
        key: pytest.approx(value, rel=1e-12, abs=1e-12)
        for key, value in cphi_stats.items()
    }
    assert entry["overall"]["phase_synchrony"] == {
        key: pytest.approx(value, rel=1e-12, abs=1e-12)
        for key, value in synchrony_stats.items()
    }

    assert "Lap 1" in entry.get("laps", {})
    lap_one = entry["laps"]["Lap 1"]
    lap_two = entry["laps"]["Lap 2"]
    assert lap_one["samples"] == 2
    assert lap_two["samples"] == 1

    lap_one_delta = _variance_payload_reference(delta_series[:2])
    lap_two_delta = _variance_payload_reference(delta_series[2:])
    lap_one_integral = _variance_payload_reference(
        _delta_integral_series(bundles, [0, 1])
    )
    lap_two_integral = _variance_payload_reference(
        _delta_integral_series(bundles, [2])
    )

    assert lap_one["delta_nfr"] == {
        key: pytest.approx(value, rel=1e-12, abs=1e-12)
        for key, value in lap_one_delta.items()
    }
    assert lap_two["delta_nfr"] == {
        key: pytest.approx(value, rel=1e-12, abs=1e-12)
        for key, value in lap_two_delta.items()
    }
    assert lap_one["delta_nfr_integral"] == {
        key: pytest.approx(value, rel=1e-12, abs=1e-12)
        for key, value in lap_one_integral.items()
    }
    assert lap_two["delta_nfr_integral"] == {
        key: pytest.approx(value, rel=1e-12, abs=1e-12)
        for key, value in lap_two_integral.items()
    }


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


def _reference_recursive_filter(
    series: Sequence[float], *, seed: float = 0.0, decay: float = 0.5
) -> list[float]:
    state = seed
    trace: list[float] = []
    for value in series:
        state = (decay * state) + ((1.0 - decay) * value)
        trace.append(state)
    return trace


@pytest.mark.parametrize("seed", [0.0, -0.25, 1.5])
@pytest.mark.parametrize("decay", [0.0, 0.2, 0.5, 0.85])
@pytest.mark.parametrize(
    "series",
    [
        [],
        [0.0, 0.5, 1.0],
        [0.3, -0.6, 0.4, 0.2, -0.1],
    ],
)
def test_recursive_filter_operator_matches_reference(
    series: Sequence[float], decay: float, seed: float
):
    expected = _reference_recursive_filter(series, seed=seed, decay=decay)
    result = recursive_filter_operator(series, seed=seed, decay=decay)
    assert len(result) == len(expected)
    for actual, expected_value in zip(result, expected):
        assert actual == pytest.approx(expected_value)


def test_recursividad_operator_alias_emits_deprecation_warning():
    series = [0.0, 0.5, 1.0]

    with pytest.deprecated_call():
        legacy = recursividad_operator(series, decay=0.3)

    expected = recursive_filter_operator(series, decay=0.3)
    assert legacy == expected


def test_pipeline_stage_sense_matches_recursive_filter_trace():
    series = [0.2, 0.4, 0.1, -0.3]
    decay = 0.25
    expected_memory = recursive_filter_operator(series, seed=series[0], decay=decay)

    stage = _run_pipeline_stage_sense(series, recursion_decay=decay)

    assert stage["series"] == series
    assert stage["memory"] == pytest.approx(expected_memory)
    assert stage["average"] == pytest.approx(mean(series))
    assert stage["decay"] == decay


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


def test_recursivity_operator_trace_respects_history_limit():
    state: dict[str, dict[str, object]] = {}
    session = ("FZR", "aston", "soft")

    for step in range(5):
        recursivity_operator(
            state,
            session,
            "ms-1",
            {
                "thermal_load": float(step),
                "style_index": 0.6,
                "phase": "entry",
                "timestamp": float(step),
            },
            decay=0.0,
            history=3,
            convergence_window=10,
        )

    session_key = "|".join(session)
    active_state = state["sessions"][session_key]["active"]
    trace = active_state["ms-1"]["trace"]

    assert isinstance(trace, deque)
    assert list(entry["thermal_load"] for entry in trace) == [2.0, 3.0, 4.0]


def test_recursivity_operator_upgrades_legacy_trace_list():
    state: dict[str, dict[str, object]] = {}
    session = ("FZR", "aston", "soft")

    recursivity_operator(
        state,
        session,
        "ms-1",
        {
            "thermal_load": 1.0,
            "style_index": 0.6,
            "phase": "entry",
            "timestamp": 0.0,
        },
        decay=0.0,
        history=4,
    )

    session_key = "|".join(session)
    micro_state = state["sessions"][session_key]["active"]["ms-1"]
    micro_state["trace"] = [
        {"phase": "entry", "thermal_load": 1.0},
        {"phase": "mid", "thermal_load": 2.0},
    ]

    result = recursivity_operator(
        state,
        session,
        "ms-1",
        {
            "thermal_load": 3.0,
            "style_index": 0.7,
            "phase": "exit",
            "timestamp": 1.0,
        },
        decay=0.0,
        history=2,
    )

    trace = micro_state["trace"]

    assert isinstance(trace, deque)
    assert len(trace) == 2
    assert [entry["phase"] for entry in trace] == ["mid", "exit"]
    assert tuple(entry["phase"] for entry in result["trace"]) == ("mid", "exit")


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
    aggregated = pipeline_aggregate_operator_events([enriched])
    events = aggregated.get("events", {})
    assert "SILENCE" in events
    assert events["SILENCE"][0]["name"] == canonical_operator_label("SILENCE")
    assert events["SILENCE"][0]["microsector"] == enriched.index
    latent = aggregated.get("latent_states", {})
    assert "SILENCE" in latent
    summary = latent["SILENCE"][enriched.index]
    assert pytest.approx(summary["coverage"], rel=1e-6) == 1.0
    assert pytest.approx(summary["duration"], rel=1e-6) == 4.0

