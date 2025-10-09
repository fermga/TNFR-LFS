from __future__ import annotations

import math
import pytest
from typing import Mapping

from tnfr_lfs.core.metrics import (
    AeroBalanceDrift,
    AeroCoherence,
    BrakeHeadroom,
    BumpstopHistogram,
    CPHIReport,
    CPHIThresholds,
    LockingWindowScore,
    SlideCatchBudget,
    SuspensionVelocityBands,
    WindowMetrics,
    compute_aero_coherence,
    compute_window_metrics,
    resolve_aero_mechanical_coherence,
)
from tnfr_lfs.core.spectrum import motor_input_correlations, phase_to_latency_ms
from dataclasses import replace
from statistics import pvariance, pstdev
from types import SimpleNamespace
from tnfr_lfs.core.contextual_delta import (
    apply_contextual_delta,
    load_context_matrix,
    resolve_context_from_bundle,
    resolve_context_from_record,
)
from tnfr_lfs.core.epi import DeltaCalculator, TelemetryRecord, _ackermann_parallel_delta
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
from tnfr_lfs.core.utils import normalised_entropy


def _record(timestamp: float, nfr: float, si: float = 0.8, **overrides) -> TelemetryRecord:
    base = TelemetryRecord(
        timestamp=timestamp,
        vertical_load=5000.0,
        slip_ratio=0.0,
        lateral_accel=0.0,
        longitudinal_accel=0.0,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        brake_pressure=0.0,
        locking=0.0,
        nfr=nfr,
        si=si,
        speed=0.0,
        yaw_rate=0.0,
        slip_angle=0.0,
        steer=0.0,
        throttle=0.0,
        gear=3,
        vertical_load_front=2500.0,
        vertical_load_rear=2500.0,
        mu_eff_front=1.0,
        mu_eff_rear=1.0,
        mu_eff_front_lateral=1.0,
        mu_eff_front_longitudinal=0.95,
        mu_eff_rear_lateral=1.0,
        mu_eff_rear_longitudinal=0.95,
        suspension_travel_front=0.0,
        suspension_travel_rear=0.0,
        suspension_velocity_front=0.0,
        suspension_velocity_rear=0.0,
    )
    if overrides:
        base = replace(base, **overrides)
    return base


def test_record_optional_defaults_are_nan() -> None:
    record = _record(0.0, 100.0)

    assert math.isnan(record.slip_ratio_fl)
    assert math.isnan(record.slip_angle_fl)
    assert math.isnan(record.tyre_temp_fl)
    assert math.isnan(record.tyre_pressure_fl)
    assert math.isnan(record.brake_temp_fl)
    assert math.isnan(record.rpm)
    assert math.isnan(record.line_deviation)
    assert math.isnan(record.instantaneous_radius)
    assert math.isnan(record.front_track_width)
    assert math.isnan(record.wheelbase)


def test_phase_to_latency_ms_converts_expected_delay() -> None:
    frequency = 2.5
    phase = math.pi / 4
    expected = phase / (2.0 * math.pi * frequency) * 1000.0
    assert phase_to_latency_ms(frequency, phase) == pytest.approx(expected)


def test_motor_input_correlations_prefers_steer_yaw_pair() -> None:
    frequency = 1.6
    phase_offset = math.pi / 5
    records = []
    for index in range(96):
        timestamp = index * 0.04
        steer = math.sin(2.0 * math.pi * frequency * timestamp)
        yaw = math.sin(2.0 * math.pi * frequency * timestamp - phase_offset)
        records.append(
            _record(
                timestamp,
                100.0,
                yaw_rate=yaw,
                steer=steer,
                throttle=0.0,
                brake_pressure=0.0,
                longitudinal_accel=0.0,
                lateral_accel=0.0,
            )
        )

    correlations = motor_input_correlations(records)

    assert ("steer", "yaw") in correlations
    dominant = correlations[("steer", "yaw")]
    expected_latency = phase_to_latency_ms(frequency, phase_offset)
    assert dominant.frequency == pytest.approx(frequency, rel=0.1)
    assert dominant.latency_ms == pytest.approx(expected_latency, abs=5.0)


def _steering_bundle(record: TelemetryRecord, ackermann_delta: float) -> EPIBundle:
    share = record.nfr / 7.0
    return EPIBundle(
        timestamp=record.timestamp,
        epi=0.0,
        delta_nfr=record.nfr,
        delta_nfr_proj_longitudinal=0.0,
        delta_nfr_proj_lateral=0.0,
        sense_index=record.si,
        tyres=TyresNode(delta_nfr=share, sense_index=record.si),
        suspension=SuspensionNode(delta_nfr=share, sense_index=record.si),
        chassis=ChassisNode(
            delta_nfr=share,
            sense_index=record.si,
            yaw=record.yaw,
            pitch=record.pitch,
            roll=record.roll,
            yaw_rate=record.yaw_rate,
            lateral_accel=record.lateral_accel,
            longitudinal_accel=record.longitudinal_accel,
        ),
        brakes=BrakesNode(delta_nfr=share, sense_index=record.si),
        transmission=TransmissionNode(
            delta_nfr=share,
            sense_index=record.si,
            throttle=record.throttle,
            gear=record.gear,
            speed=record.speed,
            longitudinal_accel=record.longitudinal_accel,
            rpm=record.rpm,
            line_deviation=record.line_deviation,
        ),
        track=TrackNode(
            delta_nfr=share,
            sense_index=record.si,
            axle_load_balance=0.0,
            axle_velocity_balance=0.0,
            yaw=record.yaw,
            lateral_accel=record.lateral_accel,
        ),
        driver=DriverNode(
            delta_nfr=share,
            sense_index=record.si,
            steer=record.steer,
            throttle=record.throttle,
            style_index=record.si,
        ),
        ackermann_parallel_index=ackermann_delta,
    )


def test_ackermann_parallel_delta_uses_wheel_slip_angles() -> None:
    baseline = _record(
        0.0,
        100.0,
        yaw_rate=0.6,
        slip_angle=0.02,
        slip_angle_fl=0.06,
        slip_angle_fr=0.01,
    )
    sample = replace(
        baseline,
        yaw_rate=0.8,
        slip_angle_fl=0.10,
        slip_angle_fr=-0.02,
    )
    delta = _ackermann_parallel_delta(sample, baseline)
    assert delta == pytest.approx(0.07, rel=1e-6)


def test_ackermann_parallel_delta_swaps_wheels_on_right_turn() -> None:
    baseline = _record(
        0.0,
        95.0,
        yaw_rate=-0.5,
        slip_angle=0.015,
        slip_angle_fl=0.02,
        slip_angle_fr=0.08,
    )
    sample = replace(
        baseline,
        yaw_rate=-0.7,
        slip_angle_fl=0.01,
        slip_angle_fr=0.12,
    )
    delta = _ackermann_parallel_delta(sample, baseline)
    assert delta == pytest.approx(0.05, rel=1e-6)


def test_ackermann_parallel_delta_ignores_low_yaw_rate() -> None:
    baseline = _record(0.0, 92.0, yaw_rate=0.0, slip_angle_fl=0.03, slip_angle_fr=0.0)
    sample = replace(baseline, yaw_rate=1e-7, slip_angle_fl=0.2, slip_angle_fr=-0.1)
    delta = _ackermann_parallel_delta(sample, baseline)
    assert delta == pytest.approx(0.0)


def test_compute_window_metrics_tracks_ackermann_overshoot() -> None:
    records = [
        _record(
            0.0,
            100.0,
            yaw_rate=0.52,
            steer=0.18,
            slip_angle=0.04,
            slip_angle_fl=0.09,
            slip_angle_fr=0.01,
        ),
        _record(
            0.5,
            101.0,
            yaw_rate=0.55,
            steer=0.24,
            slip_angle=0.035,
            slip_angle_fl=0.07,
            slip_angle_fr=0.02,
        ),
        _record(
            1.0,
            99.0,
            yaw_rate=0.58,
            steer=0.3,
            slip_angle=0.03,
            slip_angle_fl=0.045,
            slip_angle_fr=0.025,
        ),
    ]
    baseline = DeltaCalculator.derive_baseline(records)
    ackermann_values = [
        _ackermann_parallel_delta(record, baseline) for record in records
    ]
    bundles = [
        _steering_bundle(record, ackermann)
        for record, ackermann in zip(records, ackermann_values)
    ]
    metrics = compute_window_metrics(records, bundles=bundles)
    expected_overshoot = sum(abs(value) for value in ackermann_values) / len(
        ackermann_values
    )
    normalised = min(1.0, expected_overshoot / math.radians(5.0))
    assert metrics.slide_catch_budget.overshoot_ratio == pytest.approx(
        normalised,
        rel=1e-6,
    )
    assert metrics.ackermann_parallel_index == pytest.approx(
        sum(ackermann_values) / len(ackermann_values),
        rel=1e-6,
    )


def test_compute_window_metrics_trending_series() -> None:
    records = [
        _record(0.0, 100.0, si=0.75),
        _record(1.0, 102.0, si=0.80),
        _record(2.0, 104.0, si=0.85),
        _record(3.0, 106.0, si=0.82),
        _record(4.0, 108.0, si=0.78),
        _record(5.0, 110.0, si=0.76),
    ]

    metrics = compute_window_metrics(records)

    assert isinstance(metrics, WindowMetrics)
    assert metrics.si == pytest.approx(sum(record.si for record in records) / len(records))
    assert metrics.d_nfr_couple == pytest.approx(1.84)
    assert metrics.d_nfr_res == pytest.approx(1.84)
    assert metrics.d_nfr_flat == pytest.approx(1.84)
    assert metrics.nu_f == pytest.approx(0.0)
    assert metrics.nu_exc == pytest.approx(0.0)
    assert metrics.rho == pytest.approx(0.0)
    assert metrics.phase_lag == pytest.approx(0.0)
    assert metrics.phase_alignment == pytest.approx(1.0)
    assert metrics.phase_synchrony_index == pytest.approx(1.0)
    assert metrics.useful_dissonance_ratio == pytest.approx(0.0)
    assert metrics.useful_dissonance_percentage == pytest.approx(0.0)
    assert metrics.coherence_index == pytest.approx(0.0)
    assert metrics.frequency_label == ""
    assert isinstance(metrics.aero_coherence, AeroCoherence)
    assert metrics.aero_coherence.high_speed_samples == 0
    assert metrics.aero_coherence.low_speed_samples == 0
    assert metrics.aero_mechanical_coherence == pytest.approx(0.0)
    assert isinstance(metrics.aero_balance_drift, AeroBalanceDrift)
    assert metrics.aero_balance_drift.guidance == ""
    expected_variance = pvariance([record.si for record in records])
    assert metrics.si_variance == pytest.approx(expected_variance, rel=1e-6)
    context_matrix = load_context_matrix()
    record_context = [
        resolve_context_from_record(context_matrix, record) for record in records
    ]
    contextual_delta = [
        apply_contextual_delta(
            getattr(record, "delta_nfr", record.nfr),
            factors,
            context_matrix=context_matrix,
        )
        for record, factors in zip(records, record_context)
    ]
    expected_std = pstdev(contextual_delta)
    assert metrics.delta_nfr_std == pytest.approx(expected_std, rel=1e-6)
    assert metrics.nodal_delta_nfr_std == pytest.approx(0.0, abs=1e-9)
    assert metrics.phase_delta_nfr_std == {}
    assert metrics.phase_nodal_delta_nfr_std == {}
    assert metrics.delta_nfr_entropy == pytest.approx(0.0)
    assert metrics.node_entropy == pytest.approx(0.0)
    assert metrics.phase_delta_nfr_entropy == {}
    assert metrics.phase_node_entropy == {}
    assert metrics.epi_derivative_abs == pytest.approx(0.0, abs=1e-9)
    assert metrics.exit_gear_match == pytest.approx(0.0)
    assert metrics.shift_stability == pytest.approx(1.0)


def test_compute_window_metrics_motor_coupling_correlations() -> None:
    records: list[TelemetryRecord] = []
    phase_indices: dict[str, tuple[int, ...]] = {"entry": tuple(range(5)), "exit": tuple(range(5, 10))}
    for index in range(10):
        timestamp = float(index)
        throttle = index / 9.0
        brake = 1.0 - throttle
        acceleration = 2.0 * throttle - 1.0
        records.append(
            _record(
                timestamp,
                100.0 + index,
                throttle=throttle,
                brake_pressure=brake,
                longitudinal_accel=acceleration,
                locking=0.0,
            )
        )

    metrics = compute_window_metrics(records, phase_indices=phase_indices)

    assert metrics.throttle_longitudinal_correlation == pytest.approx(1.0, rel=1e-6)
    assert metrics.brake_longitudinal_correlation == pytest.approx(1.0, rel=1e-6)
    assert metrics.phase_throttle_longitudinal_correlation["entry"] == pytest.approx(
        1.0, rel=1e-6
    )
    assert metrics.phase_brake_longitudinal_correlation["exit"] == pytest.approx(
        1.0, rel=1e-6
    )


def test_compute_window_metrics_motor_coupling_handles_constant_series() -> None:
    records = [
        _record(
            float(index),
            120.0 + index,
            throttle=0.5,
            brake_pressure=0.0,
            longitudinal_accel=(-1.0) ** index,
        )
        for index in range(8)
    ]

    metrics = compute_window_metrics(records)

    assert metrics.throttle_longitudinal_correlation == pytest.approx(0.0, abs=1e-9)
    assert metrics.brake_longitudinal_correlation == pytest.approx(0.0, abs=1e-9)


def test_compute_window_metrics_variance_and_derivative(
    acceptance_records, acceptance_bundle_series
) -> None:
    metrics = compute_window_metrics(acceptance_records, bundles=acceptance_bundle_series)
    expected_variance = pvariance([record.si for record in acceptance_records])
    expected_derivative = sum(
        abs(float(bundle.dEPI_dt)) for bundle in acceptance_bundle_series
    ) / len(acceptance_bundle_series)
    context_matrix = load_context_matrix()
    bundle_context = [
        resolve_context_from_bundle(context_matrix, bundle)
        for bundle in acceptance_bundle_series
    ]
    contextual_delta = [
        apply_contextual_delta(
            bundle.delta_nfr,
            factors,
            context_matrix=context_matrix,
        )
        for bundle, factors in zip(acceptance_bundle_series, bundle_context)
    ]
    expected_delta_std = pstdev(contextual_delta)
    support_samples = [
        max(0.0, bundle.tyres.delta_nfr) + max(0.0, bundle.suspension.delta_nfr)
        for bundle in acceptance_bundle_series
    ]
    expected_nodal_std = pstdev(support_samples)
    assert metrics.si_variance == pytest.approx(expected_variance, rel=1e-6)
    assert metrics.epi_derivative_abs == pytest.approx(expected_derivative, rel=1e-6)
    assert metrics.delta_nfr_std == pytest.approx(expected_delta_std, rel=1e-6)
    assert metrics.nodal_delta_nfr_std == pytest.approx(expected_nodal_std, rel=1e-6)


def test_compute_window_metrics_phase_std_with_bundles(
    acceptance_records, acceptance_bundle_series
) -> None:
    phase_indices = {"entry": (0, 1), "exit": (2, 3)}
    metrics = compute_window_metrics(
        acceptance_records,
        bundles=acceptance_bundle_series,
        phase_indices=phase_indices,
    )
    context_matrix = load_context_matrix()
    bundle_context = [
        resolve_context_from_bundle(context_matrix, bundle)
        for bundle in acceptance_bundle_series
    ]
    contextual_delta = [
        apply_contextual_delta(
            bundle.delta_nfr,
            factors,
            context_matrix=context_matrix,
        )
        for bundle, factors in zip(acceptance_bundle_series, bundle_context)
    ]
    expected_overall = pstdev(contextual_delta)
    assert metrics.delta_nfr_std == pytest.approx(expected_overall, rel=1e-6)
    for label, indices in phase_indices.items():
        subset = [contextual_delta[idx] for idx in indices]
        expected_phase = pstdev(subset)
        assert metrics.phase_delta_nfr_std[label] == pytest.approx(
            expected_phase, rel=1e-6
        )
    support_samples = [
        max(0.0, bundle.tyres.delta_nfr) + max(0.0, bundle.suspension.delta_nfr)
        for bundle in acceptance_bundle_series
    ]
    expected_nodal = pstdev(support_samples)
    assert metrics.nodal_delta_nfr_std == pytest.approx(expected_nodal, rel=1e-6)
    for label, indices in phase_indices.items():
        subset = [support_samples[idx] for idx in indices]
        expected_phase = pstdev(subset)
        assert metrics.phase_nodal_delta_nfr_std[label] == pytest.approx(
            expected_phase, rel=1e-6
        )


def test_compute_window_metrics_entropy_maps(monkeypatch) -> None:
    records = [
        _record(0.0, 100.0),
        _record(0.5, 101.0),
        _record(1.0, 102.0),
        _record(1.5, 103.0),
    ]

    contributions: Mapping[float, Mapping[str, float]] = {
        0.0: {"tyres": 2.0, "brakes": 1.0},
        0.5: {"tyres": 1.0, "brakes": 1.0},
        1.0: {"tyres": 3.0, "suspension": 0.5},
        1.5: {"tyres": 0.5, "suspension": 2.5},
    }

    def _fake_distribution(record: TelemetryRecord) -> Mapping[str, float]:
        return contributions[record.timestamp]

    monkeypatch.setattr(
        "tnfr_lfs.core.metrics.delta_nfr_by_node",
        _fake_distribution,
    )

    phase_indices = {"entry1": (0, 1), "exit4": (2, 3)}
    metrics = compute_window_metrics(records, phase_indices=phase_indices)

    def _total_magnitude(values: Mapping[str, float]) -> float:
        return sum(abs(value) for value in values.values())

    phase_totals = {
        phase: sum(
            _total_magnitude(contributions[records[index].timestamp])
            for index in indices
        )
        for phase, indices in phase_indices.items()
    }
    total = sum(phase_totals.values())

    expected_phase_entropy = normalised_entropy(phase_totals.values())
    assert metrics.delta_nfr_entropy == pytest.approx(
        expected_phase_entropy,
        rel=1e-6,
    )

    expected_phase_distribution = {
        phase: value / total for phase, value in phase_totals.items()
    }
    for phase, probability in expected_phase_distribution.items():
        assert metrics.phase_delta_nfr_entropy[phase] == pytest.approx(
            probability,
            rel=1e-6,
        )
    assert metrics.phase_delta_nfr_entropy["entry"] == pytest.approx(
        expected_phase_distribution["entry1"],
        rel=1e-6,
    )
    assert metrics.phase_delta_nfr_entropy["exit"] == pytest.approx(
        expected_phase_distribution["exit4"],
        rel=1e-6,
    )

    node_totals: dict[str, float] = {}
    for values in contributions.values():
        for node, magnitude in values.items():
            node_totals[node] = node_totals.get(node, 0.0) + abs(magnitude)

    expected_node_entropy = normalised_entropy(node_totals.values())
    assert metrics.node_entropy == pytest.approx(
        expected_node_entropy,
        rel=1e-6,
    )

    phase_node_totals: dict[str, dict[str, float]] = {}
    for phase, indices in phase_indices.items():
        phase_totals_by_node: dict[str, float] = {}
        for index in indices:
            for node, magnitude in contributions[records[index].timestamp].items():
                phase_totals_by_node[node] = (
                    phase_totals_by_node.get(node, 0.0) + abs(magnitude)
                )
        phase_node_totals[phase] = phase_totals_by_node

    for phase, totals in phase_node_totals.items():
        expected_entropy = normalised_entropy(totals.values())
        assert metrics.phase_node_entropy[phase] == pytest.approx(
            expected_entropy,
            rel=1e-6,
        )

    entry_aggregate = {node: 0.0 for node in node_totals}
    exit_aggregate = {node: 0.0 for node in node_totals}
    for phase, totals in phase_node_totals.items():
        if phase.startswith("entry"):
            for node, magnitude in totals.items():
                entry_aggregate[node] += magnitude
        if phase.startswith("exit"):
            for node, magnitude in totals.items():
                exit_aggregate[node] += magnitude

    assert metrics.phase_node_entropy["entry"] == pytest.approx(
        normalised_entropy(entry_aggregate.values()),
        rel=1e-6,
    )
    assert metrics.phase_node_entropy["exit"] == pytest.approx(
        normalised_entropy(exit_aggregate.values()),
        rel=1e-6,
    )

def test_compute_window_metrics_handles_small_windows() -> None:
    single = _record(0.0, 120.0)
    metrics = compute_window_metrics([single])
    assert metrics.si == pytest.approx(0.8)
    assert metrics.d_nfr_couple == 0.0
    assert metrics.d_nfr_res == 0.0
    assert metrics.d_nfr_flat == 0.0
    assert metrics.nu_f == pytest.approx(0.0)
    assert metrics.nu_exc == pytest.approx(0.0)
    assert metrics.rho == pytest.approx(0.0)
    assert metrics.phase_lag == pytest.approx(0.0)
    assert metrics.phase_alignment == pytest.approx(1.0)
    assert metrics.useful_dissonance_ratio == pytest.approx(0.0)
    assert metrics.useful_dissonance_percentage == pytest.approx(0.0)
    assert metrics.coherence_index == pytest.approx(0.0)
    assert metrics.frequency_label == ""
    assert metrics.aero_coherence.high_speed_samples == 0
    assert metrics.aero_mechanical_coherence == pytest.approx(0.0)
    assert metrics.si_variance == pytest.approx(0.0, abs=1e-9)
    assert metrics.epi_derivative_abs == pytest.approx(0.0, abs=1e-9)
    assert metrics.exit_gear_match == pytest.approx(0.0)
    assert metrics.shift_stability == pytest.approx(1.0)


def test_compute_window_metrics_suspension_velocity_histograms() -> None:
    base = _record(0.0, 110.0)
    front_values = [0.02, 0.08, 0.3, -0.02, -0.15, -0.35]
    rear_values = [0.01, 0.12, 0.25, -0.04, -0.18, -0.5]
    records = [
        replace(
            base,
            timestamp=float(index),
            nfr=110.0 + index,
            suspension_velocity_front=front_values[index],
            suspension_velocity_rear=rear_values[index],
        )
        for index in range(len(front_values))
    ]

    metrics = compute_window_metrics(records)
    front_bands = metrics.suspension_velocity_front
    rear_bands = metrics.suspension_velocity_rear

    assert front_bands.compression_low_ratio == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert front_bands.compression_medium_ratio == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert front_bands.compression_high_ratio == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert front_bands.rebound_low_ratio == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert front_bands.rebound_medium_ratio == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert front_bands.rebound_high_ratio == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert front_bands.compression_high_speed_percentage == pytest.approx(33.333, rel=1e-3)
    assert front_bands.rebound_high_speed_percentage == pytest.approx(33.333, rel=1e-3)
    assert front_bands.ar_index == pytest.approx(0.769, rel=1e-3)

    assert rear_bands.compression_high_ratio == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert rear_bands.rebound_high_ratio == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert rear_bands.compression_high_speed_percentage == pytest.approx(33.333, rel=1e-3)
    assert rear_bands.rebound_high_speed_percentage == pytest.approx(33.333, rel=1e-3)
    assert rear_bands.ar_index == pytest.approx(0.528, rel=1e-3)


def _longitudinal_bundle(
    timestamp: float,
    delta_long: float,
    travel_front: float,
    travel_rear: float,
    *,
    si: float = 0.8,
) -> EPIBundle:
    share = delta_long / 7.0
    return EPIBundle(
        timestamp=timestamp,
        epi=0.0,
        delta_nfr=delta_long,
        delta_nfr_proj_longitudinal=delta_long,
        delta_nfr_proj_lateral=0.0,
        sense_index=si,
        tyres=TyresNode(delta_nfr=share, sense_index=si),
        suspension=SuspensionNode(
            delta_nfr=share,
            sense_index=si,
            travel_front=travel_front,
            travel_rear=travel_rear,
        ),
        chassis=ChassisNode(
            delta_nfr=share,
            sense_index=si,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            yaw_rate=0.0,
            lateral_accel=0.0,
            longitudinal_accel=0.0,
        ),
        brakes=BrakesNode(
            delta_nfr=share,
            sense_index=si,
            brake_pressure=0.0,
            locking=0.0,
        ),
        transmission=TransmissionNode(
            delta_nfr=share,
            sense_index=si,
            throttle=0.0,
            gear=3,
            speed=0.0,
            longitudinal_accel=0.0,
            rpm=0.0,
            line_deviation=0.0,
        ),
        track=TrackNode(
            delta_nfr=share,
            sense_index=si,
            axle_load_balance=0.0,
            axle_velocity_balance=0.0,
            yaw=0.0,
            lateral_accel=0.0,
        ),
        driver=DriverNode(
            delta_nfr=share,
            sense_index=si,
            steer=0.0,
            throttle=0.0,
            style_index=si,
        ),
    )


def test_compute_window_metrics_shift_metrics() -> None:
    records = [
        replace(
            _record(0.0, 30.0, si=0.78),
            speed=42.0,
            gear=3,
            rpm=4800.0,
        ),
        replace(
            _record(0.1, 28.0, si=0.76),
            speed=18.0,
            gear=2,
            rpm=5200.0,
        ),
        replace(
            _record(0.2, 32.0, si=0.79),
            speed=25.0,
            gear=3,
            rpm=4800.0,
        ),
        replace(
            _record(0.3, 34.0, si=0.8),
            speed=30.0,
            gear=4,
            rpm=3000.0,
        ),
        replace(
            _record(0.4, 36.0, si=0.82),
            speed=40.0,
            gear=4,
            rpm=5000.0,
        ),
    ]

    metrics = compute_window_metrics(records)

    assert metrics.shift_stability == pytest.approx(1.0 - 2.0 / 3.0, rel=1e-6)
    assert metrics.exit_gear_match == pytest.approx(0.8888888889, rel=1e-6)


def test_compute_window_metrics_bottoming_ratio_tracks_overlap() -> None:
    records: list[TelemetryRecord] = []
    bundles: list[EPIBundle] = []
    travels_front = [0.02, 0.012, 0.011, 0.018, 0.009]
    delta_long = [0.12, 0.52, 0.55, 0.18, 0.57]
    for index, (front, delta_value) in enumerate(zip(travels_front, delta_long)):
        timestamp = index * 0.1
        record = _record(timestamp, 100.0 + index, si=0.82)
        records.append(
            replace(
                record,
                suspension_travel_front=front,
                suspension_travel_rear=0.026,
            )
        )
        bundles.append(
            _longitudinal_bundle(
                timestamp,
                delta_value,
                travel_front=front,
                travel_rear=0.026,
                si=record.si,
            )
        )

    metrics = compute_window_metrics(
        records,
        bundles=bundles,
        objectives={"bottoming_delta_nfr_threshold": 0.4},
    )

    assert metrics.bottoming_ratio_front == pytest.approx(1.0)
    assert metrics.bottoming_ratio_rear == pytest.approx(0.0)


def test_compute_window_metrics_bumpstop_histogram_energy() -> None:
    records: list[TelemetryRecord] = []
    bundles: list[EPIBundle] = []
    travels_front = [0.02, 0.012, 0.009, 0.006, 0.003]
    travels_rear = [0.025, 0.024, 0.026, 0.014, 0.004]
    delta_long = [0.05, 0.35, 0.48, 0.5, 0.6]
    for index, (front, rear, delta_value) in enumerate(
        zip(travels_front, travels_rear, delta_long)
    ):
        timestamp = index * 0.1
        record = _record(timestamp, 120.0 + index, si=0.83)
        records.append(
            replace(
                record,
                suspension_travel_front=front,
                suspension_travel_rear=rear,
            )
        )
        bundles.append(
            _longitudinal_bundle(
                timestamp,
                delta_value,
                travel_front=front,
                travel_rear=rear,
                si=record.si,
            )
        )

    metrics = compute_window_metrics(records, bundles=bundles)

    histogram = metrics.bumpstop_histogram
    assert histogram.front_total_density == pytest.approx(0.8, rel=1e-6)
    assert histogram.rear_total_density == pytest.approx(0.4, rel=1e-6)
    assert histogram.front_energy[0] == pytest.approx(0.05, rel=1e-6)
    assert histogram.front_energy[1] == pytest.approx(0.0685714286, rel=1e-6)
    assert histogram.front_energy[2] == pytest.approx(0.0642857143, rel=1e-6)
    assert histogram.front_energy[3] == pytest.approx(0.0447204969, rel=1e-6)
    assert histogram.rear_energy[0] == pytest.approx(0.0071428571, rel=1e-6)
    assert histogram.rear_energy[2] == pytest.approx(0.0409937888, rel=1e-6)
    assert histogram.front_total_energy == pytest.approx(0.2275776398, rel=1e-6)
    assert histogram.rear_total_energy == pytest.approx(0.0481366460, rel=1e-6)


def test_compute_window_metrics_mu_usage_ratios() -> None:
    records = [
        replace(
            _record(0.0, 100.0),
            mu_eff_front_lateral=0.8,
            mu_eff_front_longitudinal=0.6,
            mu_eff_rear_lateral=0.7,
            mu_eff_rear_longitudinal=0.5,
        ),
        replace(
            _record(0.5, 101.0),
            mu_eff_front_lateral=0.9,
            mu_eff_front_longitudinal=0.4,
            mu_eff_rear_lateral=0.65,
            mu_eff_rear_longitudinal=0.45,
        ),
        replace(
            _record(1.1, 102.0),
            mu_eff_front_lateral=0.85,
            mu_eff_front_longitudinal=0.55,
            mu_eff_rear_lateral=0.75,
            mu_eff_rear_longitudinal=0.5,
        ),
    ]

    objectives = {"mu_max_front": 1.1, "mu_max_rear": 1.05}

    metrics = compute_window_metrics(records, phase_indices=[1, 2], objectives=objectives)

    assert metrics.mu_usage_front_ratio == pytest.approx(0.9090054479581015, rel=1e-6)
    assert metrics.mu_usage_rear_ratio == pytest.approx(0.8104912544074854, rel=1e-6)
    assert metrics.phase_mu_usage_front_ratio == pytest.approx(0.9203843968780266, rel=1e-6)
    assert metrics.phase_mu_usage_rear_ratio == pytest.approx(0.8584645893961879, rel=1e-6)
    context_matrix = load_context_matrix()
    record_context = [
        resolve_context_from_record(context_matrix, record) for record in records
    ]
    contextual_delta = [
        apply_contextual_delta(
            getattr(record, "delta_nfr", record.nfr),
            factors,
            context_matrix=context_matrix,
        )
        for record, factors in zip(records, record_context)
    ]
    expected_overall_std = pstdev(contextual_delta)
    expected_phase_std = pstdev(contextual_delta[1:3])
    assert metrics.delta_nfr_std == pytest.approx(expected_overall_std, rel=1e-6)
    assert metrics.phase_delta_nfr_std.get("active") == pytest.approx(
        expected_phase_std, rel=1e-6
    )
    assert metrics.nodal_delta_nfr_std == pytest.approx(0.0, abs=1e-9)
    assert metrics.phase_nodal_delta_nfr_std.get("active", 0.0) == pytest.approx(
        0.0, abs=1e-9
    )


def test_compute_window_metrics_aero_balance_drift_bins() -> None:
    base = _record(0.0, 110.0, si=0.82)

    def _rake_value(pitch: float, front: float, rear: float) -> float:
        travel_delta = rear - front
        if abs(front) > 1e-9:
            travel_ratio = rear / front
        elif abs(rear) > 1e-9:
            travel_ratio = math.copysign(10.0, rear)
        else:
            travel_ratio = 1.0
        if not math.isfinite(travel_ratio):
            travel_ratio = 1.0
        travel_ratio = max(-10.0, min(10.0, travel_ratio))
        return pitch + math.atan2(travel_delta, travel_ratio)

    records = [
        replace(
            base,
            timestamp=0.0,
            speed=20.0,
            pitch=0.008,
            suspension_travel_front=0.015,
            suspension_travel_rear=0.025,
            mu_eff_front=1.05,
            mu_eff_rear=1.08,
        ),
        replace(
            base,
            timestamp=0.1,
            speed=40.0,
            pitch=0.012,
            suspension_travel_front=0.02,
            suspension_travel_rear=0.03,
            mu_eff_front=1.02,
            mu_eff_rear=0.99,
        ),
        replace(
            base,
            timestamp=0.2,
            speed=60.0,
            pitch=0.018,
            suspension_travel_front=0.03,
            suspension_travel_rear=0.05,
            mu_eff_front=1.1,
            mu_eff_rear=0.9,
        ),
        replace(
            base,
            timestamp=0.3,
            speed=62.0,
            pitch=0.019,
            suspension_travel_front=0.028,
            suspension_travel_rear=0.055,
            mu_eff_front=1.15,
            mu_eff_rear=0.88,
        ),
    ]

    metrics = compute_window_metrics(records)
    drift = metrics.aero_balance_drift

    assert isinstance(drift, AeroBalanceDrift)
    assert drift.low_speed.samples == 1
    assert drift.medium_speed.samples == 1
    assert drift.high_speed.samples == 2

    expected_high_rakes = [
        _rake_value(0.018, 0.03, 0.05),
        _rake_value(0.019, 0.028, 0.055),
    ]
    expected_high_mean = sum(expected_high_rakes) / len(expected_high_rakes)
    expected_high_std = math.sqrt(pvariance(expected_high_rakes))
    expected_high_mu_front = (1.1 + 1.15) / 2.0
    expected_high_mu_rear = (0.9 + 0.88) / 2.0
    expected_high_delta = expected_high_mu_front - expected_high_mu_rear
    expected_high_ratio = expected_high_mu_front / expected_high_mu_rear

    assert drift.high_speed.rake_mean == pytest.approx(expected_high_mean, rel=1e-6)
    assert drift.high_speed.rake_std == pytest.approx(expected_high_std, rel=1e-6)
    assert drift.high_speed.mu_front_mean == pytest.approx(expected_high_mu_front, rel=1e-6)
    assert drift.high_speed.mu_rear_mean == pytest.approx(expected_high_mu_rear, rel=1e-6)
    assert drift.high_speed.mu_delta == pytest.approx(expected_high_delta, rel=1e-6)
    assert drift.high_speed.mu_ratio == pytest.approx(expected_high_ratio, rel=1e-6)

    dominant = drift.dominant_bin()
    assert dominant is not None
    band_label, direction, payload = dominant
    assert band_label == "high"
    assert direction == "front axle"
    assert payload is drift.high_speed
    assert "μΔ" in drift.guidance or drift.guidance == ""


def test_compute_window_metrics_aero_balance_drift_balance_slopes() -> None:
    base = _record(0.0, 110.0, si=0.82)

    def _sample(
        timestamp: float,
        speed: float,
        pitch: float,
        front_lat: float,
        front_long: float,
        rear_lat: float,
        rear_long: float,
    ) -> TelemetryRecord:
        front_total = front_lat + front_long
        rear_total = rear_lat + rear_long
        return replace(
            base,
            timestamp=timestamp,
            speed=speed,
            pitch=pitch,
            suspension_travel_front=0.0,
            suspension_travel_rear=0.0,
            mu_eff_front=front_total / 2.0,
            mu_eff_rear=rear_total / 2.0,
            mu_eff_front_lateral=front_lat,
            mu_eff_front_longitudinal=front_long,
            mu_eff_rear_lateral=rear_lat,
            mu_eff_rear_longitudinal=rear_long,
        )

    records = [
        _sample(0.0, 25.0, 0.01, 0.9, 1.0, 1.1, 1.0),
        _sample(0.1, 30.0, 0.02, 1.2, 1.2, 1.0, 1.0),
        _sample(0.2, 40.0, 0.02, 1.3, 1.2, 0.8, 0.7),
        _sample(0.3, 45.0, 0.05, 1.1, 0.9, 0.9, 0.7),
        _sample(0.4, 60.0, 0.03, 1.1, 0.9, 0.8, 0.7),
        _sample(0.5, 65.0, 0.06, 1.4, 1.2, 0.8, 0.6),
    ]

    metrics = compute_window_metrics(records)
    drift = metrics.aero_balance_drift

    assert drift.low_speed.samples == 2
    assert drift.medium_speed.samples == 2
    assert drift.high_speed.samples == 2

    assert drift.low_speed.mu_balance_slope > 0.0
    assert drift.medium_speed.mu_balance_slope < 0.0
    assert drift.high_speed.mu_balance_slope > 0.0

    assert drift.low_speed.mu_balance_sign_change is True
    assert drift.medium_speed.mu_balance_sign_change is False
    assert drift.high_speed.mu_balance_sign_change is False

    assert "μβ sensitivity" in drift.guidance or drift.guidance == ""


def test_compute_window_metrics_brake_headroom_components() -> None:
    longitudinal = [-4.0, -6.5, -8.0, -7.2, -5.5]
    locking = [0.0, 0.2, 0.7, 0.85, 0.92]
    slip_profiles = [
        (-0.02, -0.03, -0.01, -0.02),
        (-0.08, -0.12, -0.07, -0.09),
        (-0.4, -0.32, -0.18, -0.22),
        (-0.5, -0.47, -0.38, -0.2),
        (-0.1, -0.08, -0.07, -0.06),
    ]
    records = [
        replace(
            _record(float(index), 100.0 + index * 2.0),
            longitudinal_accel=longitudinal[index],
            locking=locking[index],
            slip_ratio_fl=slip_profiles[index][0],
            slip_ratio_fr=slip_profiles[index][1],
            slip_ratio_rl=slip_profiles[index][2],
            slip_ratio_rr=slip_profiles[index][3],
        )
        for index in range(len(longitudinal))
    ]

    metrics = compute_window_metrics(records)

    headroom = metrics.brake_headroom
    assert headroom.peak_decel == pytest.approx(8.0)
    assert headroom.abs_activation_ratio == pytest.approx(sum(locking) / len(locking))
    assert headroom.partial_locking_ratio == pytest.approx(0.6, rel=1e-6)
    assert headroom.sustained_locking_ratio == pytest.approx(0.4, rel=1e-6)
    assert headroom.value == pytest.approx(0.08324, rel=1e-5)


def test_compute_window_metrics_brake_fade_and_ventilation() -> None:
    base = _record(0.0, 100.0)
    timestamps = [0.0, 0.4, 0.8, 1.2]
    decels = [9.0, 8.6, 8.2, 7.0]
    brake_profiles = [
        (620.0, 625.0, 610.0, 615.0),
        (650.0, 655.0, 640.0, 645.0),
        (680.0, 685.0, 670.0, 675.0),
        (710.0, 715.0, 700.0, 705.0),
    ]
    records = []
    for idx, timestamp in enumerate(timestamps):
        temps = brake_profiles[idx]
        records.append(
            replace(
                base,
                timestamp=timestamp,
                longitudinal_accel=-decels[idx],
                brake_pressure=0.92,
                brake_temp_fl=temps[0],
                brake_temp_fr=temps[1],
                brake_temp_rl=temps[2],
                brake_temp_rr=temps[3],
            )
        )

    metrics = compute_window_metrics(records)
    headroom = metrics.brake_headroom

    duration = timestamps[-1] - timestamps[0]
    drop = decels[0] - decels[-1]
    expected_slope = drop / duration
    expected_ratio = drop / decels[0]
    expected_peak = max(temp for profile in brake_profiles for temp in profile)
    expected_mean = sum(sum(profile) / len(profile) for profile in brake_profiles) / len(brake_profiles)

    assert headroom.fade_slope == pytest.approx(expected_slope, rel=1e-6)
    assert headroom.fade_ratio == pytest.approx(expected_ratio, rel=1e-6)
    assert headroom.temperature_peak == pytest.approx(expected_peak, rel=1e-6)
    assert headroom.temperature_mean == pytest.approx(expected_mean, rel=1e-6)
    assert headroom.ventilation_alert == "critica"
    assert headroom.ventilation_index == pytest.approx(1.0, rel=1e-6)
    assert headroom.value < 0.5


def test_compute_window_metrics_without_brake_temperature_samples() -> None:
    base = _record(0.0, 100.0)
    records = [
        replace(
            base,
            timestamp=index * 0.5,
            brake_pressure=0.95,
            longitudinal_accel=-6.0,
            brake_temp_fl=math.nan,
            brake_temp_fr=math.nan,
            brake_temp_rl=math.nan,
            brake_temp_rr=math.nan,
        )
        for index in range(4)
    ]

    metrics = compute_window_metrics(records)
    headroom = metrics.brake_headroom

    assert headroom.temperature_available is False
    assert math.isnan(headroom.temperature_peak)
    assert math.isnan(headroom.temperature_mean)
    assert math.isnan(headroom.fade_ratio)
    assert math.isnan(headroom.fade_slope)
    assert math.isnan(headroom.ventilation_index)
    assert headroom.fade_available is False


def test_slide_catch_budget_aggregates_components() -> None:
    base = _record(0.0, 100.0)
    records = [
        replace(base, timestamp=0.0, yaw_rate=0.0, steer=0.0, throttle=0.3),
        replace(base, timestamp=0.5, yaw_rate=0.3, steer=0.4, throttle=0.35),
        replace(base, timestamp=1.0, yaw_rate=-0.2, steer=-0.3, throttle=0.32),
        replace(base, timestamp=1.5, yaw_rate=0.1, steer=0.1, throttle=0.34),
    ]
    bundles = [_steering_bundle(record, 0.18) for record in records]
    metrics = compute_window_metrics(records, bundles=bundles)
    budget = metrics.slide_catch_budget
    assert budget.yaw_acceleration_ratio == pytest.approx(1.0)
    assert budget.steer_velocity_ratio == pytest.approx(0.285714, rel=1e-3)
    assert budget.overshoot_ratio == pytest.approx(1.0)
    expected_combined = 0.5 * 1.0 + 0.3 * 0.285714 + 0.2 * 1.0
    assert budget.value == pytest.approx(max(0.0, 1.0 - expected_combined), rel=1e-3)


def test_locking_window_score_detects_throttle_transitions() -> None:
    base = _record(0.0, 120.0)
    def _sample(ts: float, throttle: float, locking: float, yaw_rate: float, delta_long: float):
        record = replace(
            base,
            timestamp=ts,
            throttle=throttle,
            locking=locking,
            yaw_rate=yaw_rate,
        )
        object.__setattr__(record, "delta_nfr_proj_longitudinal", delta_long)
        return record

    records = [
        _sample(0.0, 0.05, 0.1, 0.1, 20.0),
        _sample(0.4, 0.1, 0.2, 0.3, 30.0),
        _sample(0.8, 0.6, 0.8, 0.8, 200.0),
        _sample(1.2, 0.7, 0.7, 1.0, 210.0),
        _sample(1.6, 0.2, 0.6, 0.5, 180.0),
        _sample(2.0, 0.05, 0.2, 0.2, 40.0),
    ]

    metrics = compute_window_metrics(records)
    score = metrics.locking_window_score

    assert score.transition_samples == 3
    assert score.on_throttle == pytest.approx(0.186154, rel=1e-6)
    assert score.off_throttle == pytest.approx(0.585385, rel=1e-6)
    assert score.value == pytest.approx(0.452308, rel=1e-6)


def test_aero_balance_drift_normalises_mu_metrics() -> None:
    base = _record(0.0, 110.0, speed=60.0)

    def _sample(
        timestamp: float,
        speed: float,
        front_lat: float,
        front_long: float,
        rear_lat: float,
        rear_long: float,
    ) -> TelemetryRecord:
        return replace(
            base,
            timestamp=timestamp,
            speed=speed,
            mu_eff_front=front_lat + front_long,
            mu_eff_rear=rear_lat + rear_long,
            mu_eff_front_lateral=front_lat,
            mu_eff_front_longitudinal=front_long,
            mu_eff_rear_lateral=rear_lat,
            mu_eff_rear_longitudinal=rear_long,
        )

    records = [
        _sample(0.0, 60.0, 1.0, 0.5, 0.8, 0.4),
        _sample(0.5, 62.0, 1.2, 0.6, 0.7, 0.5),
        _sample(1.0, 65.0, 0.9, 0.7, 0.9, 0.6),
    ]

    metrics = compute_window_metrics(records, phase_indices={"entry": (0, 1), "exit": (2,)})
    drift = metrics.aero_balance_drift.high_speed

    front_lat = [1.0, 1.2, 0.9]
    front_long = [0.5, 0.6, 0.7]
    rear_lat = [0.8, 0.7, 0.9]
    rear_long = [0.4, 0.5, 0.6]

    def _symmetry(lat_values: list[float], long_values: list[float]) -> float:
        lat_mean = sum(lat_values) / len(lat_values)
        long_mean = sum(long_values) / len(long_values)
        denominator = abs(lat_mean) + abs(long_mean)
        return (lat_mean - long_mean) / denominator if denominator > 0 else 0.0

    def _balance() -> float:
        front_total = (sum(front_lat) / len(front_lat)) + (sum(front_long) / len(front_long))
        rear_total = (sum(rear_lat) / len(rear_lat)) + (sum(rear_long) / len(rear_long))
        denominator = abs(front_total) + abs(rear_total)
        return (front_total - rear_total) / denominator if denominator > 0 else 0.0

    assert drift.mu_balance == pytest.approx(_balance(), rel=1e-6)
    assert drift.mu_symmetry_front == pytest.approx(_symmetry(front_lat, front_long), rel=1e-6)
    assert drift.mu_symmetry_rear == pytest.approx(_symmetry(rear_lat, rear_long), rel=1e-6)
    assert metrics.mu_balance == pytest.approx(_balance(), rel=1e-6)

    window_symmetry = metrics.mu_symmetry.get("window", {})
    assert window_symmetry.get("front", 0.0) == pytest.approx(
        _symmetry(front_lat, front_long), rel=1e-6
    )
    assert window_symmetry.get("rear", 0.0) == pytest.approx(
        _symmetry(rear_lat, rear_long), rel=1e-6
    )

    entry_symmetry = metrics.mu_symmetry.get("entry", {})
    assert entry_symmetry.get("front", 0.0) == pytest.approx(
        _symmetry(front_lat[:2], front_long[:2]), rel=1e-6
    )
    assert entry_symmetry.get("rear", 0.0) == pytest.approx(
        _symmetry(rear_lat[:2], rear_long[:2]), rel=1e-6
    )

    exit_symmetry = metrics.mu_symmetry.get("exit", {})
    assert exit_symmetry.get("front", 0.0) == pytest.approx(
        _symmetry(front_lat[2:], front_long[2:]), rel=1e-6
    )
    assert exit_symmetry.get("rear", 0.0) == pytest.approx(
        _symmetry(rear_lat[2:], rear_long[2:]), rel=1e-6
    )


def test_compute_window_metrics_empty_window() -> None:
    metrics = compute_window_metrics([])
    assert metrics == WindowMetrics(
        si=0.0,
        si_variance=0.0,
        d_nfr_couple=0.0,
        d_nfr_res=0.0,
        d_nfr_flat=0.0,
        nu_f=0.0,
        nu_exc=0.0,
        rho=0.0,
        phase_lag=0.0,
        phase_alignment=1.0,
        phase_synchrony_index=1.0,
        motor_latency_ms=0.0,
        phase_motor_latency_ms={},
        useful_dissonance_ratio=0.0,
        useful_dissonance_percentage=0.0,
        coherence_index=0.0,
        ackermann_parallel_index=0.0,
        slide_catch_budget=SlideCatchBudget(),
        locking_window_score=LockingWindowScore(),
        support_effective=0.0,
        load_support_ratio=0.0,
        structural_expansion_longitudinal=0.0,
        structural_contraction_longitudinal=0.0,
        structural_expansion_lateral=0.0,
        structural_contraction_lateral=0.0,
        bottoming_ratio_front=0.0,
        bottoming_ratio_rear=0.0,
        mu_usage_front_ratio=0.0,
        mu_usage_rear_ratio=0.0,
        phase_mu_usage_front_ratio=0.0,
        phase_mu_usage_rear_ratio=0.0,
        mu_balance=0.0,
        mu_symmetry={},
        exit_gear_match=0.0,
        shift_stability=0.0,
        frequency_label="",
        aero_coherence=AeroCoherence(),
        aero_mechanical_coherence=0.0,
        epi_derivative_abs=0.0,
        brake_headroom=BrakeHeadroom(),
        bumpstop_histogram=BumpstopHistogram(),
        cphi=CPHIReport(),
        phase_cphi={},
        suspension_velocity_front=SuspensionVelocityBands(),
        suspension_velocity_rear=SuspensionVelocityBands(),
        aero_balance_drift=AeroBalanceDrift(),
        phase_delta_nfr_std={},
        phase_nodal_delta_nfr_std={},
    )


def test_compute_aero_coherence_splits_bins() -> None:
    records = [
        replace(_record(0.0, 100.0, si=0.8), speed=25.0),
        replace(_record(1.0, 101.0, si=0.805), speed=35.0),
        replace(_record(2.0, 102.0, si=0.81), speed=60.0),
    ]
    bundles = [
        SimpleNamespace(
            delta_breakdown={"tyres": {"mu_eff_front": 0.4, "mu_eff_rear": 0.2}},
            transmission=SimpleNamespace(speed=25.0),
        ),
        SimpleNamespace(
            delta_breakdown={
                "tyres": {
                    "mu_eff_front_lateral": 0.3,
                    "mu_eff_front_longitudinal": 0.1,
                    "mu_eff_rear_lateral": 0.2,
                    "mu_eff_rear_longitudinal": 0.05,
                }
            },
            transmission=SimpleNamespace(speed=35.0),
        ),
        SimpleNamespace(
            delta_breakdown={"tyres": {"mu_eff_front": 0.1, "mu_eff_rear": 0.6}},
            transmission=SimpleNamespace(speed=60.0),
        ),
    ]

    aero = compute_aero_coherence(records, bundles, low_speed_threshold=30.0, high_speed_threshold=40.0)

    assert aero.low_speed_samples == 1
    assert aero.medium_speed_samples == 1
    assert aero.high_speed_samples == 1
    assert aero.low_speed_imbalance == pytest.approx(0.2)
    assert aero.medium_speed_imbalance == pytest.approx(0.15)
    assert aero.high_speed_imbalance == pytest.approx(-0.5)
    assert aero.medium_speed.lateral.front == pytest.approx(0.3)
    assert aero.medium_speed.lateral.rear == pytest.approx(0.2)
    assert aero.medium_speed.longitudinal.front == pytest.approx(0.1)
    assert aero.medium_speed.longitudinal.rear == pytest.approx(0.05)
    assert "medium speed bias" in aero.guidance


def test_aero_mechanical_coherence_blends_components() -> None:
    records = [
        replace(_record(0.0, 95.0, si=0.76), speed=28.0),
        replace(_record(1.0, 96.5, si=0.77), speed=32.0),
        replace(_record(2.0, 97.5, si=0.75), speed=58.0),
        replace(_record(3.0, 98.5, si=0.74), speed=62.0),
    ]
    bundles = [
        EPIBundle(
            timestamp=record.timestamp,
            epi=0.0,
            delta_nfr=0.18 + 0.12,
            sense_index=record.si,
            tyres=TyresNode(delta_nfr=0.18, sense_index=record.si),
            suspension=SuspensionNode(delta_nfr=0.12, sense_index=record.si),
            chassis=ChassisNode(delta_nfr=0.0, sense_index=record.si, yaw_rate=record.yaw_rate),
            brakes=BrakesNode(delta_nfr=0.0, sense_index=record.si),
            transmission=TransmissionNode(
                delta_nfr=0.0,
                sense_index=record.si,
                speed=record.speed,
            ),
            track=TrackNode(delta_nfr=0.0, sense_index=record.si),
            driver=DriverNode(delta_nfr=0.0, sense_index=record.si),
            structural_timestamp=float(index) * 0.4,
            delta_breakdown={
                "tyres": {
                    "mu_eff_front": 0.24 - 0.02 * index,
                    "mu_eff_rear": 0.14 + 0.04 * index,
                }
            },
            delta_nfr_proj_longitudinal=0.06,
            delta_nfr_proj_lateral=0.03,
            coherence_index=0.78,
        )
        for index, record in enumerate(records)
    ]

    objectives = {
        "target_sense_index": 0.75,
        "target_delta_nfr": 0.5,
        "target_mechanical_ratio": 0.45,
        "target_aero_imbalance": 0.2,
    }

    metrics = compute_window_metrics(records, bundles=bundles, objectives=objectives)

    assert 0.0 <= metrics.aero_mechanical_coherence <= 1.0
    suspension_deltas = [bundle.suspension.delta_nfr for bundle in bundles]
    tyre_deltas = [bundle.tyres.delta_nfr for bundle in bundles]
    ackermann_values = [float(bundle.ackermann_parallel_index) for bundle in bundles]
    ackermann_clean = [value for value in ackermann_values if math.isfinite(value)]
    if ackermann_clean:
        ackermann_parallel = sum(ackermann_clean) / len(ackermann_clean)
    else:
        ackermann_parallel = 0.0
    rake_velocity_profile = [
        (metrics.aero_balance_drift.low_speed.rake_mean, metrics.aero_balance_drift.low_speed.samples),
        (
            metrics.aero_balance_drift.medium_speed.rake_mean,
            metrics.aero_balance_drift.medium_speed.samples,
        ),
        (
            metrics.aero_balance_drift.high_speed.rake_mean,
            metrics.aero_balance_drift.high_speed.samples,
        ),
    ]
    expected = resolve_aero_mechanical_coherence(
        metrics.coherence_index,
        metrics.aero_coherence,
        suspension_deltas=suspension_deltas,
        tyre_deltas=tyre_deltas,
        target_delta_nfr=objectives["target_delta_nfr"],
        target_mechanical_ratio=objectives["target_mechanical_ratio"],
        target_aero_imbalance=objectives["target_aero_imbalance"],
        rake_velocity_profile=rake_velocity_profile,
        ackermann_parallel_index=ackermann_parallel,
        ackermann_parallel_samples=len(ackermann_clean),
    )
    assert metrics.aero_mechanical_coherence == pytest.approx(expected)


def test_cphi_report_structure_and_legacy_mapping() -> None:
    base = _record(
        0.0,
        100.0,
        slip_ratio=0.0,
        slip_angle=0.0,
        slip_ratio_fl=0.04,
        slip_ratio_fr=0.05,
        slip_ratio_rl=0.03,
        slip_ratio_rr=0.03,
        slip_angle_fl=0.04,
        slip_angle_fr=0.05,
        slip_angle_rl=0.03,
        slip_angle_rr=0.03,
        wheel_load_fl=3200.0,
        wheel_load_fr=3150.0,
        wheel_load_rl=3000.0,
        wheel_load_rr=3050.0,
        wheel_lateral_force_fl=2400.0,
        wheel_lateral_force_fr=2350.0,
        wheel_lateral_force_rl=2100.0,
        wheel_lateral_force_rr=2050.0,
        wheel_longitudinal_force_fl=1500.0,
        wheel_longitudinal_force_fr=1450.0,
        wheel_longitudinal_force_rl=1200.0,
        wheel_longitudinal_force_rr=1150.0,
    )
    records = [
        base,
        replace(
            base,
            timestamp=0.1,
            slip_angle=0.02,
            slip_angle_fl=0.05,
            slip_angle_fr=0.06,
            slip_angle_rl=0.04,
            slip_angle_rr=0.035,
        ),
        replace(
            base,
            timestamp=0.2,
            slip_angle=0.03,
            slip_angle_fl=0.06,
            slip_angle_fr=0.07,
            slip_angle_rl=0.05,
            slip_angle_rr=0.045,
        ),
    ]

    metrics = compute_window_metrics(records)
    report = metrics.cphi

    assert isinstance(report, CPHIReport)
    legacy = report.as_legacy_mapping()
    structured = report.as_dict()

    for suffix in ("fl", "fr", "rl", "rr"):
        wheel = report[suffix]
        assert legacy[f"cphi_{suffix}"] == pytest.approx(wheel.value)
        assert legacy[f"cphi_{suffix}_temperature"] == pytest.approx(
            wheel.temperature_component
        )
        assert legacy[f"cphi_{suffix}_gradient"] == pytest.approx(
            wheel.gradient_component
        )
        assert legacy[f"cphi_{suffix}_mu"] == pytest.approx(wheel.mu_component)
        assert legacy[f"cphi_{suffix}_temp_delta"] == pytest.approx(wheel.temperature_delta)
        assert legacy[f"cphi_{suffix}_gradient_rate"] == pytest.approx(wheel.gradient_rate)

        wheel_payload = structured["wheels"][suffix]
        assert wheel_payload["value"] == pytest.approx(wheel.value)
        assert wheel_payload["status"] == report.classification_for(suffix)
        assert wheel_payload["optimal"] == report.is_optimal_for(suffix)

    assert structured["thresholds"]["red"] == pytest.approx(report.thresholds.red)
    assert structured["thresholds"]["amber"] == pytest.approx(report.thresholds.amber)
    assert structured["thresholds"]["green"] == pytest.approx(report.thresholds.green)


def test_cphi_thresholds_classification() -> None:
    thresholds = CPHIThresholds(red=0.5, amber=0.75, green=0.9)

    assert thresholds.classify(0.3) == "red"
    assert thresholds.classify(0.7) == "amber"
    assert thresholds.classify(0.95) == "green"
    assert thresholds.classify(float("nan")) == "unknown"
    assert not thresholds.is_optimal(0.82)
    assert thresholds.is_optimal(0.93)
