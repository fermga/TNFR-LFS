from __future__ import annotations

import math

import pytest

from tnfr_lfs.core.metrics import (
    AeroCoherence,
    BrakeHeadroom,
    SlideCatchBudget,
    WindowMetrics,
    compute_aero_coherence,
    compute_window_metrics,
    resolve_aero_mechanical_coherence,
)
from dataclasses import replace
from statistics import pvariance
from types import SimpleNamespace
from tnfr_lfs.core.epi import TelemetryRecord
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


def _record(timestamp: float, nfr: float, si: float = 0.8) -> TelemetryRecord:
    return TelemetryRecord(
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


def _steering_bundle(record: TelemetryRecord, ackermann_delta: float) -> EPIBundle:
    share = record.nfr / 7.0
    return EPIBundle(
        timestamp=record.timestamp,
        epi=0.0,
        delta_nfr=record.nfr,
        delta_nfr_longitudinal=0.0,
        delta_nfr_lateral=0.0,
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
    assert metrics.useful_dissonance_ratio == pytest.approx(0.0)
    assert metrics.useful_dissonance_percentage == pytest.approx(0.0)
    assert metrics.coherence_index == pytest.approx(0.0)
    assert metrics.frequency_label == ""
    assert isinstance(metrics.aero_coherence, AeroCoherence)
    assert metrics.aero_coherence.high_speed_samples == 0
    assert metrics.aero_coherence.low_speed_samples == 0
    assert metrics.aero_mechanical_coherence == pytest.approx(0.0)
    expected_variance = pvariance([record.si for record in records])
    assert metrics.si_variance == pytest.approx(expected_variance, rel=1e-6)
    assert metrics.epi_derivative_abs == pytest.approx(0.0, abs=1e-9)
    assert metrics.exit_gear_match == pytest.approx(0.0)
    assert metrics.shift_stability == pytest.approx(1.0)


def test_compute_window_metrics_variance_and_derivative(
    acceptance_records, acceptance_bundle_series
) -> None:
    metrics = compute_window_metrics(acceptance_records, bundles=acceptance_bundle_series)
    expected_variance = pvariance([record.si for record in acceptance_records])
    expected_derivative = sum(
        abs(float(bundle.dEPI_dt)) for bundle in acceptance_bundle_series
    ) / len(acceptance_bundle_series)
    assert metrics.si_variance == pytest.approx(expected_variance, rel=1e-6)
    assert metrics.epi_derivative_abs == pytest.approx(expected_derivative, rel=1e-6)


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
        delta_nfr_longitudinal=delta_long,
        delta_nfr_lateral=0.0,
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
        useful_dissonance_ratio=0.0,
        useful_dissonance_percentage=0.0,
        coherence_index=0.0,
        ackermann_parallel_index=0.0,
        slide_catch_budget=SlideCatchBudget(),
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
        exit_gear_match=0.0,
        shift_stability=0.0,
        frequency_label="",
        aero_coherence=AeroCoherence(),
        aero_mechanical_coherence=0.0,
        epi_derivative_abs=0.0,
        brake_headroom=BrakeHeadroom(),
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
    assert "media velocidad" in aero.guidance


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
            delta_nfr_longitudinal=0.06,
            delta_nfr_lateral=0.03,
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
    expected = resolve_aero_mechanical_coherence(
        metrics.coherence_index,
        metrics.aero_coherence,
        suspension_deltas=suspension_deltas,
        tyre_deltas=tyre_deltas,
        target_delta_nfr=objectives["target_delta_nfr"],
        target_mechanical_ratio=objectives["target_mechanical_ratio"],
        target_aero_imbalance=objectives["target_aero_imbalance"],
    )
    assert metrics.aero_mechanical_coherence == pytest.approx(expected)
