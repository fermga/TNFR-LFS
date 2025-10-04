from __future__ import annotations

import pytest

from tnfr_lfs.core.metrics import (
    AeroCoherence,
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
        support_effective=0.0,
        load_support_ratio=0.0,
        structural_expansion_longitudinal=0.0,
        structural_contraction_longitudinal=0.0,
        structural_expansion_lateral=0.0,
        structural_contraction_lateral=0.0,
        frequency_label="",
        aero_coherence=AeroCoherence(),
        aero_mechanical_coherence=0.0,
        epi_derivative_abs=0.0,
    )


def test_compute_aero_coherence_splits_bins() -> None:
    records = [
        replace(_record(0.0, 100.0, si=0.8), speed=25.0),
        replace(_record(1.0, 102.0, si=0.81), speed=60.0),
    ]
    bundles = [
        SimpleNamespace(
            delta_breakdown={"tyres": {"mu_eff_front": 0.4, "mu_eff_rear": 0.2}},
            transmission=SimpleNamespace(speed=25.0),
        ),
        SimpleNamespace(
            delta_breakdown={"tyres": {"mu_eff_front": 0.1, "mu_eff_rear": 0.6}},
            transmission=SimpleNamespace(speed=60.0),
        ),
    ]

    aero = compute_aero_coherence(records, bundles, low_speed_threshold=30.0, high_speed_threshold=40.0)

    assert aero.low_speed_samples == 1
    assert aero.high_speed_samples == 1
    assert aero.low_speed_imbalance == pytest.approx(0.2)
    assert aero.high_speed_imbalance == pytest.approx(-0.5)
    assert "Aero" in aero.guidance or "Alta velocidad" in aero.guidance


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
