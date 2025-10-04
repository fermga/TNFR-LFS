from __future__ import annotations

import pytest

from tnfr_lfs.core.metrics import (
    AeroCoherence,
    WindowMetrics,
    compute_aero_coherence,
    compute_window_metrics,
)
from dataclasses import replace
from types import SimpleNamespace
from tnfr_lfs.core.epi import TelemetryRecord


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
    assert metrics.d_nfr_couple == pytest.approx(2.0)
    assert metrics.d_nfr_res == pytest.approx(2.0)
    assert metrics.d_nfr_flat == pytest.approx(2.0)
    assert metrics.nu_f == pytest.approx(0.0)
    assert metrics.nu_exc == pytest.approx(0.0)
    assert metrics.rho == pytest.approx(0.0)
    assert metrics.phase_lag == pytest.approx(0.0)
    assert metrics.phase_alignment == pytest.approx(1.0)
    assert metrics.useful_dissonance_ratio == pytest.approx(0.0)
    assert metrics.useful_dissonance_percentage == pytest.approx(0.0)
    assert isinstance(metrics.aero_coherence, AeroCoherence)
    assert metrics.aero_coherence.high_speed_samples == 0
    assert metrics.aero_coherence.low_speed_samples == 0


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
    assert metrics.aero_coherence.high_speed_samples == 0


def test_compute_window_metrics_empty_window() -> None:
    metrics = compute_window_metrics([])
    assert metrics == WindowMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)


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
