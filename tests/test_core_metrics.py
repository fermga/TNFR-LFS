"""Unit tests for window metrics support calculations."""

from __future__ import annotations

import pytest

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
from tnfr_lfs.core.metrics import WindowMetrics, compute_window_metrics


def _record(timestamp: float, load: float, yaw_rate: float = 0.0) -> TelemetryRecord:
    return TelemetryRecord(
        timestamp=timestamp,
        vertical_load=load,
        slip_ratio=0.0,
        lateral_accel=0.0,
        longitudinal_accel=0.0,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        brake_pressure=0.0,
        locking=0.0,
        nfr=100.0,
        si=0.8,
        speed=0.0,
        yaw_rate=yaw_rate,
        slip_angle=0.0,
        steer=0.0,
        throttle=0.0,
        gear=3,
        vertical_load_front=load * 0.52,
        vertical_load_rear=load * 0.48,
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


def _bundle(
    timestamp: float,
    structural: float,
    tyre_delta: float,
    suspension_delta: float,
    longitudinal_delta: float,
    lateral_delta: float,
    yaw_rate: float,
) -> EPIBundle:
    return EPIBundle(
        timestamp=timestamp,
        epi=0.0,
        delta_nfr=tyre_delta + suspension_delta,
        sense_index=0.8,
        tyres=TyresNode(delta_nfr=tyre_delta, sense_index=0.8),
        suspension=SuspensionNode(delta_nfr=suspension_delta, sense_index=0.8),
        chassis=ChassisNode(delta_nfr=0.0, sense_index=0.8, yaw_rate=yaw_rate),
        brakes=BrakesNode(delta_nfr=0.0, sense_index=0.8),
        transmission=TransmissionNode(delta_nfr=0.0, sense_index=0.8),
        track=TrackNode(delta_nfr=0.0, sense_index=0.8),
        driver=DriverNode(delta_nfr=0.0, sense_index=0.8),
        structural_timestamp=structural,
        delta_nfr_proj_longitudinal=longitudinal_delta,
        delta_nfr_proj_lateral=lateral_delta,
    )


def test_window_metrics_support_efficiency_uses_structural_windows() -> None:
    records = [
        _record(0.0, 4800.0, yaw_rate=0.05),
        _record(1.0, 5000.0, yaw_rate=0.08),
        _record(2.0, 5100.0, yaw_rate=0.02),
        _record(3.0, 5300.0, yaw_rate=-0.03),
    ]
    bundles = [
        _bundle(0.0, 0.0, 0.4, 0.3, 0.5, -0.1, 0.05),
        _bundle(1.0, 0.4, 0.5, 0.2, 0.2, 0.3, 0.08),
        _bundle(2.0, 0.9, -0.1, 0.4, -0.2, 0.1, 0.02),
        _bundle(3.0, 1.5, 0.2, 0.1, 0.4, -0.2, -0.03),
    ]

    metrics = compute_window_metrics(records, bundles=bundles)

    assert isinstance(metrics, WindowMetrics)
    assert metrics.support_effective == pytest.approx(0.44, rel=1e-6)
    assert metrics.load_support_ratio == pytest.approx(0.44 / 5050.0, rel=1e-6)
    assert metrics.structural_expansion_longitudinal == pytest.approx(0.2133333333, rel=1e-6)
    assert metrics.structural_contraction_longitudinal == pytest.approx(0.0666666667, rel=1e-6)
    assert metrics.structural_expansion_lateral == pytest.approx(0.1133333333, rel=1e-6)
    assert metrics.structural_contraction_lateral == pytest.approx(0.08, rel=1e-6)
    assert metrics.si_variance == pytest.approx(0.0, abs=1e-9)
    assert metrics.epi_derivative_abs == pytest.approx(0.0, abs=1e-9)

