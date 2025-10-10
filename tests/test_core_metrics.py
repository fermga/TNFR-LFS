"""Unit tests for window metrics support calculations."""

from __future__ import annotations

import pytest

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

from tests.helpers import build_telemetry_record


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
        build_telemetry_record(
            0.0,
            vertical_load=4800.0,
            vertical_load_front=4800.0 * 0.52,
            vertical_load_rear=4800.0 * 0.48,
            yaw_rate=0.05,
        ),
        build_telemetry_record(
            1.0,
            vertical_load=5000.0,
            vertical_load_front=5000.0 * 0.52,
            vertical_load_rear=5000.0 * 0.48,
            yaw_rate=0.08,
        ),
        build_telemetry_record(
            2.0,
            vertical_load=5100.0,
            vertical_load_front=5100.0 * 0.52,
            vertical_load_rear=5100.0 * 0.48,
            yaw_rate=0.02,
        ),
        build_telemetry_record(
            3.0,
            vertical_load=5300.0,
            vertical_load_front=5300.0 * 0.52,
            vertical_load_rear=5300.0 * 0.48,
            yaw_rate=-0.03,
        ),
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

