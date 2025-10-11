"""Unit tests for window metrics support calculations."""

from __future__ import annotations

import pytest

from tnfr_lfs.core.metrics import WindowMetrics, compute_window_metrics

from tests.helpers import build_support_bundle, build_telemetry_record


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
        build_support_bundle(
            timestamp=0.0,
            structural_timestamp=0.0,
            tyre_delta=0.4,
            suspension_delta=0.3,
            longitudinal_delta=0.5,
            lateral_delta=-0.1,
            yaw_rate=0.05,
        ),
        build_support_bundle(
            timestamp=1.0,
            structural_timestamp=0.4,
            tyre_delta=0.5,
            suspension_delta=0.2,
            longitudinal_delta=0.2,
            lateral_delta=0.3,
            yaw_rate=0.08,
        ),
        build_support_bundle(
            timestamp=2.0,
            structural_timestamp=0.9,
            tyre_delta=-0.1,
            suspension_delta=0.4,
            longitudinal_delta=-0.2,
            lateral_delta=0.1,
            yaw_rate=0.02,
        ),
        build_support_bundle(
            timestamp=3.0,
            structural_timestamp=1.5,
            tyre_delta=0.2,
            suspension_delta=0.1,
            longitudinal_delta=0.4,
            lateral_delta=-0.2,
            yaw_rate=-0.03,
        ),
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

