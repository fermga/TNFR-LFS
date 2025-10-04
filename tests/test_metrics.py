from __future__ import annotations

import pytest

from tnfr_lfs.core.metrics import WindowMetrics, compute_window_metrics
from tnfr_lfs.core.epi import TelemetryRecord, resolve_nu_f_by_node


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
    expected_nu_f = sum(
        sum(resolve_nu_f_by_node(record).values()) / len(resolve_nu_f_by_node(record))
        for record in records
    ) / len(records)
    assert metrics.nu_f == pytest.approx(expected_nu_f)


def test_compute_window_metrics_handles_small_windows() -> None:
    single = _record(0.0, 120.0)
    metrics = compute_window_metrics([single])
    assert metrics.si == pytest.approx(0.8)
    assert metrics.d_nfr_couple == 0.0
    assert metrics.d_nfr_res == 0.0
    assert metrics.d_nfr_flat == 0.0
    expected_nu_f = sum(resolve_nu_f_by_node(single).values()) / len(
        resolve_nu_f_by_node(single)
    )
    assert metrics.nu_f == pytest.approx(expected_nu_f)


def test_compute_window_metrics_empty_window() -> None:
    metrics = compute_window_metrics([])
    assert metrics == WindowMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
