from dataclasses import replace

import pytest

from tnfr_core.metrics import compute_window_metrics
from tnfr_core.structural_time import compute_structural_timestamps

from tests.helpers import build_telemetry_record


def test_structural_time_dense_sequences_expand_span() -> None:
    sparse_records = [
        build_telemetry_record(
            float(index),
            nfr=0.0,
            brake_pressure=0.1 if index == 3 else 0.0,
            yaw_rate=0.05,
            steer=0.02,
        )
        for index in range(8)
    ]
    dense_records = [
        build_telemetry_record(
            float(index),
            nfr=0.0,
            brake_pressure=0.6 if index % 2 else 0.0,
            throttle=0.7 if index % 3 == 0 else 0.2,
            yaw_rate=0.4 if index >= 2 else 0.05,
            steer=0.3 if index >= 4 else 0.1,
        )
        for index in range(8)
    ]

    sparse_axis = compute_structural_timestamps(sparse_records, window_size=3)
    dense_axis = compute_structural_timestamps(dense_records, window_size=3)

    assert sparse_axis[0] == pytest.approx(dense_axis[0])
    assert dense_axis[-1] - dense_axis[0] > sparse_axis[-1] - sparse_axis[0]
    assert all(
        later >= earlier for earlier, later in zip(dense_axis, dense_axis[1:])
    ), "structural axis must be monotonic"


def test_structural_time_modulates_gradients_with_weights() -> None:
    records = [
        build_telemetry_record(
            float(index),
            nfr=float(index),
            brake_pressure=0.8 if index in {1, 2} else 0.1,
        )
        for index in range(6)
    ]

    brake_weights = {"brake_pressure": 0.7, "throttle": 0.1, "yaw_rate": 0.1, "steer": 0.1}
    throttle_weights = {"brake_pressure": 0.1, "throttle": 0.7, "yaw_rate": 0.1, "steer": 0.1}

    brake_axis = compute_structural_timestamps(records, weights=brake_weights, window_size=2)
    throttle_axis = compute_structural_timestamps(records, weights=throttle_weights, window_size=2)

    brake_records = [
        replace(record, structural_timestamp=axis)
        for record, axis in zip(records, brake_axis)
    ]
    throttle_records = [
        replace(record, structural_timestamp=axis)
        for record, axis in zip(records, throttle_axis)
    ]

    brake_metrics = compute_window_metrics(brake_records)
    throttle_metrics = compute_window_metrics(throttle_records)

    assert brake_metrics.d_nfr_couple < throttle_metrics.d_nfr_couple
    assert brake_metrics.d_nfr_res <= throttle_metrics.d_nfr_res
