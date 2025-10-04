import math

import pytest

from tnfr_lfs.core.contextual_delta import (
    ContextFactors,
    apply_contextual_delta,
    load_context_matrix,
    resolve_context_from_record,
)
from tnfr_lfs.core.epi import TelemetryRecord


def _sample_record(**overrides):
    base = {
        "timestamp": 0.0,
        "vertical_load": 5200.0,
        "slip_ratio": 0.0,
        "lateral_accel": 1.2,
        "longitudinal_accel": 0.2,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "brake_pressure": 0.0,
        "locking": 0.0,
        "nfr": 500.0,
        "si": 0.8,
        "speed": 40.0,
        "yaw_rate": 0.02,
        "slip_angle": 0.01,
        "steer": 0.1,
        "throttle": 0.5,
        "gear": 3,
        "vertical_load_front": 2600.0,
        "vertical_load_rear": 2600.0,
        "mu_eff_front": 1.1,
        "mu_eff_rear": 1.1,
        "mu_eff_front_lateral": 1.1,
        "mu_eff_front_longitudinal": 1.1,
        "mu_eff_rear_lateral": 1.1,
        "mu_eff_rear_longitudinal": 1.1,
        "suspension_travel_front": 0.02,
        "suspension_travel_rear": 0.02,
        "suspension_velocity_front": 0.1,
        "suspension_velocity_rear": 0.1,
    }
    base.update(overrides)
    return TelemetryRecord(**base)


def test_apply_contextual_delta_clamps_multiplier():
    matrix = load_context_matrix()
    delta = 1.0
    excessive = ContextFactors(curve=5.0, surface=5.0, traffic=5.0)
    adjusted = apply_contextual_delta(delta, excessive, context_matrix=matrix)
    assert math.isclose(adjusted, matrix.max_multiplier * delta, rel_tol=1e-6)


@pytest.mark.parametrize(
    "lateral_accel, longitudinal_accel, expected_multiplier",
    [
        (0.5, 0.05, 0.92),
        (1.2, 0.2, 1.0),
        (1.5, 0.3, 0.95),
    ],
)
def test_resolve_context_from_record_matches_profiles(lateral_accel, longitudinal_accel, expected_multiplier):
    matrix = load_context_matrix()
    record = _sample_record(lateral_accel=lateral_accel, longitudinal_accel=longitudinal_accel)
    factors = resolve_context_from_record(matrix, record)
    multiplier = max(
        matrix.min_multiplier,
        min(matrix.max_multiplier, factors.multiplier),
    )
    assert multiplier == pytest.approx(expected_multiplier, rel=1e-3)
