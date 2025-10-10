import math

import pytest

from tests.helpers import build_contextual_delta_record

from tnfr_lfs.core.contextual_delta import (
    ContextFactors,
    apply_contextual_delta,
    load_context_matrix,
    resolve_context_from_record,
)


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
    record = build_contextual_delta_record(
        lateral_accel=lateral_accel, longitudinal_accel=longitudinal_accel
    )
    factors = resolve_context_from_record(matrix, record)
    multiplier = max(
        matrix.min_multiplier,
        min(matrix.max_multiplier, factors.multiplier),
    )
    assert multiplier == pytest.approx(expected_multiplier, rel=1e-3)
