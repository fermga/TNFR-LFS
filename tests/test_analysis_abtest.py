import random
from typing import Sequence

import pytest

from tnfr_lfs.analysis import ABResult, ab_compare_by_lap

from tests.test_abtest import build_metrics, scale_samples


@pytest.mark.parametrize(
    (
        "baseline_samples",
        "variant_samples",
        "expected_baseline_mean",
        "expected_variant_mean",
    ),
    [
        (
            (
                (0.80, 0.79, 0.81),
                (0.82, 0.80, 0.79),
                (0.78, 0.80, 0.81),
            ),
            None,
            0.8000,
            0.8960,
        ),
        (
            ((0.60, 0.62), (0.61, 0.63)),
            ((0.65, 0.67), (0.66, 0.68)),
            0.615,
            0.665,
        ),
    ],
)
def test_ab_compare_by_lap_computes_statistics(
    baseline_samples: Sequence[Sequence[float]],
    variant_samples: Sequence[Sequence[float]] | None,
    expected_baseline_mean: float,
    expected_variant_mean: float,
) -> None:
    if variant_samples is None:
        variant_samples = scale_samples(baseline_samples, 1.12)

    baseline_metrics = build_metrics(baseline_samples)
    variant_metrics = build_metrics(variant_samples)

    rng = random.Random(2024)
    result = ab_compare_by_lap(
        baseline_metrics,
        variant_metrics,
        metric="sense_index",
        iterations=256,
        rng=rng,
    )

    assert isinstance(result, ABResult)
    assert result.metric == "sense_index"
    assert len(result.baseline_laps) == len(baseline_samples)
    assert len(result.variant_laps) == len(variant_samples)
    assert result.baseline_mean == pytest.approx(expected_baseline_mean, rel=1e-6)
    assert result.variant_mean == pytest.approx(expected_variant_mean, rel=1e-6)

    improvement = (
        result.variant_mean - result.baseline_mean
    ) / result.baseline_mean
    expected_improvement = (
        expected_variant_mean - expected_baseline_mean
    ) / expected_baseline_mean
    assert improvement == pytest.approx(expected_improvement, rel=1e-6)
    assert result.mean_difference == pytest.approx(
        expected_variant_mean - expected_baseline_mean,
        rel=1e-6,
    )

    assert result.bootstrap_low <= result.mean_difference <= result.bootstrap_high
    assert 0.0 <= result.permutation_p_value <= 1.0
    assert 0.0 <= result.estimated_power <= 1.0


def test_ab_compare_by_lap_requires_samples() -> None:
    with pytest.raises(ValueError):
        ab_compare_by_lap({}, {}, metric="sense_index")
