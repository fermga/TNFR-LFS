from __future__ import annotations

from random import Random
from typing import Iterable, Sequence

import pytest

from tnfr_lfs.analysis.abtest import ab_compare_by_lap

from test_search import _build_bundle


def _build_stage(lap_samples: Sequence[Sequence[float]]) -> dict[str, object]:
    bundles = []
    lap_indices = []
    timestamp = 0.0
    for lap_index, samples in enumerate(lap_samples):
        for value in samples:
            bundles.append(_build_bundle(timestamp, 0.0, value))
            lap_indices.append(lap_index)
            timestamp += 0.1
    return {
        "stages": {
            "coherence": {"bundles": tuple(bundles)},
            "recepcion": {"lap_indices": tuple(lap_indices)},
        }
    }


def _scale(samples: Iterable[Sequence[float]], factor: float) -> list[list[float]]:
    return [[value * factor for value in lap] for lap in samples]


def test_abtest_reports_expected_improvement() -> None:
    baseline_samples = [
        (0.80, 0.79, 0.81),
        (0.82, 0.80, 0.79),
        (0.78, 0.80, 0.81),
    ]
    variant_samples = _scale(baseline_samples, 1.12)

    baseline_metrics = _build_stage(baseline_samples)
    variant_metrics = _build_stage(variant_samples)

    rng = Random(2024)
    result = ab_compare_by_lap(
        baseline_metrics,
        variant_metrics,
        metric="sense_index",
        iterations=128,
        rng=rng,
    )

    assert len(result.baseline_laps) == 3
    assert len(result.variant_laps) == 3
    assert result.baseline_mean == pytest.approx(0.8000, rel=1e-6)
    assert result.variant_mean == pytest.approx(0.8960, rel=1e-6)
    improvement = (result.variant_mean - result.baseline_mean) / result.baseline_mean
    assert improvement == pytest.approx(0.12, rel=1e-6)
    assert result.mean_difference == pytest.approx(0.0960, rel=1e-6)
    assert 0.0 <= result.permutation_p_value <= 1.0
