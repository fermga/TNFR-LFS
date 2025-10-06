import random

import pytest

from tnfr_lfs.analysis import ABResult, ab_compare_by_lap


class DummyBundle:
    def __init__(self, value: float) -> None:
        self.sense_index = value
        self.delta_nfr = value
        self.coherence_index = value
        self.delta_nfr_longitudinal = value
        self.delta_nfr_lateral = value


def _metrics_from_laps(lap_samples: list[list[float]]) -> dict[str, object]:
    bundles = [DummyBundle(value) for lap in lap_samples for value in lap]
    lap_indices: list[int] = []
    for lap_index, samples in enumerate(lap_samples):
        lap_indices.extend([lap_index] * len(samples))
    return {
        "stages": {
            "coherence": {"bundles": bundles},
            "recepcion": {"lap_indices": lap_indices},
        },
    }


def test_ab_compare_by_lap_computes_statistics() -> None:
    baseline = _metrics_from_laps([[0.60, 0.62], [0.61, 0.63]])
    variant = _metrics_from_laps([[0.65, 0.67], [0.66, 0.68]])
    rng = random.Random(42)

    result = ab_compare_by_lap(
        baseline,
        variant,
        metric="sense_index",
        iterations=256,
        rng=rng,
    )

    assert isinstance(result, ABResult)
    assert result.metric == "sense_index"
    assert pytest.approx(result.baseline_mean, rel=1e-6) == 0.615
    assert pytest.approx(result.variant_mean, rel=1e-6) == 0.665
    assert len(result.baseline_laps) == 2
    assert len(result.variant_laps) == 2
    assert result.bootstrap_low <= result.mean_difference <= result.bootstrap_high
    assert 0.0 <= result.permutation_p_value <= 1.0
    assert 0.0 <= result.estimated_power <= 1.0


def test_ab_compare_by_lap_requires_samples() -> None:
    with pytest.raises(ValueError):
        ab_compare_by_lap({}, {}, metric="sense_index")
