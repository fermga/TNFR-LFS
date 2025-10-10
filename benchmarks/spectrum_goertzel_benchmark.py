from __future__ import annotations

import math

import numpy as np
import pytest

from tnfr_lfs.core import spectrum

pytestmark = pytest.mark.benchmark(group="spectrum")


def _sine_wave(
    frequency: float,
    sample_rate: float,
    *,
    length: int = 4096,
    phase: float = 0.0,
) -> list[float]:
    time = np.arange(length, dtype=float) / sample_rate
    return np.sin((2.0 * math.pi * frequency * time) + phase).tolist()


def test_fft_cross_spectrum_baseline(benchmark: pytest.BenchmarkFixture) -> None:
    sample_rate = 120.0
    frequency = 2.25
    control = _sine_wave(frequency, sample_rate)
    response = _sine_wave(frequency, sample_rate, phase=math.pi / 7.0)

    def run_fft() -> spectrum.PhaseCorrelation | None:
        return spectrum._resolve_dominant_correlation(
            control,
            response,
            sample_rate,
            dominant_only=True,
            method="fft",
        )

    correlation = benchmark(run_fft)
    assert correlation is not None
    assert correlation.frequency == pytest.approx(frequency, rel=0.05)


@pytest.mark.skipif(
    spectrum._SCIPY_GOERTZEL is None,
    reason="SciPy Goertzel backend not available",
)
def test_goertzel_sparse_bins(benchmark: pytest.BenchmarkFixture) -> None:
    sample_rate = 120.0
    frequency = 2.25
    control = _sine_wave(frequency, sample_rate)
    response = _sine_wave(frequency, sample_rate, phase=math.pi / 7.0)
    candidate_bins = (frequency * 0.8, frequency, frequency * 1.2)

    def run_goertzel() -> spectrum.PhaseCorrelation | None:
        return spectrum._resolve_dominant_correlation(
            control,
            response,
            sample_rate,
            dominant_only=True,
            method="goertzel",
            candidate_frequencies=candidate_bins,
        )

    correlation = benchmark(run_goertzel)
    assert correlation is not None
    assert correlation.frequency == pytest.approx(frequency, rel=0.05)
