from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pytest

from tnfr_core import spectrum


def _sine_wave(
    frequency: float,
    sample_rate: float,
    *,
    length: int = 512,
    phase: float = 0.0,
    amplitude: float = 1.0,
) -> Sequence[float]:
    times = np.arange(length, dtype=float) / sample_rate
    return (amplitude * np.sin((2.0 * math.pi * frequency * times) + phase)).tolist()


def test_resolve_dominant_correlation_fft_path() -> None:
    sample_rate = 50.0
    frequency = 3.0
    control = _sine_wave(frequency, sample_rate, length=512)
    response = _sine_wave(frequency, sample_rate, length=512, phase=math.pi / 6.0)

    correlation = spectrum._resolve_dominant_correlation(
        control,
        response,
        sample_rate,
        dominant_only=True,
        method="fft",
    )

    assert correlation is not None
    assert correlation.frequency == pytest.approx(frequency, rel=0.05)
    assert abs(correlation.phase) == pytest.approx(math.pi / 6.0, rel=0.2)


def test_resolve_dominant_correlation_goertzel_path(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_rate = 120.0
    frequency = 2.5
    control = _sine_wave(frequency, sample_rate, length=600)
    response = _sine_wave(frequency, sample_rate, length=600, phase=-math.pi / 4.0)
    candidate_bins = [frequency, frequency * 1.6]

    call_counter = {"count": 0}

    def fake_goertzel(values: Sequence[float], angular_frequency: float) -> complex:
        call_counter["count"] += 1
        values_array = np.asarray(values, dtype=float)
        samples = np.arange(values_array.size, dtype=float)
        exponent = np.exp(-1j * angular_frequency * samples)
        return np.sum(values_array * exponent)

    monkeypatch.setattr(spectrum, "_SCIPY_GOERTZEL", fake_goertzel)

    correlation = spectrum._resolve_dominant_correlation(
        control,
        response,
        sample_rate,
        dominant_only=True,
        method="goertzel",
        candidate_frequencies=candidate_bins,
    )

    assert correlation is not None
    assert correlation.frequency == pytest.approx(frequency, rel=0.05)
    assert abs(correlation.phase) == pytest.approx(abs(-math.pi / 4.0), rel=0.2)
    assert call_counter["count"] == len(candidate_bins) * 2


def test_goertzel_strategy_requires_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_rate = 100.0
    frequency = 1.5
    control = _sine_wave(frequency, sample_rate)
    response = _sine_wave(frequency, sample_rate, phase=math.pi / 3.0)

    monkeypatch.setattr(spectrum, "_SCIPY_GOERTZEL", None)

    with pytest.raises(RuntimeError):
        spectrum._resolve_dominant_correlation(
            control,
            response,
            sample_rate,
            dominant_only=True,
            method="goertzel",
            candidate_frequencies=[frequency],
        )


def test_goertzel_auto_falls_back_to_fft(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_rate = 90.0
    frequency = 4.0
    control = _sine_wave(frequency, sample_rate, length=400)
    response = _sine_wave(frequency, sample_rate, length=400, phase=math.pi / 5.0)

    monkeypatch.setattr(spectrum, "_SCIPY_GOERTZEL", None)

    correlation = spectrum._resolve_dominant_correlation(
        control,
        response,
        sample_rate,
        dominant_only=True,
        method="auto",
        candidate_frequencies=[frequency],
    )

    assert correlation is not None
    assert correlation.frequency == pytest.approx(frequency, rel=0.05)
    assert abs(correlation.phase) == pytest.approx(math.pi / 5.0, rel=0.2)
