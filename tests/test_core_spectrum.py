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


def test_phase_alignment_accepts_pre_normalised_series(
    synthetic_records,
) -> None:
    records = list(synthetic_records)
    baseline_freq, baseline_lag, baseline_alignment = spectrum.phase_alignment(records)
    sample_rate = spectrum.estimate_sample_rate(records)

    def _extract(attr: str) -> list[float]:
        series: list[float] = []
        for record in records:
            try:
                value = float(getattr(record, attr))
            except (AttributeError, TypeError, ValueError):
                value = 0.0
            if not math.isfinite(value):
                value = 0.0
            series.append(value)
        return series

    def _normalise(values: list[float]) -> list[float]:
        if not values:
            return []
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        if variance <= 1e-12:
            return [0.0 for _ in values]
        scale = math.sqrt(variance)
        return [(value - mean_value) / scale for value in values]

    steer_norm = _normalise(_extract("steer"))
    yaw_norm = _normalise(_extract("yaw_rate"))
    lat_norm = _normalise(_extract("lateral_accel"))

    frequency, lag, alignment = spectrum.phase_alignment(
        records,
        sample_rate=sample_rate,
        steer_norm=steer_norm,
        yaw_norm=yaw_norm,
        lat_norm=lat_norm,
    )

    assert frequency == pytest.approx(baseline_freq, rel=1e-6, abs=1e-6)
    assert lag == pytest.approx(baseline_lag, rel=1e-6, abs=1e-6)
    assert alignment == pytest.approx(baseline_alignment, rel=1e-6, abs=1e-6)
