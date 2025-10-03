"""Modal resonance analysis helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from .epi import TelemetryRecord


@dataclass(frozen=True)
class ModalPeak:
    """Single resonant peak extracted from the spectral analysis."""

    frequency: float
    energy: float
    classification: str


@dataclass(frozen=True)
class ModalAnalysis:
    """Aggregated spectral information for a single rotational axis."""

    sample_rate: float
    total_energy: float
    peaks: List[ModalPeak]


AxisSeries = Dict[str, List[float]]


def _extract_axis_series(records: Sequence[TelemetryRecord]) -> AxisSeries:
    series: AxisSeries = {"yaw": [], "roll": [], "pitch": []}
    for record in records:
        series["yaw"].append(float(record.yaw))
        series["roll"].append(float(record.roll))
        series["pitch"].append(float(record.pitch))
    return series


def _estimate_sample_rate(records: Sequence[TelemetryRecord]) -> float:
    deltas: List[float] = []
    previous: float | None = None
    for record in records:
        timestamp = float(record.timestamp)
        if previous is not None:
            delta = timestamp - previous
            if delta > 1e-9 and math.isfinite(delta):
                deltas.append(delta)
        previous = timestamp
    if not deltas:
        return 0.0
    average_delta = sum(deltas) / len(deltas)
    if average_delta <= 0.0:
        return 0.0
    return 1.0 / average_delta


def _hann_window(length: int) -> List[float]:
    if length <= 1:
        return [1.0] * length
    factor = 2.0 * math.pi / (length - 1)
    return [0.5 - 0.5 * math.cos(factor * idx) for idx in range(length)]


def _apply_window(samples: Sequence[float], window: Sequence[float]) -> List[float]:
    return [value * window[idx] for idx, value in enumerate(samples)]


def _detrend(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    centre = mean(values)
    return [value - centre for value in values]


def _discrete_spectrum(samples: Sequence[float], sample_rate: float) -> List[tuple[float, float]]:
    length = len(samples)
    if length < 2 or sample_rate <= 0.0:
        return []

    window = _hann_window(length)
    windowed = _apply_window(samples, window)

    spectrum: List[tuple[float, float]] = []
    # Only positive frequencies are relevant for modal analysis.
    upper = length // 2
    for index in range(1, upper + 1):
        real = 0.0
        imag = 0.0
        angle_factor = -2.0 * math.pi * index / length
        for sample_index, value in enumerate(windowed):
            angle = angle_factor * sample_index
            real += value * math.cos(angle)
            imag += value * math.sin(angle)
        amplitude = math.hypot(real, imag)
        # Normalise by the window energy to make energy comparisons consistent.
        energy = (amplitude ** 2) / length
        frequency = index * sample_rate / length
        spectrum.append((frequency, energy))
    return spectrum


def _extract_peaks(
    spectrum: Iterable[tuple[float, float]],
    max_peaks: int = 3,
) -> List[ModalPeak]:
    peaks = sorted(spectrum, key=lambda item: item[1], reverse=True)[:max_peaks]
    if not peaks:
        return []
    dominant_energy = peaks[0][1]
    results: List[ModalPeak] = []
    for idx, (frequency, energy) in enumerate(peaks):
        if dominant_energy <= 0.0:
            classification = "parasitic"
        elif idx == 0 and 0.05 <= frequency <= 5.0:
            classification = "useful"
        elif 0.05 <= frequency <= 5.0 and energy >= dominant_energy * 0.5:
            classification = "useful"
        else:
            classification = "parasitic"
        results.append(
            ModalPeak(
                frequency=float(frequency),
                energy=float(energy),
                classification=classification,
            )
        )
    return results


def analyse_modal_resonance(
    records: Sequence[TelemetryRecord],
    *,
    max_peaks: int = 3,
) -> Dict[str, ModalAnalysis]:
    """Compute modal energy for yaw/roll/pitch axes."""

    sample_rate = _estimate_sample_rate(records)
    axis_series = _extract_axis_series(records)
    analysis: Dict[str, ModalAnalysis] = {}
    for axis, values in axis_series.items():
        detrended = _detrend(values)
        total_energy = sum(value * value for value in detrended)
        spectrum = _discrete_spectrum(detrended, sample_rate)
        peaks = _extract_peaks(spectrum, max_peaks=max_peaks)
        analysis[axis] = ModalAnalysis(
            sample_rate=float(sample_rate),
            total_energy=float(total_energy),
            peaks=peaks,
        )
    return analysis

