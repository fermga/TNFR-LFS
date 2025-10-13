"""Modal resonance analysis helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from tnfr_core.operators.interfaces import SupportsTelemetrySample
from tnfr_core.metrics.spectrum import detrend, estimate_sample_rate, power_spectrum

__all__ = ["ModalPeak", "ModalAnalysis", "analyse_modal_resonance"]


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
    nu_exc: float
    rho: float


AxisSeries = Dict[str, List[float]]


def _extract_axis_series(records: Sequence[SupportsTelemetrySample]) -> AxisSeries:
    series: AxisSeries = {"yaw": [], "roll": [], "pitch": []}
    for record in records:
        series["yaw"].append(float(record.yaw))
        series["roll"].append(float(record.roll))
        series["pitch"].append(float(record.pitch))
    return series


def _normalise(values: Sequence[float]) -> List[float]:
    detrended = detrend(values)
    if not detrended:
        return []
    variance = sum(value * value for value in detrended) / len(detrended)
    if variance <= 1e-12:
        return [0.0] * len(detrended)
    scale = math.sqrt(variance)
    return [value / scale for value in detrended]


def _excitation_series(records: Sequence[SupportsTelemetrySample]) -> List[float]:
    steer = _normalise([float(record.steer) for record in records])
    front = _normalise(
        [float(record.suspension_velocity_front) for record in records]
    )
    rear = _normalise(
        [float(record.suspension_velocity_rear) for record in records]
    )
    length = max(len(steer), len(front), len(rear))
    if length == 0:
        return []
    series: List[float] = []
    for index in range(length):
        steer_value = steer[index] if index < len(steer) else 0.0
        front_value = front[index] if index < len(front) else 0.0
        rear_value = rear[index] if index < len(rear) else 0.0
        combined = 0.5 * steer_value + 0.25 * front_value + 0.25 * rear_value
        series.append(combined)
    return series


def estimate_excitation_frequency(
    records: Sequence[SupportsTelemetrySample], sample_rate: float | None = None
) -> float:
    """Return the dominant excitation frequency from :class:`SupportsTelemetrySample` data."""

    if not records:
        return 0.0
    if sample_rate is None:
        sample_rate = estimate_sample_rate(records)
    if sample_rate <= 0.0:
        return 0.0
    excitation = _excitation_series(records)
    if len(excitation) < 2:
        return 0.0
    spectrum = power_spectrum(excitation, sample_rate)
    if not spectrum:
        return 0.0
    frequency, energy = max(spectrum, key=lambda item: item[1])
    if energy <= 0.0:
        return 0.0
    return float(frequency)
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
    records: Sequence[SupportsTelemetrySample],
    *,
    max_peaks: int = 3,
) -> Dict[str, ModalAnalysis]:
    """Compute modal energy for yaw/roll/pitch axes from telemetry-like sequences."""

    sample_rate = estimate_sample_rate(records)
    axis_series = _extract_axis_series(records)
    nu_exc = estimate_excitation_frequency(records, sample_rate)
    analysis: Dict[str, ModalAnalysis] = {}
    for axis, values in axis_series.items():
        detrended = detrend(values)
        total_energy = sum(value * value for value in detrended)
        spectrum = power_spectrum(detrended, sample_rate)
        peaks = _extract_peaks(spectrum, max_peaks=max_peaks)
        dominant_frequency = peaks[0].frequency if peaks else 0.0
        rho = nu_exc / dominant_frequency if dominant_frequency > 1e-9 else 0.0
        analysis[axis] = ModalAnalysis(
            sample_rate=float(sample_rate),
            total_energy=float(total_energy),
            peaks=peaks,
            nu_exc=float(nu_exc),
            rho=float(rho),
        )
    return analysis

