"""Modal resonance analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .epi import TelemetryRecord
from .spectrum import detrend, estimate_sample_rate, power_spectrum


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

    sample_rate = estimate_sample_rate(records)
    axis_series = _extract_axis_series(records)
    analysis: Dict[str, ModalAnalysis] = {}
    for axis, values in axis_series.items():
        detrended = detrend(values)
        total_energy = sum(value * value for value in detrended)
        spectrum = power_spectrum(detrended, sample_rate)
        peaks = _extract_peaks(spectrum, max_peaks=max_peaks)
        analysis[axis] = ModalAnalysis(
            sample_rate=float(sample_rate),
            total_energy=float(total_energy),
            peaks=peaks,
        )
    return analysis

