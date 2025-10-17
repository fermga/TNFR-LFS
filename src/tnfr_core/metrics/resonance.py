"""Modal resonance analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from tnfr_core.metrics.spectrum import (
    detrend,
    estimate_sample_rate,
    power_spectrum,
)
from tnfr_core.operators._shared import _HAS_JAX, jnp
from tnfr_core.operators.interfaces import SupportsTelemetrySample

xp = jnp if _HAS_JAX and jnp is not None else np

__all__ = ["ModalPeak", "ModalAnalysis", "analyse_modal_resonance"]


def _xp_length(values: Sequence[float]) -> int:
    size = getattr(values, "size", None)
    if size is not None:
        return int(size)
    shape = getattr(values, "shape", ())
    if shape:
        return int(shape[0])
    return len(values)


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


AxisSeries = Dict[str, Sequence[float]]


def _extract_axis_series(records: Sequence[SupportsTelemetrySample]) -> AxisSeries:
    if not records:
        empty = xp.asarray([], dtype=float)
        return {"yaw": empty, "roll": empty, "pitch": empty}

    matrix = xp.asarray(
        [
            [float(record.yaw), float(record.roll), float(record.pitch)]
            for record in records
        ],
        dtype=float,
    )
    yaw, roll, pitch = matrix.T
    return {"yaw": yaw, "roll": roll, "pitch": pitch}


def _normalise(values: Sequence[float]) -> Sequence[float]:
    detrended = detrend(xp.asarray(values, dtype=float), xp_module=xp)
    if _xp_length(detrended) == 0:
        return detrended
    variance = xp.mean(detrended ** 2)
    if float(variance) <= 1e-12:
        return xp.zeros_like(detrended)
    scale = xp.sqrt(variance)
    return detrended / scale


def _excitation_series(records: Sequence[SupportsTelemetrySample]) -> Sequence[float]:
    steer = _normalise([float(record.steer) for record in records])
    front = _normalise(
        [float(record.suspension_velocity_front) for record in records]
    )
    rear = _normalise(
        [float(record.suspension_velocity_rear) for record in records]
    )
    length = max(_xp_length(steer), _xp_length(front), _xp_length(rear))
    if length == 0:
        return xp.asarray([], dtype=float)

    def _pad(values: Sequence[float]) -> Sequence[float]:
        pad_width = length - _xp_length(values)
        if pad_width <= 0:
            return xp.asarray(values, dtype=float)
        return xp.pad(xp.asarray(values, dtype=float), (0, pad_width), mode="constant")

    steer_aligned = _pad(steer)
    front_aligned = _pad(front)
    rear_aligned = _pad(rear)

    combined = xp.stack([steer_aligned, front_aligned, rear_aligned], axis=0)
    weights = xp.asarray([0.5, 0.25, 0.25], dtype=float)
    return weights @ combined


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
    excitation_length = _xp_length(excitation)
    if excitation_length < 2:
        return 0.0
    spectrum = power_spectrum(excitation, sample_rate, xp_module=xp)
    if hasattr(spectrum, "shape") and getattr(spectrum, "size", 0) == 0:
        return 0.0
    if not hasattr(spectrum, "shape"):
        spectrum = xp.asarray(list(spectrum), dtype=float)
    if getattr(spectrum, "size", 0) == 0:
        return 0.0
    energies = spectrum[:, 1]
    max_index = int(xp.argmax(energies))
    frequency = float(spectrum[max_index, 0])
    energy = float(energies[max_index])
    if energy <= 0.0:
        return 0.0
    return float(frequency)


def _extract_peaks(
    spectrum: Iterable[tuple[float, float]],
    max_peaks: int = 3,
) -> List[ModalPeak]:
    if hasattr(spectrum, "shape") and not isinstance(spectrum, (list, tuple)):
        spectrum_array = spectrum
    else:
        spectrum_list = list(spectrum)
        if not spectrum_list:
            return []
        spectrum_array = xp.asarray(spectrum_list, dtype=float)

    dtype = getattr(spectrum_array, "dtype", None)
    if dtype is not None and np.dtype(dtype) != np.dtype(float):
        spectrum_array = spectrum_array.astype(float)
    if max_peaks <= 0 or spectrum_array.size == 0:
        return []

    frequencies = spectrum_array[:, 0]
    energies = spectrum_array[:, 1]
    count = int(frequencies.shape[0])
    if count == 0:
        return []

    num_peaks = min(max_peaks, count)

    if num_peaks == count:
        candidate_indices = xp.arange(count)
    else:
        if hasattr(xp, "argpartition"):
            partition = xp.argpartition(-energies, num_peaks - 1)
            candidate_indices = partition[:num_peaks]
        else:
            candidate_indices = xp.argsort(-energies)[:num_peaks]

    candidate_indices = xp.asarray(candidate_indices)
    candidate_energies = xp.take(energies, candidate_indices)
    ordered_positions = xp.argsort(-candidate_energies)
    sorted_indices = xp.take(candidate_indices, ordered_positions)

    top_frequencies = xp.take(frequencies, sorted_indices)[:num_peaks]
    top_energies = xp.take(energies, sorted_indices)[:num_peaks]

    dominant_energy = float(top_energies[0]) if num_peaks else 0.0

    freq_band_mask = xp.logical_and(top_frequencies >= 0.05, top_frequencies <= 5.0)
    first_peak_mask = xp.arange(num_peaks) == 0
    energy_ratio_mask = xp.logical_and(
        top_energies >= dominant_energy * 0.5, freq_band_mask
    )
    useful_mask = xp.logical_or(
        xp.logical_and(freq_band_mask, first_peak_mask), energy_ratio_mask
    )
    if dominant_energy <= 0.0:
        useful_mask = xp.zeros_like(useful_mask, dtype=bool)

    frequencies_np = np.asarray(top_frequencies, dtype=float)
    energies_np = np.asarray(top_energies, dtype=float)
    useful_np = np.asarray(useful_mask, dtype=bool)

    results: List[ModalPeak] = []
    for frequency, energy, useful in zip(frequencies_np, energies_np, useful_np):
        classification = "useful" if bool(useful) else "parasitic"
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
        detrended = detrend(xp.asarray(values, dtype=float), xp_module=xp)
        total_energy = float(xp.sum(detrended ** 2))
        spectrum = power_spectrum(detrended, sample_rate, xp_module=xp)
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

