"""Spectral helpers shared across modal and phase alignment analyses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .epi import TelemetryRecord

__all__ = [
    "estimate_sample_rate",
    "hann_window",
    "apply_window",
    "detrend",
    "power_spectrum",
    "cross_spectrum",
    "phase_alignment",
    "PhaseCorrelation",
    "phase_to_latency_ms",
    "motor_input_correlations",
]


@dataclass(frozen=True)
class PhaseCorrelation:
    """Correlation payload describing the dominant cross-spectrum component."""

    frequency: float
    phase: float
    alignment: float
    latency_ms: float
    magnitude: float


_DEFAULT_CONTROL_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "steer": ("steer",),
    "brake": ("brake_pressure", "brake"),
    "throttle": ("throttle",),
}

_DEFAULT_RESPONSE_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "yaw": ("yaw_rate",),
    "ax": ("longitudinal_accel",),
    "ay": ("lateral_accel",),
}


def phase_to_latency_ms(frequency: float, phase: float) -> float:
    """Convert a phase lag expressed in radians to milliseconds."""

    if frequency <= 1e-9:
        return 0.0
    delay_seconds = phase / (2.0 * math.pi * frequency)
    return delay_seconds * 1000.0


def _resolve_dominant_correlation(
    input_series: Sequence[float],
    response_series: Sequence[float],
    sample_rate: float,
) -> PhaseCorrelation | None:
    length = min(len(input_series), len(response_series))
    if length < 2 or sample_rate <= 0.0:
        return None

    input_values = list(_normalise(input_series))[-length:]
    response_values = list(_normalise(response_series))[-length:]
    spectrum = cross_spectrum(input_values, response_values, sample_rate)
    if not spectrum:
        return None

    dominant_frequency = 0.0
    dominant_phase = 0.0
    dominant_magnitude = 0.0
    for frequency, cross_real, cross_imag in spectrum:
        magnitude = math.hypot(cross_real, cross_imag)
        if magnitude > dominant_magnitude:
            dominant_frequency = float(frequency)
            dominant_phase = math.atan2(cross_imag, cross_real)
            dominant_magnitude = magnitude

    alignment = math.cos(dominant_phase)
    latency = phase_to_latency_ms(dominant_frequency, dominant_phase)
    return PhaseCorrelation(
        frequency=dominant_frequency,
        phase=dominant_phase,
        alignment=alignment,
        latency_ms=latency,
        magnitude=max(0.0, dominant_magnitude),
    )


def _extract_series(
    records: Sequence[TelemetryRecord],
    attributes: Sequence[str],
) -> list[float]:
    samples: list[float] = []
    for record in records:
        value = 0.0
        resolved = False
        for attr in attributes:
            try:
                candidate = getattr(record, attr)
            except AttributeError:
                continue
            try:
                numeric = float(candidate)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                value = numeric
                resolved = True
                break
        if not resolved:
            value = 0.0
        samples.append(value)
    return samples


def _has_variation(samples: Sequence[float]) -> bool:
    if len(samples) < 2:
        return False
    min_value = min(samples)
    max_value = max(samples)
    return math.isfinite(min_value) and math.isfinite(max_value) and (max_value - min_value) > 1e-6


def motor_input_correlations(
    records: Sequence[TelemetryRecord],
    *,
    controls: Mapping[str, Sequence[str]] | None = None,
    responses: Mapping[str, Sequence[str]] | None = None,
    sample_rate: float | None = None,
) -> Dict[Tuple[str, str], PhaseCorrelation]:
    """Return dominant correlations between driver inputs and chassis responses."""

    if controls is None:
        controls = _DEFAULT_CONTROL_FIELDS
    if responses is None:
        responses = _DEFAULT_RESPONSE_FIELDS
    if sample_rate is None:
        sample_rate = estimate_sample_rate(records)
    if sample_rate <= 0.0:
        return {}

    results: Dict[Tuple[str, str], PhaseCorrelation] = {}
    for control_label, control_attrs in controls.items():
        control_series = _extract_series(records, control_attrs)
        if not _has_variation(control_series):
            continue
        for response_label, response_attrs in responses.items():
            response_series = _extract_series(records, response_attrs)
            if not _has_variation(response_series):
                continue
            correlation = _resolve_dominant_correlation(
                control_series, response_series, sample_rate
            )
            if correlation is None:
                continue
            results[(str(control_label), str(response_label))] = correlation
    return results


def estimate_sample_rate(records: Sequence[TelemetryRecord]) -> float:
    """Estimate the sampling frequency of ``records`` in Hertz."""

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


def hann_window(length: int) -> List[float]:
    """Return a Hann window matching ``length`` samples."""

    if length <= 1:
        return [1.0] * length
    factor = 2.0 * math.pi / (length - 1)
    return [0.5 - 0.5 * math.cos(factor * idx) for idx in range(length)]


def apply_window(
    samples: Sequence[float], window: Sequence[float]
) -> Sequence[float] | np.ndarray:
    """Multiply ``samples`` by ``window`` element wise."""

    if isinstance(samples, np.ndarray):
        return samples * np.asarray(window, dtype=float)
    if isinstance(window, np.ndarray):
        return np.asarray(samples, dtype=float) * window
    return [value * window[idx] for idx, value in enumerate(samples)]


def detrend(values: Sequence[float]) -> Sequence[float] | np.ndarray:
    """Remove the arithmetic mean from ``values``."""

    if isinstance(values, np.ndarray):
        if values.size == 0:
            return values
        centre = float(np.mean(values))
        return values - centre
    if not values:
        return []
    centre = mean(values)
    return [value - centre for value in values]


def _fourier_components(
    samples: Sequence[float], sample_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    sample_values = np.asarray(samples, dtype=float)
    length = int(sample_values.size)
    if length < 2 or sample_rate <= 0.0:
        return np.asarray([], dtype=float), np.asarray([], dtype=complex)

    detrended = np.asarray(detrend(sample_values), dtype=float)
    window = np.asarray(hann_window(length), dtype=float)
    windowed = np.asarray(apply_window(detrended, window), dtype=float)

    fft_values = np.fft.rfft(windowed)
    frequencies = np.fft.rfftfreq(length, d=1.0 / sample_rate)

    # Skip the DC component to preserve the historical single-sided behaviour.
    return frequencies[1:], fft_values[1:]


def power_spectrum(samples: Sequence[float], sample_rate: float) -> List[Tuple[float, float]]:
    """Return the single-sided power spectrum of ``samples``."""

    frequencies, fft_values = _fourier_components(samples, sample_rate)
    length = len(samples)
    if length == 0 or frequencies.size == 0:
        return []

    magnitudes = np.abs(fft_values)
    energy = np.square(magnitudes) / float(length)
    return list(zip(frequencies.tolist(), energy.tolist()))


def cross_spectrum(
    input_series: Sequence[float],
    response_series: Sequence[float],
    sample_rate: float,
) -> List[Tuple[float, float, float]]:
    """Return the cross-spectrum between ``input_series`` and ``response_series``."""

    input_values = np.asarray(input_series, dtype=float)
    response_values = np.asarray(response_series, dtype=float)
    length = min(len(input_values), len(response_values))
    if length < 2 or sample_rate <= 0.0:
        return []

    input_values = detrend(input_values)[-length:]
    response_values = detrend(response_values)[-length:]
    window = np.asarray(hann_window(length), dtype=float)
    input_windowed = np.asarray(apply_window(input_values, window), dtype=float)
    response_windowed = np.asarray(apply_window(response_values, window), dtype=float)

    spectrum: List[Tuple[float, float, float]] = []
    upper = length // 2
    for index in range(1, upper + 1):
        x_real = 0.0
        x_imag = 0.0
        y_real = 0.0
        y_imag = 0.0
        angle_factor = -2.0 * math.pi * index / length
        for sample_index in range(length):
            angle = angle_factor * sample_index
            cos_val = math.cos(angle)
            sin_val = math.sin(angle)
            value_x = input_windowed[sample_index]
            value_y = response_windowed[sample_index]
            x_real += value_x * cos_val
            x_imag += value_x * sin_val
            y_real += value_y * cos_val
            y_imag += value_y * sin_val
        cross_real = x_real * y_real + x_imag * y_imag
        cross_imag = x_imag * y_real - x_real * y_imag
        frequency = index * sample_rate / length
        spectrum.append((frequency, cross_real, cross_imag))
    return spectrum


def _normalise(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    if variance <= 1e-12:
        return [0.0 for _ in values]
    scale = math.sqrt(variance)
    return [(value - mean_value) / scale for value in values]


def phase_alignment(
    records: Sequence[TelemetryRecord],
    *,
    steer_series: Iterable[float] | None = None,
    response_series: Iterable[float] | None = None,
) -> Tuple[float, float, float]:
    """Estimate the dominant frequency and phase lag between steer and response.

    Parameters
    ----------
    records:
        Telemetry samples from which the sampling frequency is extracted.
    steer_series, response_series:
        Explicit sequences containing steer input and the combined chassis
        response.  When omitted they are derived from ``records``.
    """

    if steer_series is None or response_series is None:
        steer_values = [float(record.steer) for record in records]
        yaw_values = [float(record.yaw_rate) for record in records]
        lat_values = [float(record.lateral_accel) for record in records]
        combined_response = []
        yaw_norm = _normalise(yaw_values)
        lat_norm = _normalise(lat_values)
        for idx in range(len(records)):
            yaw_component = yaw_norm[idx] if idx < len(yaw_norm) else 0.0
            lat_component = lat_norm[idx] if idx < len(lat_norm) else 0.0
            combined_response.append((yaw_component + lat_component) * 0.5)
        steer_values = _normalise(steer_values)
    else:
        steer_values = list(steer_series)
        combined_response = list(response_series)

    length = min(len(steer_values), len(combined_response))
    if length < 2:
        return (0.0, 0.0, 1.0)

    sample_rate = estimate_sample_rate(records)
    if sample_rate <= 0.0:
        return (0.0, 0.0, 1.0)

    steer_trimmed = steer_values[-length:]
    response_trimmed = combined_response[-length:]
    correlation = _resolve_dominant_correlation(steer_trimmed, response_trimmed, sample_rate)
    if correlation is None:
        return (0.0, 0.0, 1.0)
    return correlation.frequency, correlation.phase, correlation.alignment

