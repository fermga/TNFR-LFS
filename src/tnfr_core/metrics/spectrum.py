"""Spectral helpers shared across modal and phase alignment analyses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from typing import Literal

import numpy as np

from tnfr_core.operators._shared import _HAS_JAX, jnp
from tnfr_core.operators.interfaces import SupportsTelemetrySample

try:  # pragma: no cover - SciPy is optional
    from scipy.signal import goertzel as _SCIPY_GOERTZEL
except Exception:  # pragma: no cover - SciPy is optional
    _SCIPY_GOERTZEL = None

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

if _HAS_JAX and jnp is not None:  # pragma: no cover - exercised only with JAX
    try:  # pragma: no cover - exercised only with JAX 0.4+
        from jax import Array as _JaxArray  # type: ignore[import]

        _JAX_ARRAY_TYPES: Tuple[type, ...] = (_JaxArray,)
    except Exception:  # pragma: no cover - exercised with legacy JAX
        try:
            from jax.interpreters.xla import DeviceArray as _DeviceArray  # type: ignore[import-not-found]

            _JAX_ARRAY_TYPES = (_DeviceArray,)
        except Exception:  # pragma: no cover - fallback when neither type is present
            _JAX_ARRAY_TYPES = ()
else:  # pragma: no cover - exercised when JAX is unavailable
    _JAX_ARRAY_TYPES = ()


def _has_jax_array(value: Any) -> bool:
    if not _JAX_ARRAY_TYPES:
        return False
    if isinstance(value, _JAX_ARRAY_TYPES):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_jax_array(item) for item in value)
    return False


def _resolve_backend(xp_module: Any | None, *values: Any) -> Any:
    if xp_module is not None:
        return xp_module
    if _has_jax_array(values):
        return jnp  # type: ignore[return-value]
    return np


def _xp_array(values: Any, xp_module: Any, dtype: Any = float) -> Any:
    return xp_module.asarray(values, dtype=dtype)


def _xp_size(values: Any) -> int:
    if hasattr(values, "size"):
        return int(getattr(values, "size"))
    if hasattr(values, "shape"):
        shape = getattr(values, "shape")
        if shape:
            return int(shape[0])
        return 0
    return len(values)


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


_DEFAULT_GOERTZEL_CANDIDATES: Tuple[float, ...] = (
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    5.0,
    6.5,
    8.0,
    10.0,
)


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
    *,
    dominant_only: bool = False,
    method: Literal["auto", "fft", "goertzel"] = "auto",
    candidate_frequencies: Sequence[float] | None = None,
) -> PhaseCorrelation | None:
    length = min(len(input_series), len(response_series))
    if length < 2 or sample_rate <= 0.0:
        return None

    input_values = list(_normalise(input_series))[-length:]
    response_values = list(_normalise(response_series))[-length:]
    if dominant_only and method in ("auto", "goertzel"):
        goertzel_candidates = (
            tuple(candidate_frequencies)
            if candidate_frequencies is not None
            else _DEFAULT_GOERTZEL_CANDIDATES
        )
        correlation = _dominant_frequency_goertzel(
            input_values,
            response_values,
            sample_rate,
            goertzel_candidates,
            require_backend=method == "goertzel",
        )
        if correlation is not None:
            return correlation
        if method == "goertzel":
            return None
    spectrum = cross_spectrum(input_values, response_values, sample_rate)
    if not spectrum:
        return None

    dominant_frequency = 0.0
    dominant_phase = 0.0
    dominant_magnitude = 0.0
    for frequency, cross_real, cross_imag in spectrum:
        if frequency <= 1e-9:
            continue
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


def _dominant_frequency_goertzel(
    input_series: Sequence[float],
    response_series: Sequence[float],
    sample_rate: float,
    candidate_frequencies: Sequence[float],
    *,
    require_backend: bool = False,
) -> PhaseCorrelation | None:
    """Return the dominant correlation using Goertzel on fixed frequency bins.

    Evaluating ``k`` candidate bins via Goertzel costs :math:`O(n·k)`; with the
    default constant ``k`` the helper runs in :math:`O(n)` with respect to the
    number of samples.
    """

    if _SCIPY_GOERTZEL is None:
        if require_backend:
            raise RuntimeError(
                "SciPy is required for the Goertzel-based dominant frequency path"
            )
        return None
    if sample_rate <= 0.0:
        return None

    valid_bins = [
        float(freq)
        for freq in candidate_frequencies
        if freq > 1e-9 and freq <= (sample_rate * 0.5) + 1e-9
    ]
    if not valid_bins:
        return None

    input_values = np.asarray(input_series, dtype=float)
    response_values = np.asarray(response_series, dtype=float)
    length = min(input_values.size, response_values.size)
    if length < 2:
        return None

    input_values = np.asarray(detrend(input_values)[-length:], dtype=float)
    response_values = np.asarray(detrend(response_values)[-length:], dtype=float)
    window = np.asarray(hann_window(length), dtype=float)
    input_windowed = np.asarray(apply_window(input_values, window), dtype=float)
    response_windowed = np.asarray(apply_window(response_values, window), dtype=float)

    dominant_frequency = 0.0
    dominant_cross = 0.0 + 0.0j
    dominant_magnitude = 0.0

    for frequency in valid_bins:
        angular = 2.0 * math.pi * (frequency / sample_rate)
        input_component = _SCIPY_GOERTZEL(input_windowed, angular)
        response_component = _SCIPY_GOERTZEL(response_windowed, angular)
        cross_component = input_component * np.conj(response_component)
        magnitude = abs(cross_component)
        if magnitude > dominant_magnitude:
            dominant_frequency = float(frequency)
            dominant_cross = cross_component
            dominant_magnitude = float(magnitude)

    if dominant_frequency <= 1e-9:
        return None

    dominant_phase = math.atan2(dominant_cross.imag, dominant_cross.real)
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
    records: Sequence[SupportsTelemetrySample],
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
    records: Sequence[SupportsTelemetrySample],
    *,
    controls: Mapping[str, Sequence[str]] | None = None,
    responses: Mapping[str, Sequence[str]] | None = None,
    sample_rate: float | None = None,
    dominant_strategy: Literal["auto", "fft", "goertzel"] = "auto",
    candidate_frequencies: Sequence[float] | None = None,
) -> Dict[Tuple[str, str], PhaseCorrelation]:
    """Return dominant correlations between driver inputs and chassis responses.

    ``dominant_strategy`` selects whether to evaluate the dense FFT
    cross-spectrum (``"fft"``) or the sparse Goertzel helper when SciPy is
    installed (``"auto"``/``"goertzel"``). ``candidate_frequencies`` constrains
    the Goertzel search to custom bins when provided.
    """

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
                control_series,
                response_series,
                sample_rate,
                dominant_only=True,
                method=dominant_strategy,
                candidate_frequencies=candidate_frequencies,
            )
            if correlation is None:
                continue
            results[(str(control_label), str(response_label))] = correlation
    return results


def estimate_sample_rate(records: Sequence[SupportsTelemetrySample]) -> float:
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


def hann_window(length: int, *, xp_module: Any | None = None) -> Any:
    """Return a Hann window matching ``length`` samples using ``xp_module``."""

    xp_backend = _resolve_backend(xp_module)
    if length <= 1:
        return xp_backend.ones(length, dtype=float)
    indices = xp_backend.arange(length, dtype=float)
    factor = 2.0 * math.pi / (length - 1)
    return 0.5 - 0.5 * xp_backend.cos(indices * factor)


def apply_window(
    samples: Sequence[float],
    window: Sequence[float],
    *,
    xp_module: Any | None = None,
) -> Any:
    """Multiply ``samples`` by ``window`` element wise using ``xp_module``."""

    xp_backend = _resolve_backend(xp_module, samples, window)
    sample_array = _xp_array(samples, xp_backend, dtype=float)
    window_array = _xp_array(window, xp_backend, dtype=float)
    return sample_array * window_array


def detrend(values: Sequence[float], *, xp_module: Any | None = None) -> Any:
    """Remove the arithmetic mean from ``values`` using ``xp_module``."""

    xp_backend = _resolve_backend(xp_module, values)
    array = _xp_array(values, xp_backend, dtype=float)
    if _xp_size(array) == 0:
        return array
    centre = xp_backend.mean(array)
    return array - centre


def _fourier_components(
    samples: Sequence[float],
    sample_rate: float,
    *,
    xp_module: Any | None = None,
) -> Tuple[Any, Any]:
    xp_backend = _resolve_backend(xp_module, samples)
    sample_values = _xp_array(samples, xp_backend, dtype=float)
    length = _xp_size(sample_values)
    if length < 2 or sample_rate <= 0.0:
        return (
            xp_backend.asarray([], dtype=float),
            xp_backend.asarray([], dtype=complex),
        )

    detrended = detrend(sample_values, xp_module=xp_backend)
    window = hann_window(length, xp_module=xp_backend)
    windowed = apply_window(detrended, window, xp_module=xp_backend)

    fft_values = xp_backend.fft.rfft(windowed)
    frequencies = xp_backend.fft.rfftfreq(length, d=1.0 / sample_rate)

    # Skip the DC component to preserve the historical single-sided behaviour.
    return frequencies[1:], fft_values[1:]


PowerSpectrumResult = Sequence[Tuple[float, float]] | Any
CrossSpectrumResult = Sequence[Tuple[float, float, float]] | Any


def power_spectrum(
    samples: Sequence[float],
    sample_rate: float,
    *,
    xp_module: Any | None = None,
) -> PowerSpectrumResult:
    """Return the single-sided power spectrum of ``samples`` using ``xp_module``."""

    xp_backend = _resolve_backend(xp_module, samples)
    frequencies, fft_values = _fourier_components(
        samples, sample_rate, xp_module=xp_backend
    )
    sample_values = _xp_array(samples, xp_backend, dtype=float)
    length = _xp_size(sample_values)
    if length == 0 or _xp_size(frequencies) == 0:
        if xp_module is None:
            return []
        return xp_backend.asarray([], dtype=float).reshape(0, 2)

    magnitudes = xp_backend.abs(fft_values)
    energy = xp_backend.square(magnitudes) / float(length)

    if xp_module is None:
        freq_np = np.asarray(frequencies, dtype=float)
        energy_np = np.asarray(energy, dtype=float)
        return list(zip(freq_np.tolist(), energy_np.tolist()))

    frequencies = xp_backend.asarray(frequencies, dtype=float)
    energy = xp_backend.asarray(energy, dtype=float)
    if _xp_size(frequencies) == 0:
        return xp_backend.asarray([], dtype=float).reshape(0, 2)
    return xp_backend.stack((frequencies, energy), axis=-1)


def cross_spectrum(
    input_series: Sequence[float],
    response_series: Sequence[float],
    sample_rate: float,
    *,
    xp_module: Any | None = None,
) -> CrossSpectrumResult:
    """Return the cross-spectrum between ``input_series`` and ``response_series`` using ``xp_module``."""

    xp_backend = _resolve_backend(xp_module, input_series, response_series)
    input_values = _xp_array(input_series, xp_backend, dtype=float)
    response_values = _xp_array(response_series, xp_backend, dtype=float)
    length = min(_xp_size(input_values), _xp_size(response_values))
    if length < 2 or sample_rate <= 0.0:
        if xp_module is None:
            return []
        return xp_backend.asarray([], dtype=float).reshape(0, 3)

    input_values = detrend(input_values, xp_module=xp_backend)[-length:]
    response_values = detrend(response_values, xp_module=xp_backend)[-length:]
    window = hann_window(length, xp_module=xp_backend)
    input_windowed = apply_window(input_values, window, xp_module=xp_backend)
    response_windowed = apply_window(response_values, window, xp_module=xp_backend)

    input_fft = xp_backend.fft.rfft(input_windowed)
    response_fft = xp_backend.fft.rfft(response_windowed)
    cross_values = input_fft * xp_backend.conj(response_fft)
    frequencies = xp_backend.fft.rfftfreq(length, d=1.0 / sample_rate)

    if xp_module is None:
        freq_np = np.asarray(frequencies, dtype=float)
        cross_np = np.asarray(cross_values, dtype=complex)

        spectrum: List[Tuple[float, float, float]] = []
        for frequency, value in zip(freq_np, cross_np):
            if frequency <= 1e-9:
                continue
            spectrum.append((float(frequency), float(value.real), float(value.imag)))
        return spectrum

    frequencies = xp_backend.asarray(frequencies, dtype=float)
    cross_values = xp_backend.asarray(cross_values, dtype=complex)

    valid_mask = frequencies > 1e-9
    if xp_backend.any(valid_mask):
        frequencies = frequencies[valid_mask]
        cross_values = cross_values[valid_mask]
    else:
        return xp_backend.asarray([], dtype=float).reshape(0, 3)

    cross_real = xp_backend.real(cross_values)
    cross_imag = xp_backend.imag(cross_values)
    return xp_backend.stack((frequencies, cross_real, cross_imag), axis=-1)


def _normalise(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    xp_backend = _resolve_backend(None, values)
    xp_values = xp_backend.asarray(values, dtype=float)
    mean_value = xp_backend.mean(xp_values)
    centred = xp_values - mean_value
    variance = xp_backend.mean(centred * centred)
    if float(variance) <= 1e-12:
        return [0.0 for _ in values]
    scale = xp_backend.sqrt(variance)
    normalised = centred / scale
    if hasattr(normalised, "tolist"):
        return normalised.tolist()
    return [float(value) for value in normalised]


def phase_alignment(
    records: Sequence[SupportsTelemetrySample],
    *,
    steer_series: Iterable[float] | None = None,
    response_series: Iterable[float] | None = None,
    sample_rate: float | None = None,
    dominant_strategy: Literal["auto", "fft", "goertzel"] = "auto",
    candidate_frequencies: Sequence[float] | None = None,
    steer_norm: Iterable[float] | None = None,
    yaw_norm: Iterable[float] | None = None,
    lat_norm: Iterable[float] | None = None,
) -> Tuple[float, float, float]:
    """Estimate the dominant frequency and phase lag between steer and response.

    Parameters
    ----------
    records:
        Telemetry samples from which the sampling frequency is extracted.
    steer_series, response_series:
        Explicit sequences containing steer input and the combined chassis
        response.  When omitted they are derived from ``records``.
    sample_rate:
        Optional sampling frequency to reuse when available.
    dominant_strategy:
        ``"auto"`` uses Goertzel when SciPy is available and falls back to the
        FFT cross-spectrum otherwise. ``"goertzel"`` enforces the Goertzel path
        and raises when SciPy is unavailable, while ``"fft"`` always evaluates
        the dense FFT grid.
    candidate_frequencies:
        Optional override for the candidate bins evaluated by the Goertzel path.
    steer_norm, yaw_norm, lat_norm:
        Optional pre-normalised steer, yaw rate and lateral acceleration series.
        When provided the function skips recomputing the z-scores for the same
        data.
    """

    if steer_series is None:
        if steer_norm is not None:
            steer_values = list(steer_norm)
        else:
            steer_values = [float(record.steer) for record in records]
            steer_values = _normalise(steer_values)
    else:
        steer_values = list(steer_series)

    if response_series is None:
        if yaw_norm is not None and lat_norm is not None:
            yaw_components = list(yaw_norm)
            lat_components = list(lat_norm)
        else:
            yaw_values = [float(record.yaw_rate) for record in records]
            lat_values = [float(record.lateral_accel) for record in records]
            yaw_components = _normalise(yaw_values)
            lat_components = _normalise(lat_values)

        xp_backend = _resolve_backend(
            None, steer_values, yaw_components, lat_components
        )
        steer_array = _xp_array(steer_values, xp_backend, dtype=float)
        yaw_array = _xp_array(yaw_components, xp_backend, dtype=float)
        lat_array = _xp_array(lat_components, xp_backend, dtype=float)

        combined_length = max(
            _xp_size(steer_array),
            _xp_size(yaw_array),
            _xp_size(lat_array),
        )

        if combined_length == 0:
            combined_array = xp_backend.asarray([], dtype=float)
        else:
            if _xp_size(yaw_array) < combined_length:
                yaw_array = xp_backend.pad(
                    yaw_array,
                    ((0, combined_length - _xp_size(yaw_array)),),
                )
            if _xp_size(lat_array) < combined_length:
                lat_array = xp_backend.pad(
                    lat_array,
                    ((0, combined_length - _xp_size(lat_array)),),
                )
            if _xp_size(yaw_array) > combined_length:
                yaw_array = yaw_array[:combined_length]
            if _xp_size(lat_array) > combined_length:
                lat_array = lat_array[:combined_length]
            combined_array = (yaw_array + lat_array) * 0.5

        combined_response = (
            combined_array.tolist()
            if hasattr(combined_array, "tolist")
            else [float(value) for value in combined_array]
        )
        steer_values = (
            steer_array.tolist()
            if hasattr(steer_array, "tolist")
            else [float(value) for value in steer_array]
        )
    else:
        combined_response = list(response_series)

    length = min(len(steer_values), len(combined_response))
    if length < 2:
        return (0.0, 0.0, 1.0)

    if sample_rate is None:
        sample_rate = estimate_sample_rate(records)
    if sample_rate <= 0.0:
        return (0.0, 0.0, 1.0)

    steer_trimmed = steer_values[-length:]
    response_trimmed = combined_response[-length:]
    correlation = _resolve_dominant_correlation(
        steer_trimmed,
        response_trimmed,
        sample_rate,
        dominant_only=True,
        method=dominant_strategy,
        candidate_frequencies=candidate_frequencies,
    )
    if correlation is None:
        return (0.0, 0.0, 1.0)
    return correlation.frequency, correlation.phase, correlation.alignment

