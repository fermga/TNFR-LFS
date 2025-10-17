"""Generic spectral signal processing helpers."""

from __future__ import annotations

import math
from typing import Any, List, Sequence, Tuple

import numpy as np

from tnfr_core.runtime.shared import SupportsTelemetrySample, _HAS_JAX, jnp

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


__all__ = [
    "estimate_sample_rate",
    "hann_window",
    "apply_window",
    "detrend",
    "power_spectrum",
    "cross_spectrum",
    "PowerSpectrumResult",
    "CrossSpectrumResult",
]


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
