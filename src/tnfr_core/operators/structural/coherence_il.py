"""Ideal line coherence smoothing utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import List

import numpy as np

try:  # pragma: no cover - optional dependency
    import jax.numpy as jnp  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - exercised when JAX is unavailable
    jnp = None  # type: ignore[assignment]


def coherence_operator_il(series: Sequence[float], window: int = 5) -> List[float]:
    """Smooth a numeric series with a mean-preserving moving average.

    The operator applies a centred moving average using the requested odd ``window``
    size.  Edge samples are computed with the available neighbours and the result
    is debiased to preserve the original series mean exactly (up to floating point
    precision).

    Args:
        series: Source values representing the structural coherence signal.
        window: Length of the sliding window.  Must be a positive odd integer.

    Returns:
        A list with the smoothed series.

    Raises:
        ValueError: If ``window`` is not a positive odd integer.
    """

    window = int(window)
    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if not series:
        return []

    xp = jnp if jnp is not None else np
    array = xp.asarray(series, dtype=xp.float64)
    kernel = xp.ones(window, dtype=array.dtype)
    numerator = xp.convolve(array, kernel, mode="same")
    counts = xp.convolve(xp.ones_like(array), kernel, mode="same")
    if numerator.shape != array.shape:
        start = (numerator.shape[0] - array.shape[0]) // 2
        end = start + array.shape[0]
        numerator = numerator[start:end]
        counts = counts[start:end]
    smoothed = numerator / counts

    mean_input = float(array.mean())
    mean_output = float(smoothed.mean())
    bias = mean_input - mean_output
    if abs(bias) >= 1e-12:
        smoothed = smoothed + bias

    return np.asarray(smoothed, dtype=float).tolist()


__all__ = ["coherence_operator_il"]
