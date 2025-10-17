"""Signal processing helpers for :mod:`tnfr_core`."""

from __future__ import annotations

from .spectrum import (  # noqa: F401
    CrossSpectrumResult,
    PowerSpectrumResult,
    apply_window,
    cross_spectrum,
    detrend,
    estimate_sample_rate,
    hann_window,
    power_spectrum,
)

__all__ = [
    "CrossSpectrumResult",
    "PowerSpectrumResult",
    "apply_window",
    "cross_spectrum",
    "detrend",
    "estimate_sample_rate",
    "hann_window",
    "power_spectrum",
]

