"""Signal processing helpers for :mod:`tnfr_core`."""

from . import spectrum as _spectrum

from .spectrum import *  # noqa: F401,F403

__all__ = list(dict.fromkeys(getattr(_spectrum, "__all__", ())))


del _spectrum
