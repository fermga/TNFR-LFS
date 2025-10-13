"""Metrics and reporting utilities for :mod:`tnfr_core`."""

from . import coherence_calibration as _coherence_calibration
from . import metrics as _metrics
from . import resonance as _resonance
from . import segmentation as _segmentation
from . import spectrum as _spectrum

from .coherence_calibration import *  # noqa: F401,F403
from .metrics import *  # noqa: F401,F403
from .resonance import *  # noqa: F401,F403
from .segmentation import *  # noqa: F401,F403
from .spectrum import *  # noqa: F401,F403

def _exported(module: object) -> list[str]:
    names = getattr(module, "__all__", None)
    if names is not None:
        return list(names)
    return [name for name in vars(module) if not name.startswith("_")]


__all__ = [
    *_exported(_coherence_calibration),
    *_exported(_metrics),
    *_exported(_resonance),
    *_exported(_segmentation),
    *_exported(_spectrum),
]

__all__ = list(dict.fromkeys(__all__))

_MODULES = (_metrics, _coherence_calibration, _resonance, _segmentation, _spectrum)

del _coherence_calibration
del _metrics
del _resonance
del _segmentation
del _spectrum


def __dir__() -> list[str]:
    names = set(__all__) | {name for name in globals() if not name.startswith("_")}
    return sorted(names)


def __setattr__(name: str, value: object) -> None:  # pragma: no cover - assignment helper
    for module in _MODULES:
        if hasattr(module, name):
            setattr(module, name, value)
    globals()[name] = value
