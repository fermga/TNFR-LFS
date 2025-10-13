"""Equation solvers and analytic primitives for :mod:`tnfr_core`."""

from . import archetypes as _archetypes
from . import coherence as _coherence
from . import constants as _constants
from . import contextual_delta as _contextual_delta
from . import delta_utils as _delta_utils
from . import dissonance as _dissonance
from . import epi as _epi
from . import epi_models as _epi_models
from . import phases as _phases
from . import utils as _utils

from .archetypes import *  # noqa: F401,F403
from .coherence import *  # noqa: F401,F403
from .constants import *  # noqa: F401,F403
from .contextual_delta import *  # noqa: F401,F403
from .delta_utils import *  # noqa: F401,F403
from .dissonance import *  # noqa: F401,F403
from .epi import *  # noqa: F401,F403
from .epi_models import *  # noqa: F401,F403
from .phases import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403

def _exported(module: object) -> list[str]:
    names = getattr(module, "__all__", None)
    if names is not None:
        return list(names)
    return [name for name in vars(module) if not name.startswith("_")]


__all__ = [
    *_exported(_archetypes),
    *_exported(_coherence),
    *_exported(_constants),
    *_exported(_contextual_delta),
    *_exported(_delta_utils),
    *_exported(_dissonance),
    *_exported(_epi),
    *_exported(_epi_models),
    *_exported(_phases),
    *_exported(_utils),
]

__all__ = list(dict.fromkeys(__all__))

del _archetypes
del _coherence
del _constants
del _contextual_delta
del _delta_utils
del _dissonance
del _epi
del _epi_models
del _phases
del _utils
