"""Compatibility bridge re-exporting :mod:`tnfr_core` symbols."""

from __future__ import annotations

import warnings

import tnfr_core as _core
from tnfr_core import equations, metrics, operators
from tnfr_core import *  # noqa: F401,F403

warnings.warn(
    "'tnfr_lfs.core' is deprecated and will be removed in a future release; "
    "import from 'tnfr_core' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = list(_core.__all__)

# Surface the reorganised modules to ease the migration path.
equations = equations
metrics = metrics
operators = operators

del _core
