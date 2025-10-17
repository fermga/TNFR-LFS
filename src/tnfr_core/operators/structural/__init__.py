"""Structural operators exposed by :mod:`tnfr_core.operators`."""

from __future__ import annotations

from . import coherence_il as _coherence_il
from . import epi_evolution as _epi_evolution
from .coherence_il import *  # noqa: F401,F403
from .epi import (
    PhaseContext,
    compute_nodal_contributions,
    extract_phase_context,
    resolve_nu_targets,
)
from .epi_evolution import *  # noqa: F401,F403

__all__ = list(
    dict.fromkeys(
        [
            *_coherence_il.__all__,
            *_epi_evolution.__all__,
            "PhaseContext",
            "compute_nodal_contributions",
            "extract_phase_context",
            "resolve_nu_targets",
        ]
    )
)

del _coherence_il
del _epi_evolution
