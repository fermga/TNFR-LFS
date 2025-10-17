"""Metadata helpers for structural EPI evolution."""

from __future__ import annotations

from tnfr_core.equations._structural_epi import (
    NON_NODAL_METADATA_KEYS,
    PhaseContext,
    extract_phase_context,
    resolve_nu_targets,
)

__all__ = [
    "PhaseContext",
    "extract_phase_context",
    "resolve_nu_targets",
    "NON_NODAL_METADATA_KEYS",
]
