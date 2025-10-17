"""Compatibility re-exports for EPI evolution helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "NodalEvolution",
    "evolve_epi",
    "PhaseContext",
    "extract_phase_context",
    "resolve_nu_targets",
    "compute_nodal_contributions",
]


if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from tnfr_core.equations._structural_epi import (
        PhaseContext as _PhaseContext,
        compute_nodal_contributions as _compute_nodal_contributions,
        extract_phase_context as _extract_phase_context,
        resolve_nu_targets as _resolve_nu_targets,
    )
    from tnfr_core.equations.epi_evolution import NodalEvolution as _NodalEvolution
    from tnfr_core.equations.epi_evolution import evolve_epi as _evolve_epi


def __getattr__(name: str) -> Any:  # pragma: no cover - simple proxy
    if name in __all__:
        if name in {"NodalEvolution", "evolve_epi"}:
            from tnfr_core.equations.epi_evolution import NodalEvolution, evolve_epi

            namespace = {"NodalEvolution": NodalEvolution, "evolve_epi": evolve_epi}
        else:
            from tnfr_core.equations._structural_epi import (
                PhaseContext,
                compute_nodal_contributions,
                extract_phase_context,
                resolve_nu_targets,
            )

            namespace = {
                "PhaseContext": PhaseContext,
                "extract_phase_context": extract_phase_context,
                "resolve_nu_targets": resolve_nu_targets,
                "compute_nodal_contributions": compute_nodal_contributions,
            }
        value = namespace[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - reflection helper
    return sorted({*globals(), *__all__})
