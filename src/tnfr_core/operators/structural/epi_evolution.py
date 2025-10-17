"""Compatibility re-exports for EPI evolution helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["NodalEvolution", "evolve_epi"]


if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from tnfr_core.equations.epi_evolution import NodalEvolution as _NodalEvolution
    from tnfr_core.equations.epi_evolution import evolve_epi as _evolve_epi


def __getattr__(name: str) -> Any:  # pragma: no cover - simple proxy
    if name in __all__:
        from tnfr_core.equations.epi_evolution import NodalEvolution, evolve_epi

        namespace = {"NodalEvolution": NodalEvolution, "evolve_epi": evolve_epi}
        value = namespace[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - reflection helper
    return sorted({*globals(), *__all__})
