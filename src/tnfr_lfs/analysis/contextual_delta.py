"""LFS-side helpers for contextual ΔNFR calibration."""

from __future__ import annotations

from typing import Mapping

try:  # pragma: no cover - ``tomllib`` only ships with Python ≥ 3.11
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fall back to ``tomli`` in 3.10
    import tomli as tomllib  # type: ignore

from tnfr_core.equations.contextual_delta import (
    ContextFactors,
    ContextMatrix,
    apply_contextual_delta,
    configure_context_matrix_loader,
    load_context_matrix,
    resolve_context_from_bundle,
    resolve_context_from_record,
    resolve_microsector_context,
    resolve_series_context,
)

from tnfr_lfs.data import CONTEXT_FACTORS_RESOURCE

__all__ = [
    "ContextFactors",
    "ContextMatrix",
    "apply_contextual_delta",
    "configure_context_matrix_loader",
    "ensure_context_loader",
    "load_context_matrix",
    "resolve_context_from_bundle",
    "resolve_context_from_record",
    "resolve_microsector_context",
    "resolve_series_context",
]


def _load_context_payload() -> Mapping[str, object]:
    with CONTEXT_FACTORS_RESOURCE.open("rb") as handle:
        return tomllib.load(handle)


def ensure_context_loader() -> None:
    """Ensure the core package knows how to resolve contextual calibrations."""

    configure_context_matrix_loader(_load_context_payload)


# Configure the loader immediately so downstream imports are transparent.
ensure_context_loader()
