"""Shared helper utilities for TNFR × LFS core modules."""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised indirectly when operators import is deferred
    from tnfr_core.runtime.shared import _HAS_JAX, jnp
except ImportError:  # pragma: no cover - fallback path for circular imports
    _HAS_JAX = False
    jnp = None

__all__ = ["normalised_entropy"]


def _ensure_backend() -> None:
    global _HAS_JAX, jnp
    if _HAS_JAX and jnp is not None:
        return
    try:
        from tnfr_core.runtime.shared import _HAS_JAX as resolved_has_jax, jnp as resolved_jnp
    except ImportError:  # pragma: no cover - defensive fallback
        return
    _HAS_JAX = resolved_has_jax
    jnp = resolved_jnp


def _normalised_entropy_iterable(values: Iterable[Any]) -> float:
    positive: list[float] = []
    total = 0.0
    for raw in values:
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric) or numeric <= 0.0:
            continue
        positive.append(numeric)
        total += numeric

    if len(positive) <= 1 or total <= 0.0:
        return 0.0

    probabilities = [value / total for value in positive]
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0.0)
    if not math.isfinite(entropy):
        return 0.0

    max_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    if max_entropy <= 0.0:
        return 0.0

    normalised = entropy / max_entropy
    if normalised <= 0.0:
        return 0.0
    if normalised >= 1.0:
        return 1.0
    return normalised


def _normalised_entropy_from_array(array: Any, backend: Any) -> float:
    values = backend.asarray(array, dtype=float)
    values = backend.ravel(values)
    if getattr(values, "size", 0) == 0:
        return 0.0

    mask = backend.isfinite(values) & (values > 0.0)
    count_positive = int(backend.count_nonzero(mask))
    if count_positive <= 1:
        return 0.0

    positive = backend.where(mask, values, 0.0)
    total = backend.sum(positive)
    total_value = float(total)
    if total_value <= 0.0 or not math.isfinite(total_value):
        return 0.0

    probabilities = positive / total
    prob_mask = probabilities > 0.0
    safe_probabilities = backend.where(prob_mask, probabilities, 1.0)
    entropy_terms = backend.where(
        prob_mask,
        probabilities * backend.log(safe_probabilities),
        0.0,
    )
    entropy = -backend.sum(entropy_terms)
    entropy_value = float(entropy)
    if not math.isfinite(entropy_value) or entropy_value <= 0.0:
        return 0.0

    max_entropy = backend.log(backend.asarray(float(count_positive), dtype=float))
    max_entropy_value = float(max_entropy)
    if max_entropy_value <= 0.0 or not math.isfinite(max_entropy_value):
        return 0.0

    normalised = entropy / max_entropy
    normalised_value = float(normalised)
    if normalised_value <= 0.0:
        return 0.0
    if normalised_value >= 1.0:
        return 1.0
    clipped = backend.clip(normalised, 0.0, 1.0)
    return float(clipped)


def normalised_entropy(values: Iterable[float]) -> float:
    """Return the Shannon entropy of ``values`` normalised to ``[0, 1]``.

    The helper gracefully handles non-numeric and non-positive entries by
    ignoring them.  The computation automatically normalises the input weights
    prior to calculating the entropy so that callers may supply either raw
    weights or probability distributions.
    """

    _ensure_backend()
    backend_candidates: list[Any] = []
    if _HAS_JAX and jnp is not None:
        backend_candidates.append(jnp)
    backend_candidates.append(np)

    materialised = values
    if not hasattr(materialised, "shape") and not isinstance(materialised, (list, tuple)):
        materialised = list(materialised)

    for backend in backend_candidates:
        try:
            return _normalised_entropy_from_array(materialised, backend)
        except (TypeError, ValueError):
            continue

    if isinstance(materialised, (list, tuple)):
        return _normalised_entropy_iterable(materialised)
    return _normalised_entropy_iterable(list(materialised))
