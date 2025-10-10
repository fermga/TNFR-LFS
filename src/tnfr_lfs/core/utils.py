"""Shared helper utilities for TNFR × LFS core modules."""

from __future__ import annotations

import math
from collections.abc import Iterable

__all__ = ["normalised_entropy"]


def normalised_entropy(values: Iterable[float]) -> float:
    """Return the Shannon entropy of ``values`` normalised to ``[0, 1]``.

    The helper gracefully handles non-numeric and non-positive entries by
    ignoring them.  The computation automatically normalises the input weights
    prior to calculating the entropy so that callers may supply either raw
    weights or probability distributions.
    """

    positive: list[float] = []
    total = 0.0
    for raw in values:
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        if numeric <= 0.0:
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
