"""Common helpers for ΔNFR distribution logic."""

from __future__ import annotations

import math
from typing import Dict, Mapping

__all__: list[str] = []


def distribute_weighted_delta(
    delta: float,
    signals: Mapping[str, float],
    *,
    min_total: float = 1e-9,
    min_signal: float = 0.0,
) -> Dict[str, float]:
    """Distribute ``delta`` proportionally to the magnitude of ``signals``.

    Parameters
    ----------
    delta:
        Total ΔNFR magnitude to distribute across the provided signals.
    signals:
        Mapping of signal identifiers to their associated strength.  The
        absolute value of each entry is used when computing the proportional
        distribution.  Non-finite or non-numeric values are ignored.
    min_total:
        Minimum aggregate signal magnitude required before falling back to a
        uniform distribution.  Any aggregate below this threshold, or a
        non-finite total, triggers an even split across the remaining keys.
    min_signal:
        Signals with an absolute magnitude at or below this threshold are
        treated as zero-strength contributions.  They still participate in the
        uniform fallback should the aggregate strength be insufficient.
    """

    weights: Dict[str, float] = {}
    fallback_keys: list[str] = []
    threshold = max(0.0, float(min_signal))

    for key, raw_value in signals.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        identifier = str(key)
        fallback_keys.append(identifier)
        weight = abs(value)
        if weight <= threshold:
            weight = 0.0
        weights[identifier] = weight

    if not weights:
        return {}

    total = sum(weights.values())
    if not math.isfinite(total) or total <= max(0.0, float(min_total)):
        count = len(fallback_keys)
        if count == 0:
            return {}
        share = delta / float(count)
        return {key: share for key in fallback_keys}

    magnitude = abs(delta)
    sign = 1.0 if delta >= 0.0 else -1.0
    return {
        key: sign * magnitude * (weight / total) if weight > 0.0 else 0.0
        for key, weight in weights.items()
    }

