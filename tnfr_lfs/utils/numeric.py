"""Numeric helper utilities."""

from __future__ import annotations

import math
from typing import SupportsFloat, SupportsInt


def _safe_float(
    value: SupportsFloat | SupportsInt | str | bytes | bytearray | memoryview | None,
    default: float | None = 0.0,
) -> float | None:
    """Return ``value`` coerced to ``float`` when finite, otherwise ``default``.

    Parameters
    ----------
    value:
        Input value to coerce. ``None`` and invalid values yield ``default``.
    default:
        Fallback value returned when ``value`` cannot be coerced to a finite float.
    """

    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


__all__ = ["_safe_float"]
