"""Sparkline rendering utilities."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

DEFAULT_SPARKLINE_BLOCKS: Sequence[str] = "▁▂▃▄▅▆▇█"

__all__ = ["DEFAULT_SPARKLINE_BLOCKS", "render_sparkline"]


def render_sparkline(
    values: Iterable[float],
    *,
    width: int | None = None,
    trim: bool = True,
    blocks: Sequence[str] = DEFAULT_SPARKLINE_BLOCKS,
    constant_block: str | None = None,
) -> str:
    """Render ``values`` as a Unicode block-character sparkline.

    Parameters
    ----------
    values:
        Iterable of numeric values to render.
    width:
        Optional maximum number of samples to include. When provided, the
        iterable is trimmed to ``width`` elements from the tail by default.
    trim:
        When ``True`` (default) and ``width`` is provided, keep the most recent
        samples. When ``False`` the sequence is truncated from the head.
    blocks:
        Sequence of characters representing increasing magnitudes.
    constant_block:
        Character to use when all values are identical. Defaults to the first
        entry in ``blocks``.
    """

    data = list(values)
    if width is not None:
        try:
            width = int(width)
        except (TypeError, ValueError):  # pragma: no cover - defensive only
            raise TypeError("width must be an integer or None") from None
        if width <= 0:
            return ""
        if trim:
            data = data[-width:]
        else:
            data = data[:width]
    if not data:
        return ""

    palette = tuple(blocks)
    if not palette:
        return ""

    minimum = min(data)
    maximum = max(data)
    if math.isclose(maximum, minimum):
        block = constant_block if constant_block is not None else palette[0]
        return block * len(data)

    span = maximum - minimum
    if span <= 0:
        span = 1.0

    buckets = len(palette) - 1
    if buckets <= 0:
        return palette[0] * len(data)

    rendered: list[str] = []
    for value in data:
        ratio = (value - minimum) / span
        index = int(round(ratio * buckets))
        index = max(0, min(buckets, index))
        rendered.append(palette[index])
    return "".join(rendered)
