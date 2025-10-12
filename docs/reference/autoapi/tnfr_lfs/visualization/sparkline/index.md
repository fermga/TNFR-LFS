# `tnfr_lfs.visualization.sparkline` module
Sparkline rendering utilities.

## Functions
- `render_sparkline(values: Iterable[float], *, width: int | None = None, trim: bool = True, blocks: Sequence[str] = DEFAULT_SPARKLINE_BLOCKS, constant_block: str | None = None) -> str`
  - Render ``values`` as a Unicode block-character sparkline.

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

## Attributes
- `DEFAULT_SPARKLINE_BLOCKS = '▁▂▃▄▅▆▇█'`

