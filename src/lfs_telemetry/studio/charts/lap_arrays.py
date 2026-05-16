"""Per-lap x/y array extraction with a tiny in-process cache.

Channel toggles re-render the same lap many times; ``LapArrayCache``
memoizes both the x-array (distance-unwrapped or time) and per-channel
LTTB-decimated y-arrays keyed by ``(lap_id, axis_kind, n_samples)``.

The unwrap for the distance axis is identical to
:func:`lfs_telemetry.telemetry.comparison._unwrapped_lap_arrays` — the
single source of truth — so the studio's distance axis lines up with
every comparison delta the rest of the package produces.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...telemetry.comparison import _unwrapped_lap_arrays
from .decimate import DECIMATE_THRESHOLD, DECIMATE_TARGET, lttb


def lap_x_array(lap, axis_kind: str) -> np.ndarray:
    """Return the x-axis array for ``lap`` in either ``distance`` or ``time``.

    ``distance`` uses the line-anchored unwrap so reset-to-zero crossings
    don't fold the trace back on itself; ``time`` is elapsed seconds
    relative to the lap start.
    """
    idx, d, t = _unwrapped_lap_arrays(lap)
    if axis_kind == "distance":
        return d
    return t


def lap_y_array(lap, column: str) -> np.ndarray:
    """Return one channel's values aligned with ``lap_x_array``."""
    idx, _, _ = _unwrapped_lap_arrays(lap)
    df = lap.enriched
    if column not in df.columns:
        raise KeyError(column)
    series = df[column].to_numpy()
    return series[idx]


class LapArrayCache:
    """Per-lap memoization for x arrays and decimated (x, y) pairs.

    One instance is owned by :class:`MultiChannelChart`; it lives as
    long as the chart panel. Bumping the lap drops stale entries.
    """

    def __init__(self) -> None:
        self._x: dict[Tuple[int, str], np.ndarray] = {}
        self._yz: dict[Tuple[int, str, str, int], Tuple[np.ndarray, np.ndarray]] = {}

    def x(self, lap, axis_kind: str) -> np.ndarray:
        key = (id(lap), axis_kind)
        hit = self._x.get(key)
        if hit is not None:
            return hit
        arr = lap_x_array(lap, axis_kind)
        self._x[key] = arr
        return arr

    def xy_decimated(
        self, lap, column: str, axis_kind: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = self.x(lap, axis_kind)
        key = (id(lap), column, axis_kind, x.size)
        hit = self._yz.get(key)
        if hit is not None:
            return hit
        y = lap_y_array(lap, column)
        if x.size <= DECIMATE_THRESHOLD:
            pair = (x, y)
        else:
            pair = lttb(x, y, DECIMATE_TARGET)
        self._yz[key] = pair
        return pair

    def drop_lap(self, lap) -> None:
        """Forget every entry owned by ``lap`` (call when lap is closed)."""
        ident = id(lap)
        self._x = {k: v for k, v in self._x.items() if k[0] != ident}
        self._yz = {k: v for k, v in self._yz.items() if k[0] != ident}

    def clear(self) -> None:
        self._x.clear()
        self._yz.clear()


__all__ = ["LapArrayCache", "lap_x_array", "lap_y_array"]
