"""Chart widgets and decimation."""

from __future__ import annotations

from .decimate import DECIMATE_TARGET, DECIMATE_THRESHOLD, lttb, maybe_decimate
from .lap_arrays import LapArrayCache, lap_x_array, lap_y_array
from .multi_chart import MultiChannelChart

__all__ = [
    "DECIMATE_TARGET", "DECIMATE_THRESHOLD",
    "LapArrayCache", "MultiChannelChart",
    "lap_x_array", "lap_y_array", "lttb", "maybe_decimate",
]
