"""Sense-index stage helpers for the ΔNFR×Si pipeline."""

from __future__ import annotations

from statistics import mean
from typing import Callable, Dict, Sequence

SenseFilter = Callable[[Sequence[float], float, float], Sequence[float]]


def _stage_sense(
    series: Sequence[float],
    *,
    recursion_decay: float,
    recursive_filter: SenseFilter,
) -> Dict[str, object]:
    if not series:
        return {
            "series": [],
            "memory": [],
            "average": 0.0,
            "decay": recursion_decay,
        }

    recursive_trace = recursive_filter(series, series[0], recursion_decay)
    return {
        "series": list(series),
        "memory": recursive_trace,
        "average": mean(series),
        "decay": recursion_decay,
    }
