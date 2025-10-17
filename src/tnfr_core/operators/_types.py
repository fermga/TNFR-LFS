"""Shared operator type definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DissonanceBreakdown:
    """Breakdown of dissonance events by usefulness."""

    value: float
    useful_magnitude: float
    parasitic_magnitude: float
    useful_ratio: float
    parasitic_ratio: float
    useful_percentage: float
    parasitic_percentage: float
    total_events: int
    useful_events: int
    parasitic_events: int
    useful_dissonance_ratio: float
    useful_dissonance_percentage: float
    high_yaw_acc_samples: int
    useful_dissonance_samples: int


__all__ = ["DissonanceBreakdown"]
