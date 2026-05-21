"""Damper-velocity histograms (bump/rebound × low/high-speed).

Race-engineering workhorse: bin per-wheel suspension velocities into a
symmetric histogram and split each side (bump = positive, rebound =
negative) into low- and high-speed populations. The reported
percentages and means are the standard MoTeC/AIM/Cosworth Pi setup
indicators used to decide bump/rebound clicks.

The module is pure ``numpy`` + ``pandas`` so it can be reused from the
Studio dock or from a notebook / batch report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Default low-speed boundary in m/s. 25 mm/s is the conventional split
# between low-speed (chassis pitch/roll) and high-speed (kerbs, bumps)
# damper work used by every major data-engineering tool.
DEFAULT_LOW_SPEED_MPS: float = 0.025
DEFAULT_BIN_WIDTH_MPS: float = 0.010
DEFAULT_MAX_ABS_MPS: float = 0.200


@dataclass(frozen=True)
class DamperHistogram:
    """Binned damper-velocity distribution + setup metrics.

    Attributes
    ----------
    bins:
        Bin centers in m/s, symmetric around 0 (rebound on the left,
        bump on the right).
    counts:
        Number of samples in each bin (raw count).
    fractions:
        ``counts / sum(counts)`` — same shape as ``bins``.
    low_speed_boundary_mps:
        Absolute speed below which a sample counts as "low-speed".
    bump_low_pct, bump_high_pct, rebound_low_pct, rebound_high_pct:
        Percentages (0..100) of total samples in each quadrant.
    bump_avg_mps, rebound_avg_mps:
        Mean of |speed| restricted to that side (0 if empty).
    """

    bins: np.ndarray
    counts: np.ndarray
    fractions: np.ndarray
    low_speed_boundary_mps: float
    bump_low_pct: float
    bump_high_pct: float
    rebound_low_pct: float
    rebound_high_pct: float
    bump_avg_mps: float
    rebound_avg_mps: float

    @property
    def bin_width_mps(self) -> float:
        if self.bins.size < 2:
            return 0.0
        return float(self.bins[1] - self.bins[0])


def damper_histogram(
    speeds_mps: np.ndarray | pd.Series,
    *,
    bin_width_mps: float = DEFAULT_BIN_WIDTH_MPS,
    max_abs_mps: float = DEFAULT_MAX_ABS_MPS,
    low_speed_mps: float = DEFAULT_LOW_SPEED_MPS,
) -> DamperHistogram:
    """Compute a damper-velocity histogram for one wheel.

    Samples outside ``[-max_abs_mps, +max_abs_mps]`` are clamped to the
    extreme bin (so spikes don't fall off the chart).
    """
    arr = np.asarray(speeds_mps, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or bin_width_mps <= 0 or max_abs_mps <= 0:
        return _empty_histogram(low_speed_mps)
    # Symmetric bin edges around 0.
    n_half = int(np.ceil(max_abs_mps / bin_width_mps))
    edges = np.linspace(
        -n_half * bin_width_mps, n_half * bin_width_mps, 2 * n_half + 1,
    )
    bins = 0.5 * (edges[:-1] + edges[1:])
    clamped = np.clip(arr, edges[0], edges[-1])
    counts, _ = np.histogram(clamped, bins=edges)
    total = float(counts.sum()) or 1.0
    fractions = counts.astype(float) / total

    low = abs(low_speed_mps)
    bump = arr[arr > 0]
    rebound = arr[arr < 0]
    bump_low_pct = 100.0 * float((bump <= low).sum()) / total if bump.size else 0.0
    bump_high_pct = 100.0 * float((bump > low).sum()) / total if bump.size else 0.0
    rebound_low_pct = (
        100.0 * float((rebound >= -low).sum()) / total if rebound.size else 0.0
    )
    rebound_high_pct = (
        100.0 * float((rebound < -low).sum()) / total if rebound.size else 0.0
    )
    bump_avg = float(bump.mean()) if bump.size else 0.0
    rebound_avg = float(-rebound.mean()) if rebound.size else 0.0

    return DamperHistogram(
        bins=bins,
        counts=counts,
        fractions=fractions,
        low_speed_boundary_mps=low,
        bump_low_pct=bump_low_pct,
        bump_high_pct=bump_high_pct,
        rebound_low_pct=rebound_low_pct,
        rebound_high_pct=rebound_high_pct,
        bump_avg_mps=bump_avg,
        rebound_avg_mps=rebound_avg,
    )


def _empty_histogram(low_speed_mps: float) -> DamperHistogram:
    return DamperHistogram(
        bins=np.zeros(0),
        counts=np.zeros(0, dtype=int),
        fractions=np.zeros(0),
        low_speed_boundary_mps=abs(low_speed_mps),
        bump_low_pct=0.0,
        bump_high_pct=0.0,
        rebound_low_pct=0.0,
        rebound_high_pct=0.0,
        bump_avg_mps=0.0,
        rebound_avg_mps=0.0,
    )


__all__ = [
    "DEFAULT_BIN_WIDTH_MPS",
    "DEFAULT_LOW_SPEED_MPS",
    "DEFAULT_MAX_ABS_MPS",
    "DamperHistogram",
    "damper_histogram",
]
