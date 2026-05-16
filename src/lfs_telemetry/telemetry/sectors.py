"""Sector slicing for laps and stints.

A "sector" is a contiguous slice of a lap defined by distance
boundaries (relative to the start/finish line). MoTeC-style apps use
sectors for two things:

* per-sector timing (so you can see *where* lap A is gaining/losing on
  lap B), and
* theoretical-best-lap construction (sum of best sector times across
  the stint).

Boundaries source — ranked by reliability for this codebase:

1. **User-supplied** ``boundaries_m`` (list of distance offsets from
   the start/finish line, in metres). This is what the UI will pass in
   when the user drags sector markers on the track map.
2. **InSim splits** persisted on the CSV (``ctx_view_last_splitN_ms``)
   — currently only the *cumulative* values are stored, not the
   per-sample geometric crossings, so this is opportunistic: if a lap
   exposes monotone split times we use them, otherwise we fall back to
   uniform segmentation.
3. **Uniform** division into ``n_equal`` equal-distance sectors
   (default ``n_equal=3`` to match the typical 3-sector layout).

Everything in this module depends only on :mod:`numpy` + :mod:`pandas`
and the existing ``LapTelemetry`` / unwrap helpers in ``comparison``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .comparison import _unwrapped_lap_arrays
from .lap import LapTelemetry


def _lap_distance_time(lap: LapTelemetry) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(d, t)`` for the lap's post-line slice, anchored at d=0, t=0.

    Uses :func:`comparison._unwrapped_lap_arrays` to get a continuous
    distance/time pair across any wraparound, then keeps only the
    post-line portion (``d_rel >= 0``) and re-anchors so ``d[0] = 0``,
    ``t[0] = 0``. This makes sectors comparable across captures even
    when each CSV is a sliding window containing a single line crossing.

    Returns empty arrays if the lap has no usable data.
    """
    _, d, t = _unwrapped_lap_arrays(lap)
    if d.size < 2:
        return np.zeros(0), np.zeros(0)
    mask = d >= 0
    if mask.sum() < 2:
        return np.zeros(0), np.zeros(0)
    d = d[mask]
    t = t[mask]
    # Re-anchor to (0, 0).
    d = d - d[0]
    t = t - t[0]
    # Drop duplicate distances (numpy.interp requires strictly increasing).
    keep = np.concatenate(([True], np.diff(d) > 0))
    return d[keep], t[keep]


@dataclass(frozen=True)
class Sector:
    """One sector of a lap, in distance and time."""

    index: int          # 0-based sector index
    start_d_m: float    # distance from line at sector start (m)
    end_d_m: float      # distance from line at sector end (m)
    start_t_s: float    # time from line at sector start (s, may be negative)
    end_t_s: float      # time from line at sector end (s)

    @property
    def time_s(self) -> float:
        """Sector duration (s). Always positive for a well-formed sector."""
        return float(self.end_t_s - self.start_t_s)

    @property
    def length_m(self) -> float:
        """Sector length (m)."""
        return float(self.end_d_m - self.start_d_m)


def lap_sectors(
    lap: LapTelemetry,
    *,
    boundaries_m: Sequence[float] | None = None,
    n_equal: int = 3,
) -> list[Sector]:
    """Slice ``lap`` into sectors and return their boundaries / times.

    Parameters
    ----------
    lap
        Source lap.
    boundaries_m
        Interior boundaries (m from line). For a lap of length ``L``
        and ``boundaries_m=[a, b]`` you get 3 sectors covering
        ``[0, a]``, ``[a, b]``, ``[b, L]``. Values outside the lap are
        clamped. Pass an empty list for a single sector.
    n_equal
        Used only when ``boundaries_m`` is ``None``. Number of equal-
        distance sectors (default 3).

    Returns
    -------
    list[Sector]
        Always at least one sector, ordered by distance. Empty list if
        the lap has no usable distance/time data.
    """
    d, t = _lap_distance_time(lap)
    if d.size < 2:
        return []
    d_lo = float(d[0])
    d_hi = float(d[-1])
    if d_hi <= d_lo:
        return []

    if boundaries_m is None:
        if n_equal < 1:
            raise ValueError("n_equal must be >= 1")
        edges = np.linspace(d_lo, d_hi, n_equal + 1)
    else:
        cleaned = sorted(float(b) for b in boundaries_m if np.isfinite(b))
        cleaned = [b for b in cleaned if d_lo < b < d_hi]
        edges = np.array([d_lo, *cleaned, d_hi], dtype=float)

    times = np.interp(edges, d, t)
    return [
        Sector(
            index=i,
            start_d_m=float(edges[i]),
            end_d_m=float(edges[i + 1]),
            start_t_s=float(times[i]),
            end_t_s=float(times[i + 1]),
        )
        for i in range(len(edges) - 1)
    ]


def sector_times_s(
    lap: LapTelemetry,
    *,
    boundaries_m: Sequence[float] | None = None,
    n_equal: int = 3,
) -> list[float]:
    """Convenience: just the per-sector durations (s)."""
    return [s.time_s for s in lap_sectors(
        lap, boundaries_m=boundaries_m, n_equal=n_equal)]


def insim_split_distances_m(lap: LapTelemetry) -> list[float]:
    """Best-effort geometric split distances from persisted InSim splits.

    Returns a sorted list of distance offsets (m, from the line) where
    each ``ctx_view_last_splitN_ms`` value lands when interpreted as
    cumulative time-since-line. Splits that fall outside the lap window
    or are NaN are skipped. May return ``[]`` if the lap has no usable
    split data.
    """
    df = lap.raw
    d, t = _lap_distance_time(lap)
    if d.size < 2:
        return []
    out: list[float] = []
    for col in ("ctx_view_last_split1_ms",
                "ctx_view_last_split2_ms",
                "ctx_view_last_split3_ms"):
        if col not in df.columns:
            continue
        try:
            v = float(df[col].dropna().iloc[-1])
        except (IndexError, TypeError, ValueError):
            continue
        if not np.isfinite(v) or v <= 0:
            continue
        t_s = v / 1000.0
        if t_s <= float(t[0]) or t_s >= float(t[-1]):
            continue
        d_at = float(np.interp(t_s, t, d))
        out.append(d_at)
    return sorted(set(out))
