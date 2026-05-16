"""Distance-aligned lap overlay and delta-time computation.

This is the core MoTeC-style feature: pick *reference* lap A and
*compare* lap B, resample both to a common distance grid, and compute
the running delta time so the app can paint:

* channel overlay (any signal vs distance from both laps), and
* a delta-time trace (how much B is ahead/behind A vs distance).

The math is straightforward:

* distance grid: uniform from 0 to min(distance_a, distance_b),
* time(distance) is monotonic-increasing → np.interp,
* delta_t(d) = t_b(d) - t_a(d).  Negative means B is faster at d.

Only :mod:`numpy` and :mod:`pandas` are imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

from .lap import LapTelemetry


@dataclass
class LapComparison:
    """Distance-aligned overlay between two laps."""

    reference: LapTelemetry          # lap A (baseline)
    candidate: LapTelemetry          # lap B (compared against A)
    n_points: int = 1000             # samples in the distance grid
    restrict_post_line: bool | None = None  # None = auto (True if any race start)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_laps(
        cls,
        reference: LapTelemetry,
        candidate: LapTelemetry,
        *,
        n_points: int = 1000,
        restrict_post_line: bool | None = None,
    ) -> "LapComparison":
        return cls(
            reference=reference,
            candidate=candidate,
            n_points=n_points,
            restrict_post_line=restrict_post_line,
        )

    # ------------------------------------------------------------------
    # Distance grid
    # ------------------------------------------------------------------

    @cached_property
    def _post_line_only(self) -> bool:
        if self.restrict_post_line is not None:
            return bool(self.restrict_post_line)
        return bool(self.reference.is_race_start or self.candidate.is_race_start)

    @cached_property
    def distance_grid_m(self) -> np.ndarray:
        """Uniform distance axis over the overlap of both laps.

        With the line-anchored unwrap (see :func:`_unwrapped_lap_arrays`),
        each lap exposes negative distances for samples **before** the
        first line crossing and positive distances after. The grid uses
        the intersection of the two ranges so the comparison is always
        defined.

        If either lap is a race start (or ``restrict_post_line=True``
        was forced), the grid is clipped to ``d >= 0`` because the
        pre-line segment of a race-start lap is the launch from grid,
        which is not comparable to the tail of a flying lap.
        """
        _, d_a, _ = _unwrapped_lap_arrays(self.reference)
        _, d_b, _ = _unwrapped_lap_arrays(self.candidate)
        if d_a.size < 2 or d_b.size < 2:
            return np.zeros(0, dtype=float)
        d_lo = float(max(d_a[0], d_b[0]))
        d_hi = float(min(d_a[-1], d_b[-1]))
        if self._post_line_only:
            d_lo = max(d_lo, 0.0)
        if not np.isfinite(d_lo) or not np.isfinite(d_hi) or d_hi <= d_lo:
            return np.zeros(0, dtype=float)
        return np.linspace(d_lo, d_hi, self.n_points)

    # ------------------------------------------------------------------
    # Delta time (the headline metric)
    # ------------------------------------------------------------------

    @cached_property
    def delta_time_s(self) -> np.ndarray:
        """Running ``t_candidate(d) - t_reference(d)`` along the grid.

        Negative values mean the candidate is **faster** than the
        reference at that point (further ahead in time). The trace is
        re-anchored so that ``delta_time_s[0] == 0``: only the *change*
        in time gap across the grid is meaningful when the start of the
        grid is not necessarily the start/finish line.
        """
        d = self.distance_grid_m
        if d.size == 0:
            return np.zeros(0, dtype=float)
        t_a = _time_at_distance(self.reference, d)
        t_b = _time_at_distance(self.candidate, d)
        delta = t_b - t_a
        if delta.size and np.isfinite(delta[0]):
            delta = delta - delta[0]
        return delta

    @cached_property
    def total_delta_s(self) -> float:
        """Final delta at the end of the comparison window (s)."""
        if self.delta_time_s.size == 0:
            return float("nan")
        return float(self.delta_time_s[-1])

    # ------------------------------------------------------------------
    # Channel overlay
    # ------------------------------------------------------------------

    def channel(self, column: str, *, enriched: bool = True) -> pd.DataFrame:
        """Distance-aligned values of ``column`` for both laps.

        Returns a DataFrame indexed by distance (m) with columns
        ``reference`` and ``candidate``. Use ``enriched=False`` to
        sample from the raw schema instead of the derived columns.
        """
        d = self.distance_grid_m
        ref = _resample_channel(self.reference, column, d, enriched=enriched)
        cand = _resample_channel(self.candidate, column, d, enriched=enriched)
        return pd.DataFrame(
            {"reference": ref, "candidate": cand},
            index=pd.Index(d, name="distance_m"),
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @cached_property
    def summary(self) -> dict[str, Any]:
        ref_time = self.reference.summary.get("lap_time_s")
        cand_time = self.candidate.summary.get("lap_time_s")
        d_grid = self.distance_grid_m
        out: dict[str, Any] = {
            "reference_lap_time_s": ref_time,
            "candidate_lap_time_s": cand_time,
            "lap_time_delta_s": (
                None if (ref_time is None or cand_time is None)
                else float(cand_time - ref_time)
            ),
            "total_delta_s_at_grid_end": self.total_delta_s,
            "grid_start_m": float(d_grid[0]) if d_grid.size else 0.0,
            "grid_end_m": float(d_grid[-1]) if d_grid.size else 0.0,
            "distance_grid_m": float(d_grid[-1] - d_grid[0]) if d_grid.size else 0.0,
            "n_points": self.n_points,
            "post_line_only": self._post_line_only,
            "reference_is_race_start": bool(self.reference.is_race_start),
            "candidate_is_race_start": bool(self.candidate.is_race_start),
        }
        d = self.delta_time_s
        if d.size:
            out["max_gain_s"] = float(-d.min())     # max time the candidate gained
            out["max_loss_s"] = float(d.max())      # max time the candidate lost
            out["max_swing_s"] = float(d.max() - d.min())
        return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _unwrapped_lap_arrays(
    lap: LapTelemetry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(idx_kept, d_rel, t_rel)`` for one lap of telemetry.

    Result is memoized on the lap instance (``lap._unwrapped_cache``)
    so repeated calls from figures, sectors, comparison and track-map
    factories reuse a single computation.

    LFS captures slice per IS_LAP events. Those events fire at the
    timing checkpoint, which is **not** always the geometric
    start/finish line — so each per-lap CSV typically holds:

        ``[ tail of lap N | line crossing | head of lap N+1 ]``

    ``current_lap_dist_m`` resets to 0 at the line (the wraparound
    inside the slice). To present this as a single coherent lap we:

    * detect every wrap (``Δd < -10 m``);
    * estimate ``track_len`` as the max distance reached just before the
      first wrap (this is the geometric lap length);
    * unwrap by adding ``track_len`` to every sample after each wrap so
      the distance becomes strictly monotone;
    * anchor at the slice's first sample (``d_rel`` starts at 0, the
      checkpoint where the slicer cut, **not** the line).

    For slices with no wrap we keep the slice as-is (anchored at 0).
    For slices with multiple wraps (rare; multi-lap slice) we keep the
    longest segment between consecutive wraps — that one is guaranteed
    to be a clean line-to-line lap.
    """
    cached = getattr(lap, "_unwrapped_cache", None)
    if cached is not None:
        return cached
    df = lap.raw
    # Prefer ``current_lap_dist_m`` (wraps at the line; we unwrap below).
    # Fall back to ``indexed_distance_m`` if the primary column is
    # missing or fully empty — robust against partial captures and
    # third-party CSVs that only carry the indexed channel.
    if "current_lap_dist_m" in df.columns:
        d = pd.to_numeric(df["current_lap_dist_m"], errors="coerce").to_numpy()
    else:
        d = np.full(len(df), np.nan)
    if not np.isfinite(d).any() and "indexed_distance_m" in df.columns:
        d = pd.to_numeric(df["indexed_distance_m"], errors="coerce").to_numpy()
    t = pd.to_numeric(df["time_ms"], errors="coerce").to_numpy() / 1000.0
    mask = np.isfinite(d) & np.isfinite(t)
    idx_all = np.where(mask)[0]
    d = d[mask].astype(float)
    t = t[mask].astype(float)
    if d.size < 2:
        result = (idx_all, d, t)
        _store_unwrapped_cache(lap, result)
        return result

    diffs = np.diff(d)
    wrap_pos = np.where(diffs < -10.0)[0]

    if wrap_pos.size == 0:
        d_rel = d - d[0]
        t_rel = t - t[0]
        result = _enforce_monotone(idx_all, d_rel, t_rel)
        _store_unwrapped_cache(lap, result)
        return result

    if wrap_pos.size == 1:
        # Single wrap: unwrap the trailing fragment and keep the whole
        # slice as one continuous lap from the checkpoint cut.
        track_len = float(d[wrap_pos[0]])
        d_un = d.copy()
        d_un[wrap_pos[0] + 1:] += track_len
        d_rel = d_un - d_un[0]
        t_rel = t - t[0]
        result = _enforce_monotone(idx_all, d_rel, t_rel)
        _store_unwrapped_cache(lap, result)
        return result

    # Multiple wraps: slice spans 2+ laps. Pick the longest inter-wrap
    # segment — that's a clean line-to-line lap.
    starts = wrap_pos[:-1] + 1
    ends = wrap_pos[1:]
    spans = d[ends] - d[starts]
    best = int(np.argmax(spans))
    s, e = int(starts[best]), int(ends[best])
    seg = slice(s, e + 1)
    d_seg = d[seg]
    t_seg = t[seg]
    d_rel = d_seg - d_seg[0]
    t_rel = t_seg - t_seg[0]
    result = _enforce_monotone(idx_all[seg], d_rel, t_rel)
    _store_unwrapped_cache(lap, result)
    return result


def _store_unwrapped_cache(
    lap: LapTelemetry,
    result: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Best-effort memoization on the lap instance."""
    try:
        lap._unwrapped_cache = result
    except (AttributeError, TypeError):
        pass


def _enforce_monotone(
    idx: np.ndarray, d_rel: np.ndarray, t_rel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop samples that break strict monotone increase of ``d_rel``."""
    if d_rel.size < 2:
        return idx, d_rel, t_rel
    mono = np.empty(d_rel.size, dtype=bool)
    mono[0] = True
    running = d_rel[0]
    for i in range(1, d_rel.size):
        if d_rel[i] > running:
            mono[i] = True
            running = d_rel[i]
        else:
            mono[i] = False
    return idx[mono], d_rel[mono], t_rel[mono]


def _lap_distance_m(lap: LapTelemetry) -> float:
    _, d, _ = _unwrapped_lap_arrays(lap)
    if d.size == 0:
        return float("nan")
    return float(d[-1])


def _time_at_distance(lap: LapTelemetry, d_grid: np.ndarray) -> np.ndarray:
    _, d, t = _unwrapped_lap_arrays(lap)
    if d.size < 2:
        return np.full_like(d_grid, np.nan, dtype=float)
    return np.interp(d_grid, d, t)


def _resample_channel(
    lap: LapTelemetry,
    column: str,
    d_grid: np.ndarray,
    *,
    enriched: bool,
) -> np.ndarray:
    df = lap.enriched if enriched else lap.raw
    if column not in df.columns:
        return np.full_like(d_grid, np.nan, dtype=float)
    idx, d, _ = _unwrapped_lap_arrays(lap)
    if d.size < 2:
        return np.full_like(d_grid, np.nan, dtype=float)
    y_full = pd.to_numeric(df[column], errors="coerce").to_numpy()
    if y_full.size <= idx.max():
        return np.full_like(d_grid, np.nan, dtype=float)
    y = y_full[idx]
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return np.full_like(d_grid, np.nan, dtype=float)
    return np.interp(d_grid, d[valid], y[valid])
