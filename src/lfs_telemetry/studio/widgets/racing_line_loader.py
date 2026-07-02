"""Tiny helper to read ``racing_lines/<TRACK>_racing.csv`` files.

Parses a track's reference line into a :class:`RacingLine` (points +
bounding box, optional arclength and an optional per-row scalar such as
target speed). Used by the Track map dock to draw the ideal line and
apex markers. Pure logic (CSV parsing + bbox), no Qt dependency.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

_LOG = logging.getLogger(__name__)


@dataclass
class RacingLine:
    points: list[tuple[float, float]]  # (x_m, y_m) along the centerline
    bbox: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    # Cumulative arclength in metres per CSV row (== LFS path node).
    # Empty list when not available. ``total_length_m`` is the last
    # value plus the closing segment (loop back to point 0).
    s_m: list[float] = field(default_factory=list)
    total_length_m: float = 0.0
    # Optional per-row scalar (e.g. target speed) captured when
    # ``value_col`` is requested in :func:`load_racing_line`; empty
    # otherwise. Aligned 1:1 with :attr:`points`.
    values: list[float] = field(default_factory=list)

    @classmethod
    def empty(cls) -> RacingLine:
        return cls(points=[], bbox=(0.0, 0.0, 0.0, 0.0))

    @property
    def is_empty(self) -> bool:
        return not self.points

    def arclength_gap_m(self, view_node: int, other_node: int) -> float | None:
        """On-track distance from ``view_node`` to ``other_node`` (forward).

        Returns ``None`` if arclength data is unavailable or the node
        indices fall outside the racing-line table. The result is in
        the ``[0, total_length_m)`` range and wraps around the
        start/finish line so a car one node behind the view appears
        almost a full lap ahead (which is exactly what you want when
        you are about to lap them or be lapped).
        """
        if not self.s_m or self.total_length_m <= 0.0:
            return None
        n = len(self.s_m)
        if n == 0:
            return None
        # CompCar.node from LFS is 0-indexed; tolerate small drift.
        vi = int(view_node) % n
        oi = int(other_node) % n
        gap = self.s_m[oi] - self.s_m[vi]
        if gap < 0.0:
            gap += self.total_length_m
        return gap


def load_racing_line(
    path: Path | str,
    *,
    x_col: str = "x_center_m",
    y_col: str = "y_center_m",
    s_col: str = "s_m",
    value_col: str | None = None,
) -> RacingLine:
    """Read a racing-line CSV and return its points + bbox.

    Returns an empty :class:`RacingLine` if the file is missing or
    cannot be parsed (Studio renders nothing in that case). When
    ``value_col`` is given, that column is captured per row into
    :attr:`RacingLine.values` (NaN where missing), aligned with points.
    """
    p = Path(path)
    if not p.exists():
        return RacingLine.empty()
    pts: list[tuple[float, float]] = []
    s_vals: list[float] = []
    vals: list[float] = []
    try:
        with p.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    x = float(row[x_col])
                    y = float(row[y_col])
                except (KeyError, TypeError, ValueError):
                    continue
                pts.append((x, y))
                # ``s_m`` is optional; if missing we'll synthesise it
                # below from euclidean segment lengths.
                try:
                    s_vals.append(float(row[s_col]))
                except (KeyError, TypeError, ValueError):
                    s_vals.append(math.nan)
                if value_col is not None:
                    try:
                        vals.append(float(row[value_col]))
                    except (KeyError, TypeError, ValueError):
                        vals.append(math.nan)
    except OSError:
        return RacingLine.empty()
    if not pts:
        return RacingLine.empty()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # Build a usable arclength table. Prefer the CSV's ``s_m`` column
    # when present (already accounts for elevation), otherwise fall
    # back to 2D euclidean segment lengths.
    if any(math.isnan(v) for v in s_vals):
        s_vals = [0.0]
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            s_vals.append(s_vals[-1] + math.hypot(dx, dy))
    # Total length closes the loop back to point 0.
    if len(pts) >= 2:
        dx = pts[0][0] - pts[-1][0]
        dy = pts[0][1] - pts[-1][1]
        total = s_vals[-1] + math.hypot(dx, dy)
    else:
        total = s_vals[-1] if s_vals else 0.0
    return RacingLine(
        points=pts,
        bbox=(min(xs), min(ys), max(xs), max(ys)),
        s_m=s_vals,
        total_length_m=total,
        values=vals,
    )


def find_racing_line_for_track(
    racing_lines_dir: Path | str, track: str | None
) -> RacingLine:
    """Locate ``<racing_lines_dir>/<TRACK>_racing.csv`` and load it."""
    if not track:
        return RacingLine.empty()
    base = Path(racing_lines_dir)
    candidate = base / f"{track}_racing.csv"
    if not candidate.exists():
        _LOG.debug(
            "racing line missing for track %s (looked at %s); "
            "no reference centreline available",
            track, candidate,
        )
    return load_racing_line(candidate)


__all__ = [
    "RacingLine",
    "find_racing_line_for_track",
    "load_racing_line",
]
