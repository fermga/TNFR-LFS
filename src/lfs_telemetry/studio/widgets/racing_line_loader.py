"""Tiny helper to read ``racing_lines/<TRACK>_racing.csv`` files.

Used by the Live overlay's mini-map module to draw the track centerline
underneath the moving cars. Pure logic (just CSV parsing + bbox).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

_LOG = logging.getLogger(__name__)


@dataclass
class RacingLine:
    points: list[tuple[float, float]]  # (x_m, y_m) along the centerline
    bbox: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)

    @classmethod
    def empty(cls) -> "RacingLine":
        return cls(points=[], bbox=(0.0, 0.0, 0.0, 0.0))

    @property
    def is_empty(self) -> bool:
        return not self.points


def load_racing_line(
    path: Path | str,
    *,
    x_col: str = "x_center_m",
    y_col: str = "y_center_m",
) -> RacingLine:
    """Read a racing-line CSV and return its centerline + bbox.

    Returns an empty :class:`RacingLine` if the file is missing or
    cannot be parsed (Studio renders nothing in that case).
    """
    p = Path(path)
    if not p.exists():
        return RacingLine.empty()
    pts: list[tuple[float, float]] = []
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
    except OSError:
        return RacingLine.empty()
    if not pts:
        return RacingLine.empty()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return RacingLine(
        points=pts,
        bbox=(min(xs), min(ys), max(xs), max(ys)),
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
            "mini-map will render no centerline",
            track, candidate,
        )
    return load_racing_line(candidate)


__all__ = [
    "RacingLine",
    "find_racing_line_for_track",
    "load_racing_line",
]
