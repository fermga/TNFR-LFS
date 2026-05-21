"""Track map: canonical X/Y geometry derived from one or more laps.

Used by MoTeC-style apps to draw the track outline, paint a coloured
racing-line plot (one colour per channel value), and place sector
markers on the plan view. The geometry is built from the OutSim
``pos_x`` / ``pos_y`` channels, resampled by distance so that two laps
captured with different sample counts share a common axis.

Only :mod:`numpy` and :mod:`pandas` are imported.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path as _FsPath

import numpy as np
import pandas as pd

from .comparison import _unwrapped_lap_arrays
from .lap import LapTelemetry

PthLike = str | _FsPath


@dataclass(frozen=True)
class TrackBounds:
    """Axis-aligned bounding box of a :class:`TrackMap` (m)."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width_m(self) -> float:
        return float(self.x_max - self.x_min)

    @property
    def height_m(self) -> float:
        return float(self.y_max - self.y_min)


@dataclass
class TrackMap:
    """A distance-parameterised X/Y curve representing the racing line.

    Build with :meth:`from_lap` or :meth:`from_laps` (the latter
    averages multiple laps for a smoother canonical line).
    """

    distance_m: np.ndarray   # shape (n,), monotone from 0 to length_m
    x_m: np.ndarray          # shape (n,)
    y_m: np.ndarray          # shape (n,)
    track: str | None = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_lap(
        cls,
        lap: LapTelemetry,
        *,
        n_points: int = 1000,
    ) -> TrackMap:
        """Build from one lap (resampled to ``n_points`` along distance)."""
        d_grid, x, y = _xy_along_distance(lap, n_points)
        track = lap.summary.get("track")
        return cls(distance_m=d_grid, x_m=x, y_m=y, track=track)

    @classmethod
    def from_laps(
        cls,
        laps: Iterable[LapTelemetry],
        *,
        n_points: int = 1000,
    ) -> TrackMap:
        """Average X/Y across multiple laps on the same distance grid.

        Laps with mismatched lengths are clipped to the shortest common
        post-line distance window so the average is well-defined.
        Laps with no usable position data are skipped.
        """
        laps = list(laps)
        if not laps:
            raise ValueError("from_laps requires at least one lap")
        # Find common distance window across all laps. The window may
        # extend below zero when the slice contains samples captured
        # *before* the start/finish-line crossing — those are valid
        # geometry and must be kept so the full circuit is drawn.
        ranges: list[tuple[float, float]] = []
        for lap in laps:
            _, d, _ = _unwrapped_lap_arrays(lap)
            if d.size >= 2 and {"pos_x", "pos_y"}.issubset(lap.raw.columns):
                ranges.append((float(d[0]), float(d[-1])))
        if not ranges:
            raise ValueError("no laps have usable distance + pos_x/pos_y data")
        d_lo = max(lo for lo, _ in ranges)
        d_hi = min(hi for _, hi in ranges)
        if d_hi <= d_lo:
            raise ValueError("laps share no common post-line distance window")
        grid = np.linspace(d_lo, d_hi, n_points)
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        track: str | None = None
        for lap in laps:
            try:
                _, x, y = _xy_along_grid(lap, grid)
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
            if track is None:
                track = lap.summary.get("track")
        if not xs:
            raise ValueError("no laps yielded resamplable XY data")
        x_mean = np.mean(np.vstack(xs), axis=0)
        y_mean = np.mean(np.vstack(ys), axis=0)
        return cls(distance_m=grid, x_m=x_mean, y_m=y_mean, track=track)

    @classmethod
    def from_pth(
        cls,
        track: str | PthLike,
        *,
        smx_dir: PthLike | None = None,
        n_points: int | None = None,
    ) -> TrackMap:
        """Build the canonical centreline from an LFS PTH file.

        This is the **ground truth** geometry shipped with LFS. Use it
        instead of :meth:`from_laps` when you want a stable track
        outline that is independent of how the player drove or how the
        capture window was clipped.

        Parameters
        ----------
        track
            Either a track id (``"BL1"``, ``"FE2R"``, …), a path to a
            ``.pth`` file, or a parsed :class:`Path` / :class:`TrackProfile`.
        smx_dir
            Override LFS smx directory. Defaults to ``C:\\LFS\\data\\smx``.
        n_points
            Resample the centreline to this many uniformly-spaced
            distance steps. ``None`` (default) keeps native PTH nodes.
        """
        # Local import so the telemetry sub-package keeps no hard
        # runtime dependency on the LFS install.
        from .track.pth import DEFAULT_SMX_DIR, TrackProfile, compute_profile, parse_pth
        from .track.pth import Path as PthPath

        profile: TrackProfile
        if isinstance(track, TrackProfile):
            profile = track
        else:
            if isinstance(track, PthPath):
                pth = track
            else:
                base = _FsPath(smx_dir) if smx_dir else DEFAULT_SMX_DIR
                if isinstance(track, (str, _FsPath)):
                    candidate = _FsPath(track)
                    if candidate.suffix.lower() == ".pth" and candidate.exists():
                        pth_path = candidate
                    else:
                        pth_path = base / f"{str(track).upper()}.pth"
                else:
                    raise TypeError(f"unsupported track argument: {type(track)!r}")
                if not pth_path.exists():
                    raise FileNotFoundError(f"PTH file not found: {pth_path}")
                pth = parse_pth(pth_path)
            profile = compute_profile(pth)

        d = np.asarray(profile.s, dtype=float)
        x = np.asarray(profile.pos[:, 0], dtype=float)
        y = np.asarray(profile.pos[:, 1], dtype=float)
        if n_points is not None and n_points >= 2 and d.size >= 2:
            grid = np.linspace(float(d[0]), float(d[-1]), int(n_points))
            x = np.interp(grid, d, x)
            y = np.interp(grid, d, y)
            d = grid
        return cls(distance_m=d, x_m=x, y_m=y, track=profile.name)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def length_m(self) -> float:
        if self.distance_m.size == 0:
            return 0.0
        return float(self.distance_m[-1] - self.distance_m[0])

    @property
    def n_points(self) -> int:
        return int(self.distance_m.size)

    def bounds(self) -> TrackBounds:
        return TrackBounds(
            x_min=float(np.min(self.x_m)) if self.x_m.size else 0.0,
            x_max=float(np.max(self.x_m)) if self.x_m.size else 0.0,
            y_min=float(np.min(self.y_m)) if self.y_m.size else 0.0,
            y_max=float(np.max(self.y_m)) if self.y_m.size else 0.0,
        )

    def xy_at_distance(self, d_m: float) -> tuple[float, float]:
        """Interpolate the (x, y) position at distance ``d_m`` (m)."""
        if self.distance_m.size < 2:
            return (float("nan"), float("nan"))
        x = float(np.interp(d_m, self.distance_m, self.x_m))
        y = float(np.interp(d_m, self.distance_m, self.y_m))
        return (x, y)

    def xy_at_distances(self, d_m: Sequence[float]) -> pd.DataFrame:
        """Interpolate ``[(x, y)]`` at multiple distances; returns DataFrame."""
        d_arr = np.asarray(list(d_m), dtype=float)
        if self.distance_m.size < 2:
            return pd.DataFrame(
                {"distance_m": d_arr,
                 "x_m": np.full(d_arr.size, np.nan),
                 "y_m": np.full(d_arr.size, np.nan)})
        x = np.interp(d_arr, self.distance_m, self.x_m)
        y = np.interp(d_arr, self.distance_m, self.y_m)
        return pd.DataFrame({"distance_m": d_arr, "x_m": x, "y_m": y})

    def to_dataframe(self) -> pd.DataFrame:
        """Return ``(distance_m, x_m, y_m)`` as a DataFrame for plotting."""
        return pd.DataFrame({
            "distance_m": self.distance_m,
            "x_m": self.x_m,
            "y_m": self.y_m,
        })


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _xy_along_distance(
    lap: LapTelemetry,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _idx, d, _ = _unwrapped_lap_arrays(lap)
    if d.size < 2:
        raise ValueError("lap has no usable distance data")
    df = lap.raw
    if not {"pos_x", "pos_y"}.issubset(df.columns):
        raise ValueError("lap is missing pos_x / pos_y columns")
    # Use the FULL distance window (including pre-line negatives) so the
    # whole circuit is drawn, not just the post-line fragment.
    d_lo = float(d[0])
    d_hi = float(d[-1])
    if d_hi <= d_lo:
        raise ValueError("lap has no positive distance window")
    grid = np.linspace(d_lo, d_hi, n_points)
    return _xy_along_grid(lap, grid)


def _xy_along_grid(
    lap: LapTelemetry,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx, d, _ = _unwrapped_lap_arrays(lap)
    df = lap.raw
    if not {"pos_x", "pos_y"}.issubset(df.columns):
        raise ValueError("lap is missing pos_x / pos_y columns")
    x_full = pd.to_numeric(df["pos_x"], errors="coerce").to_numpy()
    y_full = pd.to_numeric(df["pos_y"], errors="coerce").to_numpy()
    if x_full.size <= idx.max() or y_full.size <= idx.max():
        raise ValueError("pos_x / pos_y arrays shorter than index")
    x = x_full[idx]
    y = y_full[idx]
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        raise ValueError("lap has fewer than 2 valid (x, y) samples")
    x_grid = np.interp(grid, d[valid], x[valid])
    y_grid = np.interp(grid, d[valid], y[valid])
    return grid, x_grid, y_grid
