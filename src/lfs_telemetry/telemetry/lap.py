"""High-level :class:`LapTelemetry` facade for downstream apps.

This is the *single object* a MoTeC-style visualization app needs to
plot a captured CSV. It bundles:

* the raw schema DataFrame (pandas),
* the enriched DataFrame (28 derived signals; computed lazily),
* the resolved :class:`CarSpec`,
* basic per-lap metadata (lap_time_s, distance_m, top_speed_kmh, peaks).

The standalone app does::

    from lfs_telemetry.telemetry import LapTelemetry
    lap = LapTelemetry.from_csv("stint_bl1_fbm_lap01.csv")
    lap.enriched   # ready-to-plot DataFrame with friction_use_*, dl_*, ffb…
    lap.car        # CarSpec
    lap.summary    # dict with lap_time_s, distance_m, peak_lat_g, …

No analysis-engine imports are touched — the consumer pulls only
``pandas`` + ``numpy``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import lap_cache as _lap_cache
from .channels import ChannelInfo, channel_info, channels_by_group
from .derived import enrich_dataframe
from .observables import CarSpec, car_spec_for
from .replay import detect_schema_version, read_csv_dataframe


@dataclass
class LapTelemetry:
    """One captured lap, ready for analysis or plotting.

    Construct via :meth:`from_csv` (preferred) or directly from a
    DataFrame already loaded by the caller.
    """

    raw: pd.DataFrame
    car: CarSpec
    source_path: Path | None = None
    schema_version: str | None = None
    _enriched_cache: pd.DataFrame | None = field(default=None, repr=False)
    # Memoized line-anchored unwrap (idx, d_rel, t_rel); see
    # ``comparison._unwrapped_lap_arrays``. Heavy enough to be worth
    # caching across the multiple call sites (figures, sectors, track
    # map, comparison) that hit it on every UI interaction.
    _unwrapped_cache: tuple | None = field(
        default=None, repr=False, compare=False,
    )
    # Memoized LTTB-decimated (x, y) arrays per (column, x_axis_kind).
    # Channel toggling re-renders the same lap many times; without this
    # cache every toggle re-runs LTTB on every visible channel.
    _decimate_cache: dict = field(
        default_factory=dict, repr=False, compare=False,
    )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        car: CarSpec | str | None = None,
    ) -> LapTelemetry:
        """Load a captured CSV from :func:`write_csv_replay`.

        ``car`` accepts a :class:`CarSpec`, a short LFS car name (e.g.
        ``"FBM"``), or ``None`` to auto-detect from the first ``car``
        cell of the file.

        A user-scoped on-disk cache (``lap_cache``) memoizes both the
        parsed raw DataFrame and the enriched view keyed by file
        ``(path, mtime, size)``. Warm hits return in roughly 10-30 ms.
        """
        path = Path(path)
        cached = _lap_cache.load(path)
        if cached is not None:
            raw, enriched = cached
            spec = _resolve_car(car, raw)
            inst = cls(
                raw=raw,
                car=spec,
                source_path=path,
                schema_version=detect_schema_version(path),
            )
            inst._enriched_cache = enriched
            return inst

        df = read_csv_dataframe(path)
        spec = _resolve_car(car, df)
        return cls(
            raw=df,
            car=spec,
            source_path=path,
            schema_version=detect_schema_version(path),
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        car: CarSpec | str | None = None,
    ) -> LapTelemetry:
        """Wrap an already-loaded DataFrame (no I/O)."""
        return cls(raw=df, car=_resolve_car(car, df))

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    @property
    def enriched(self) -> pd.DataFrame:
        """Enriched DataFrame (raw + 28 derived columns). Cached."""
        if self._enriched_cache is None:
            self._enriched_cache = enrich_dataframe(self.raw, self.car)
            # Persist (raw, enriched) on first compute so the next process
            # to open the same CSV gets a warm hit.
            if self.source_path is not None:
                _lap_cache.save(
                    self.source_path, self.raw, self._enriched_cache,
                )
        return self._enriched_cache

    def reset_cache(self) -> None:
        """Drop the cached enriched DataFrame (e.g. after editing ``raw``)."""
        self._enriched_cache = None
        self._unwrapped_cache = None
        self._decimate_cache = {}

    # ------------------------------------------------------------------
    # Quick metadata for app headers / lap selectors
    # ------------------------------------------------------------------

    @cached_property
    def is_race_start(self) -> bool:
        """``True`` if this lap starts from a stopped grid position.

        Heuristic: somewhere in the **first 10 seconds** of the slice
        the car is essentially stationary (``speed_ms < 1.0`` m/s).
        This catches both:

        * recording started before lights-out (sample 0 is stopped), and
        * recording started moments before a standing restart (briefly
          stopped within the first few seconds).

        Race-start laps include the launch from grid → first line
        crossing, so their **pre-line** segment is grid acceleration,
        not the tail of a previous flying lap. Downstream consumers
        (StintTelemetry, LapComparison) use this flag to exclude or
        clip the lap appropriately.
        """
        df = self.raw
        if "speed_ms" not in df.columns or df.empty or "time_ms" not in df.columns:
            return False
        try:
            t0 = float(df["time_ms"].iloc[0])
        except (TypeError, ValueError):
            return False
        window = df.loc[df["time_ms"] < t0 + 10_000.0, "speed_ms"]
        if window.empty:
            return False
        v_min = float(window.min())
        return np.isfinite(v_min) and v_min < 1.0

    @cached_property
    def summary(self) -> dict[str, Any]:
        """Compact dict of headline numbers for UI / list views."""
        df = self.raw
        out: dict[str, Any] = {
            "samples": int(len(df)),
            "car": str(df["car"].iloc[0]) if "car" in df and len(df) else None,
            "schema_version": self.schema_version,
            "is_race_start": bool(self.is_race_start),
        }
        if "time_ms" in df and len(df):
            out["lap_time_s"] = float((df["time_ms"].iloc[-1] - df["time_ms"].iloc[0]) / 1000.0)
        if "current_lap_dist_m" in df and len(df):
            d = df["current_lap_dist_m"]
            out["distance_m"] = float(d.max() - d.min())
        if "speed_ms" in df and len(df):
            out["top_speed_kmh"] = float(df["speed_ms"].max() * 3.6)
        if "accel_x" in df and "accel_y" in df and len(df):
            out["peak_long_g"] = float(df["accel_x"].abs().max() / self.car.g)
            out["peak_lat_g"] = float(df["accel_y"].abs().max() / self.car.g)
        if "ctx_track" in df and len(df):
            track = df["ctx_track"].dropna()
            if not track.empty:
                out["track"] = str(track.iloc[0])
        if "ctx_pit_stop_count" in df and len(df):
            pc = pd.to_numeric(df["ctx_pit_stop_count"], errors="coerce")
            pc = pc.dropna()
            if not pc.empty:
                p_start = int(pc.iloc[0])
                p_end = int(pc.iloc[-1])
                out["pit_count_end"] = p_end
                out["pit_in_lap"] = bool(p_end > p_start)
        return out

    # ------------------------------------------------------------------
    # Channel inventory & distance-aligned sampling (MoTeC-style plots)
    # ------------------------------------------------------------------

    @cached_property
    def channels(self) -> list[ChannelInfo]:
        """All channels available on the enriched DataFrame, with metadata."""
        return [channel_info(c) for c in self.enriched.columns]

    def channels_by_group(self) -> dict[str, list[ChannelInfo]]:
        """Group :attr:`channels` by ``ChannelInfo.group`` for the UI tree."""
        return channels_by_group(list(self.enriched.columns))

    def distance_grid_m(self, n_points: int = 1000) -> np.ndarray:
        """Uniform distance axis from 0 to lap distance (length ``n_points``)."""
        if "current_lap_dist_m" not in self.raw.columns or self.raw.empty:
            return np.zeros(0, dtype=float)
        d = pd.to_numeric(self.raw["current_lap_dist_m"], errors="coerce")
        d = d.dropna().to_numpy()
        if d.size < 2:
            return np.zeros(0, dtype=float)
        return np.linspace(0.0, float(d.max() - d.min()), int(n_points))

    def sectors(
        self,
        *,
        boundaries_m: list[float] | None = None,
        n_equal: int = 3,
    ) -> list:
        """Return per-sector :class:`Sector` records for this lap.

        Thin wrapper around :func:`lfs_telemetry.telemetry.sectors.lap_sectors`.
        Defaults to a 3-sector equal-distance split.
        """
        from .sectors import lap_sectors  # local import avoids cycle
        return lap_sectors(
            self, boundaries_m=boundaries_m, n_equal=n_equal)

    def track_map(self, *, n_points: int = 1000):
        """Return a :class:`TrackMap` built from this lap's pos_x/pos_y."""
        from .track_map import TrackMap  # local import avoids cycle
        return TrackMap.from_lap(self, n_points=n_points)

    def channel_vs_distance(
        self,
        column: str,
        *,
        n_points: int = 1000,
        enriched: bool = True,
    ) -> pd.Series:
        """Distance-aligned ``column`` values, indexed by metres.

        Returns a :class:`pandas.Series` with name ``column`` and an
        index named ``distance_m``. NaN-filled if the column is absent.
        """
        d_grid = self.distance_grid_m(n_points=n_points)
        df = self.enriched if enriched else self.raw
        if d_grid.size == 0 or column not in df.columns:
            return pd.Series(
                np.full(d_grid.size, np.nan),
                index=pd.Index(d_grid, name="distance_m"),
                name=column,
            )
        d_full = pd.to_numeric(self.raw["current_lap_dist_m"], errors="coerce").to_numpy()
        y = pd.to_numeric(df[column], errors="coerce").to_numpy()
        mask = np.isfinite(d_full) & np.isfinite(y)
        if mask.sum() < 2:
            return pd.Series(
                np.full(d_grid.size, np.nan),
                index=pd.Index(d_grid, name="distance_m"),
                name=column,
            )
        d_full = d_full[mask] - d_full[mask][0]
        y = y[mask]
        order = np.argsort(d_full, kind="stable")
        d_full = d_full[order]
        y = y[order]
        keep = np.concatenate(([True], np.diff(d_full) > 0))
        values = np.interp(d_grid, d_full[keep], y[keep])
        return pd.Series(
            values,
            index=pd.Index(d_grid, name="distance_m"),
            name=column,
        )


def _resolve_car(car: CarSpec | str | None, df: pd.DataFrame) -> CarSpec:
    if isinstance(car, CarSpec):
        return car
    if isinstance(car, str):
        return car_spec_for(car)
    name = ""
    if "car" in df.columns and len(df):
        s = df["car"].dropna()
        if not s.empty:
            name = str(s.iloc[0])
    return car_spec_for(name)
