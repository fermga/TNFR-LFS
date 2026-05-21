"""Multi-lap stint aggregation for session-level analysis.

Where :class:`lfs_telemetry.telemetry.lap.LapTelemetry` looks at *one* lap,
:class:`StintTelemetry` looks at *N* laps from the same session and
exposes the trends a race engineer cares about:

* lap-time progression (pace, drop-off slope),
* fuel use per lap (mean, stddev, projected laps remaining),
* per-wheel tyre work accumulation (proxy for thermal / wear load),
* per-wheel friction-circle usage trend (saturation drift = grip loss),
* peak G evolution,
* aid usage per lap (TC / ABS active fraction).

Inputs
------

A stint can be built from any of:

* a list of :class:`LapTelemetry`,
* a list of CSV file paths (each one a per-lap CSV),
* a directory + glob (e.g. ``StintTelemetry.from_dir("captures",
  "stint_bl1_fbm_lap*.csv")``).

The lap order is the order of the inputs (sorted lexically when using
``from_dir``). One row per lap is kept in :attr:`per_lap` (a
:class:`pandas.DataFrame`); :attr:`trends` exposes scalar fits
(slopes, means).

This module imports only :mod:`pandas` + :mod:`numpy` on top of the
existing telemetry layer — pure pandas/numpy — so it is safe
to ship inside the future standalone visualization app.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .lap import LapTelemetry
from .observables import CarSpec
from .protocol.packets import WHEEL_ORDER


@dataclass
class StintTelemetry:
    """A consecutive set of useful laps from one session."""

    laps: list[LapTelemetry]
    invalid_lap_indices: set[int] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_csvs(
        cls,
        paths: Iterable[str | Path],
        *,
        car: CarSpec | str | None = None,
    ) -> StintTelemetry:
        """Load a stint from a list of per-lap CSVs (in order)."""
        laps = [LapTelemetry.from_csv(p, car=car) for p in paths]
        return cls(laps=laps)

    @classmethod
    def from_dir(
        cls,
        directory: str | Path,
        pattern: str = "*lap*.csv",
        *,
        car: CarSpec | str | None = None,
    ) -> StintTelemetry:
        """Load all CSVs in ``directory`` matching ``pattern`` (sorted)."""
        directory = Path(directory)
        paths = sorted(directory.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"no CSVs matching {pattern!r} in {directory}")
        return cls.from_csvs(paths, car=car)

    @classmethod
    def from_laps(cls, laps: Sequence[LapTelemetry]) -> StintTelemetry:
        """Wrap an existing list of LapTelemetry (no I/O)."""
        return cls(laps=list(laps))

    # ------------------------------------------------------------------
    # Aggregated views
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.laps)

    @cached_property
    def per_lap(self) -> pd.DataFrame:
        """One row per lap with the metrics needed for stint trends."""
        if not self.laps:
            return pd.DataFrame()
        rows = [_lap_metrics(i + 1, lap) for i, lap in enumerate(self.laps)]
        return pd.DataFrame(rows)

    @cached_property
    def trends(self) -> dict[str, Any]:
        """Scalar trend summary for the whole stint.

        Race-start laps (``is_race_start == True``) are excluded from
        every aggregate (means, slopes, projections) because the launch
        from grid is not comparable to a flying lap; they are still
        kept in :attr:`per_lap` with the flag set so the UI can show
        them. ``num_laps`` reports the count used for trends; the raw
        slice count is exposed as ``num_laps_total`` and the excluded
        indices as ``excluded_lap_indices``.
        """
        full = self.per_lap
        if full.empty:
            return {}
        if "is_race_start" in full.columns:
            mask = ~full["is_race_start"].fillna(False).astype(bool)
            df = full.loc[mask].reset_index(drop=True)
            excluded = full.loc[~mask, "lap_index"].astype(int).tolist()
        else:
            df = full
            excluded = []

        out: dict[str, Any] = {
            "num_laps_total": len(full),
            "num_laps": len(df),
            "excluded_lap_indices": excluded,
            "car": str(full["car"].iloc[0]) if "car" in full else None,
            "track": str(full["track"].iloc[0]) if "track" in full and full["track"].notna().any() else None,
        }
        if df.empty:
            return out

        # Pace: mean / best / drop-off slope (s per lap).
        if "lap_time_s" in df:
            t = df["lap_time_s"].dropna()
            if not t.empty:
                out["lap_time_mean_s"] = float(t.mean())
                out["lap_time_best_s"] = float(t.min())
                out["lap_time_stdev_s"] = float(t.std(ddof=0))
                out["pace_dropoff_s_per_lap"] = _slope(df["lap_index"], df["lap_time_s"])

        # Fuel: mean per-lap consumption; projected laps remaining at last reading.
        if "fuel_pct_used" in df:
            f = df["fuel_pct_used"].dropna()
            if not f.empty:
                mean_use = float(f.mean())
                out["fuel_pct_per_lap_mean"] = mean_use
                out["fuel_pct_per_lap_stdev"] = float(f.std(ddof=0))
                last_fuel = df["fuel_pct_end"].dropna()
                if not last_fuel.empty and mean_use > 1e-6:
                    out["fuel_laps_remaining"] = float(last_fuel.iloc[-1] / mean_use)

        # Tyre work integral progression — slope per wheel (W·s gained per lap).
        for c in WHEEL_ORDER:
            col = f"tyre_work_kj_{c}"
            if col in df:
                out[f"tyre_work_slope_kj_per_lap_{c}"] = _slope(df["lap_index"], df[col])

        # Friction-use saturation trend — slope per wheel.
        for c in WHEEL_ORDER:
            col = f"friction_use_p95_{c}"
            if col in df:
                out[f"friction_use_slope_per_lap_{c}"] = _slope(df["lap_index"], df[col])

        # Per-wheel grip index trend (100 = high grip headroom,
        # 0 = low grip headroom / saturated tyre behavior).
        for c in WHEEL_ORDER:
            col = f"grip_idx_{c}"
            if col in df:
                out[f"grip_idx_mean_{c}"] = float(df[col].mean())
                out[f"grip_idx_slope_per_lap_{c}"] = _slope(
                    df["lap_index"], df[col]
                )

        # Aid usage trend (TC/ABS fraction per lap).
        for col in ("tc_active_fraction", "abs_active_fraction"):
            if col in df:
                out[f"{col}_mean"] = float(df[col].mean())
                out[f"{col}_slope_per_lap"] = _slope(df["lap_index"], df[col])

        # Peak G evolution.
        for col in ("peak_long_g", "peak_lat_g"):
            if col in df:
                out[f"{col}_mean"] = float(df[col].mean())
                out[f"{col}_slope_per_lap"] = _slope(df["lap_index"], df[col])

        return out

    def to_csv(self, path: str | Path) -> Path:
        """Persist ``per_lap`` to a CSV (one row per lap)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.per_lap.to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------
    # MoTeC-style helpers: theoretical best lap + track map
    # ------------------------------------------------------------------

    def sector_times_per_lap(
        self,
        *,
        boundaries_m: Sequence[float] | None = None,
        n_equal: int = 3,
    ) -> pd.DataFrame:
        """Per-lap sector times.

        Returns a DataFrame indexed by ``lap_index`` with one column
        per sector (``sector_1_s``, ``sector_2_s``, …) plus
        ``is_race_start``. Race-start laps are kept; downstream
        consumers (e.g. :meth:`theoretical_best_lap`) skip them.
        """
        from .sectors import lap_sectors  # local import avoids cycle
        rows: list[dict[str, Any]] = []
        for i, lap in enumerate(self.laps, start=1):
            secs = lap_sectors(
                lap, boundaries_m=boundaries_m, n_equal=n_equal)
            row: dict[str, Any] = {
                "lap_index": i,
                "is_race_start": bool(lap.is_race_start),
            }
            for s in secs:
                row[f"sector_{s.index + 1}_s"] = s.time_s
            rows.append(row)
        if not rows:
            return pd.DataFrame(columns=["is_race_start"]).rename_axis("lap_index")
        return pd.DataFrame(rows).set_index("lap_index")

    def theoretical_best_lap(
        self,
        *,
        boundaries_m: Sequence[float] | None = None,
        n_equal: int = 3,
    ) -> dict[str, Any]:
        """Sum of best sector times across the stint.

        Race-start laps are excluded (their sectors include the launch
        from grid, which is not comparable to a flying lap).

        Returns a dict with::

            {
              "theoretical_best_s": float,
              "actual_best_s": float,
              "gap_s": float,                 # actual - theoretical
              "best_sector_s": [s1, s2, ...], # the chosen min per sector
              "best_sector_lap": [lap_idx, lap_idx, ...],
              "n_sectors": int,
              "n_laps_used": int,
              "excluded_lap_indices": list[int],
            }

        Empty dict if no usable laps.
        """
        df = self.sector_times_per_lap(
            boundaries_m=boundaries_m, n_equal=n_equal)
        if df.empty:
            return {}
        excluded = df.index[df["is_race_start"]].astype(int).tolist()
        usable = df.loc[~df["is_race_start"]].drop(columns=["is_race_start"])
        if usable.empty:
            return {
                "theoretical_best_s": float("nan"),
                "actual_best_s": float("nan"),
                "gap_s": float("nan"),
                "best_sector_s": [],
                "best_sector_lap": [],
                "n_sectors": 0,
                "n_laps_used": 0,
                "excluded_lap_indices": excluded,
            }
        sec_cols = [c for c in usable.columns if c.startswith("sector_")]
        if not sec_cols:
            return {}
        best_s = [float(usable[c].min()) for c in sec_cols]
        best_lap = [int(usable[c].idxmin()) for c in sec_cols]
        theoretical = float(np.nansum(best_s))
        actual_best = float(usable[sec_cols].sum(axis=1).min())
        return {
            "theoretical_best_s": theoretical,
            "actual_best_s": actual_best,
            "gap_s": float(actual_best - theoretical),
            "best_sector_s": best_s,
            "best_sector_lap": best_lap,
            "n_sectors": len(sec_cols),
            "n_laps_used": len(usable),
            "excluded_lap_indices": excluded,
        }

    def track_map(self, *, n_points: int = 1000):
        """Build an averaged :class:`TrackMap` across all flying laps.

        Race-start laps are excluded so the launch from grid doesn't
        skew the averaged racing line. Falls back to all laps if every
        lap is a race start.
        """
        from .track_map import TrackMap  # local import avoids cycle
        flying = [lap for lap in self.laps if not lap.is_race_start]
        source = flying or self.laps
        return TrackMap.from_laps(source, n_points=n_points)

    # ------------------------------------------------------------------
    # Validity / averages (Detect&Monitor-style)
    # ------------------------------------------------------------------

    def mark_lap_invalid(self, lap_index: int) -> None:
        """Flag ``lap_index`` (1-based) as invalid for ``clean`` averages."""
        self.invalid_lap_indices.add(int(lap_index))

    def mark_invalid_from_records(self, records) -> None:
        """Apply ``valid``/``obh_count`` from a list of LapRecord.

        Lap indices are aligned positionally with ``self.laps``: record[i]
        flags ``self.laps[i]``. Laps with ``valid is False`` or any
        object hit (OBH) are marked invalid.
        """
        for i, rec in enumerate(records, start=1):
            if i > len(self.laps):
                break
            if not getattr(rec, "valid", True) or getattr(rec, "obh_count", 0) > 0:
                self.invalid_lap_indices.add(i)

    @cached_property
    def race_start_lap_indices(self) -> frozenset[int]:
        """1-based indices of laps flagged as race-start (launch from grid).

        Cached because lap membership is fixed for a given
        :class:`StintTelemetry` instance.
        """
        return frozenset(
            i for i, lap in enumerate(self.laps, start=1)
            if lap.is_race_start
        )

    @property
    def clean_lap_indices(self) -> list[int]:
        """Flying-only, validity-filtered 1-based lap indices.

        Excludes race-start laps and any lap explicitly marked invalid
        (via :meth:`mark_lap_invalid` or :meth:`mark_invalid_from_records`).
        Recomputed on every call — cheap, and reflects mutations to
        :attr:`invalid_lap_indices`.
        """
        bad = self.invalid_lap_indices | self.race_start_lap_indices
        return [i for i in range(1, len(self.laps) + 1) if i not in bad]

    def average_lap_time(
        self,
        mode: str = "stint",
        *,
        rolling: int | None = None,
    ) -> float | None:
        """Mean lap time in seconds for the requested mode.

        Modes (matches Detect&Monitor's four buttons):

        * ``"stint"`` — all flying laps (excludes race-start). Default.
        * ``"clean"`` — flying minus laps marked invalid (HLV / OBH).
        * ``"total"`` — every lap including the race-start launch.
        * ``"rolling"`` — last ``rolling`` *clean* laps. Requires the
          ``rolling`` argument; falls back to all clean laps if smaller.

        Returns ``None`` if the selected slice is empty.
        """
        full = self.per_lap
        if full.empty or "lap_time_s" not in full.columns:
            return None
        if mode == "total":
            sel = full
        elif mode == "stint":
            sel = full[~full["lap_index"].isin(self.race_start_lap_indices)]
        elif mode == "clean":
            sel = full[full["lap_index"].isin(self.clean_lap_indices)]
        elif mode == "rolling":
            if rolling is None or rolling <= 0:
                raise ValueError("rolling mode requires rolling > 0")
            sel = full[full["lap_index"].isin(self.clean_lap_indices)].tail(rolling)
        else:
            raise ValueError(
                f"unknown mode {mode!r} "
                "(expected stint/clean/total/rolling)")
        t = sel["lap_time_s"].dropna()
        return float(t.mean()) if not t.empty else None

    @cached_property
    def fuel_usage(self) -> dict[str, Any]:
        """Fuel-consumption summary across the stint.

        Mirrors what an in-cockpit overlay needs:

        * ``per_lap_pct`` — list of per-lap fuel deltas (%); race-start
          and warmup laps included in the list but flagged via index.
        * ``mean_pct`` / ``stdev_pct`` — across flying laps only.
        * ``last_fuel_pct`` — fuel level at the end of the last lap.
        * ``laps_remaining`` — last_fuel_pct / mean_pct (None if no
          consumption observed yet).
        """
        full = self.per_lap
        if full.empty or "fuel_pct_used" not in full.columns:
            return {}
        per_lap = full["fuel_pct_used"].tolist()
        flying_mask = ~full["lap_index"].isin(self.race_start_lap_indices)
        flying_use = full.loc[flying_mask, "fuel_pct_used"].dropna()
        out: dict[str, Any] = {
            "per_lap_pct": per_lap,
            "mean_pct": (float(flying_use.mean())
                         if not flying_use.empty else None),
            "stdev_pct": (float(flying_use.std(ddof=0))
                          if len(flying_use) > 1 else None),
        }
        if "fuel_pct_end" in full.columns:
            last = full["fuel_pct_end"].dropna()
            out["last_fuel_pct"] = float(last.iloc[-1]) if not last.empty else None
        else:
            out["last_fuel_pct"] = None
        mean_use = out["mean_pct"]
        last_fuel = out["last_fuel_pct"]
        if mean_use and mean_use > 1e-6 and last_fuel is not None:
            out["laps_remaining"] = float(last_fuel / mean_use)
        else:
            out["laps_remaining"] = None
        return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _slope(x: pd.Series, y: pd.Series) -> float:
    """Linear regression slope; NaN on insufficient / degenerate data."""
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    mask = xv.notna() & yv.notna()
    if mask.sum() < 2:
        return float("nan")
    xa = xv[mask].to_numpy()
    ya = yv[mask].to_numpy()
    if np.ptp(xa) == 0:
        return float("nan")
    return float(np.polyfit(xa, ya, 1)[0])


def _lap_metrics(idx: int, lap: LapTelemetry) -> dict[str, Any]:
    """Extract one tidy row for ``per_lap``."""
    raw = lap.raw
    enr = lap.enriched
    s = lap.summary
    row: dict[str, Any] = {
        "lap_index": idx,
        "car": s.get("car"),
        "track": s.get("track"),
        "lap_time_s": s.get("lap_time_s"),
        "distance_m": s.get("distance_m"),
        "top_speed_kmh": s.get("top_speed_kmh"),
        "peak_long_g": s.get("peak_long_g"),
        "peak_lat_g": s.get("peak_lat_g"),
        "is_race_start": bool(s.get("is_race_start", False)),
        "pit_in_lap": bool(s.get("pit_in_lap", False)),
        "pit_count_end": s.get("pit_count_end"),
    }

    # Fuel: start/end + used. Schema column 'fuel' is a 0..1 fraction.
    if "fuel" in raw and len(raw):
        f0 = float(raw["fuel"].iloc[0])
        f1 = float(raw["fuel"].iloc[-1])
        row["fuel_pct_start"] = f0 * 100.0
        row["fuel_pct_end"] = f1 * 100.0
        row["fuel_pct_used"] = max(0.0, (f0 - f1) * 100.0)

    # Sample dt for time-integrated quantities.
    if "time_ms" in raw and len(raw) > 1:
        dt_s = float((raw["time_ms"].iloc[-1] - raw["time_ms"].iloc[0]) / 1000.0
                     / max(1, len(raw) - 1))
    else:
        dt_s = 0.01

    # Per-wheel tyre work: integrate W → kJ over the lap.
    for c in WHEEL_ORDER:
        col = f"tyre_work_w_{c}"
        if col in enr:
            kj = float(np.nansum(enr[col].to_numpy()) * dt_s / 1000.0)
            row[f"tyre_work_kj_{c}"] = kj

    # Per-wheel friction-circle utilisation summary.
    for c in WHEEL_ORDER:
        col = f"friction_use_{c}"
        if col in enr:
            arr = enr[col].to_numpy()
            arr = arr[np.isfinite(arr)]
            if arr.size:
                row[f"friction_use_p95_{c}"] = float(np.percentile(arr, 95))
                row[f"friction_use_max_{c}"] = float(arr.max())

    # Per-wheel slip summaries + grip index (0..100, high=better).
    for c in WHEEL_ORDER:
        slip_ratio_col = f"wheel_{c}_slip_ratio"
        slip_frac_col = f"wheel_{c}_slip_fraction"
        tan_slip_col = f"wheel_{c}_tan_slip_angle"
        temp_end = row.get(f"tyre_temp_end_c_{c}")
        fric_p95 = row.get(f"friction_use_p95_{c}")

        slip_ratio_p95 = np.nan
        if slip_ratio_col in raw and len(raw):
            arr = np.abs(pd.to_numeric(raw[slip_ratio_col], errors="coerce"))
            arr = arr.dropna()
            if not arr.empty:
                slip_ratio_p95 = float(np.percentile(arr.to_numpy(), 95))
                row[f"slip_ratio_p95_{c}"] = slip_ratio_p95

        slip_frac_p95 = np.nan
        if slip_frac_col in raw and len(raw):
            arr = np.abs(pd.to_numeric(raw[slip_frac_col], errors="coerce"))
            arr = arr.dropna()
            if not arr.empty:
                slip_frac_p95 = float(np.percentile(arr.to_numpy(), 95))
                row[f"slip_fraction_p95_{c}"] = slip_frac_p95

        tan_slip_p95 = np.nan
        if tan_slip_col in raw and len(raw):
            arr = np.abs(pd.to_numeric(raw[tan_slip_col], errors="coerce"))
            arr = arr.dropna()
            if not arr.empty:
                tan_slip_p95 = float(np.percentile(arr.to_numpy(), 95))
                row[f"tan_slip_p95_{c}"] = tan_slip_p95

        # Grip proxy from multiple tyre stress indicators.
        risk_terms: list[float] = []
        if isinstance(fric_p95, (int, float)) and np.isfinite(fric_p95):
            risk_terms.append(min(1.0, max(0.0, (float(fric_p95) - 0.78) / 0.25)))
        if np.isfinite(slip_ratio_p95):
            risk_terms.append(min(1.0, max(0.0, (float(slip_ratio_p95) - 0.08) / 0.14)))
        if np.isfinite(slip_frac_p95):
            risk_terms.append(min(1.0, max(0.0, (float(slip_frac_p95) - 0.10) / 0.25)))
        if np.isfinite(tan_slip_p95):
            risk_terms.append(min(1.0, max(0.0, (float(tan_slip_p95) - 0.07) / 0.12)))
        if isinstance(temp_end, (int, float)) and np.isfinite(float(temp_end)):
            risk_terms.append(min(1.0, max(0.0, (float(temp_end) - 102.0) / 28.0)))

        if risk_terms:
            # weighted mean (front-load tyre-dynamics terms over temp).
            if len(risk_terms) >= 5:
                w = np.array([0.30, 0.24, 0.20, 0.16, 0.10], dtype=float)
                risk = float(np.average(np.array(risk_terms[:5]), weights=w))
            else:
                risk = float(np.mean(risk_terms))
            row[f"grip_idx_{c}"] = max(0.0, min(100.0, (1.0 - risk) * 100.0))

    # Per-wheel tyre temperature: mean over the lap + end-of-lap value.
    # Source channel is OutSim's per-wheel air temperature (`wheel_{c}_air_temp_c`).
    for c in WHEEL_ORDER:
        col = f"wheel_{c}_air_temp_c"
        if col in raw and len(raw):
            arr = pd.to_numeric(raw[col], errors="coerce").dropna()
            if not arr.empty:
                row[f"tyre_temp_mean_c_{c}"] = float(arr.mean())
                row[f"tyre_temp_max_c_{c}"] = float(arr.max())
                row[f"tyre_temp_end_c_{c}"] = float(arr.iloc[-1])

    # Per-wheel peak vertical load (proxy for kerb hits / chassis stress).
    for c in WHEEL_ORDER:
        col = f"wheel_{c}_vertical_load_n"
        if col in raw and len(raw):
            arr = pd.to_numeric(raw[col], errors="coerce").dropna()
            if not arr.empty:
                row[f"vert_load_max_n_{c}"] = float(arr.max())

    # Brake bias mean across the lap (only braking samples).
    if "brake_bias_front_real" in enr:
        bb = enr["brake_bias_front_real"].dropna()
        if not bb.empty:
            row["brake_bias_front_mean"] = float(bb.mean())

    # Aid active fraction (TC / ABS).
    for src, key in (("dl_tc_active", "tc_active_fraction"),
                     ("dl_abs_active", "abs_active_fraction")):
        if src in enr:
            v = enr[src].astype(float)
            row[key] = float(v.mean()) if len(v) else float("nan")

    # FFB clip fraction (>=0.99 means pegged).
    if "ffb_load_pct" in enr:
        ffb = enr["ffb_load_pct"].abs()
        row["ffb_clip_fraction"] = float((ffb >= 0.99).mean()) if len(ffb) else float("nan")

    # Engine / oil endpoint values (degradation / heat soak indicators).
    if "eng_temp_c" in raw and len(raw):
        row["eng_temp_max_c"] = float(raw["eng_temp_c"].max())
        row["eng_temp_end_c"] = float(raw["eng_temp_c"].iloc[-1])
    if "oil_temp_c" in raw and len(raw):
        row["oil_temp_max_c"] = float(raw["oil_temp_c"].max())
        row["oil_temp_end_c"] = float(raw["oil_temp_c"].iloc[-1])

    return row
