"""Smoke-check sectors / theoretical-best / track-map on real BL1 FBM stint."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lfs_telemetry.telemetry import LapTelemetry, StintTelemetry, TrackMap
from lfs_telemetry.telemetry.comparison import _unwrapped_lap_arrays

CAP_DIR = Path("captures")
FILES = sorted(CAP_DIR.glob("stint_20260514-165127_FBM_BL1_lap*.csv"))


def _load_lap(p: Path) -> LapTelemetry:
    df = pd.read_csv(p, comment="#")
    return LapTelemetry.from_dataframe(df, car="FBM")


def main() -> None:
    print(f"Found {len(FILES)} lap CSVs\n")

    print("=== A. Canonical track geometry from LFS PTH (BL1) ===")
    pth_map = TrackMap.from_pth("BL1")
    pb = pth_map.bounds()
    print(f"  PTH length_m = {pth_map.length_m:.1f}  nodes = {pth_map.n_points}")
    print(f"  PTH bounds:   x=[{pb.x_min:.1f},{pb.x_max:.1f}]  y=[{pb.y_min:.1f},{pb.y_max:.1f}]")
    print(f"               width={pb.width_m:.1f} height={pb.height_m:.1f}\n")
    track_length_m = pth_map.length_m

    laps = [_load_lap(p) for p in FILES]

    print("=== B. Per-lap CSV inspection vs PTH ground truth ===")
    for i, lap in enumerate(laps, start=1):
        s = lap.summary
        d_raw = pd.to_numeric(lap.raw["current_lap_dist_m"], errors="coerce").to_numpy()
        _idx, d_unw, _t_unw = _unwrapped_lap_arrays(lap)
        post_line = d_unw[d_unw >= 0]
        coverage_pct = (post_line.max() / track_length_m * 100.0) if post_line.size else 0.0
        print(f"  lap {i}:  duration={s.get('lap_time_s'):.3f}s  "
              f"raw d=[{d_raw.min():.1f},{d_raw.max():.1f}]m  "
              f"unwrapped post-line=[0,{post_line.max():.1f}]m  "
              f"coverage={coverage_pct:.1f}% of PTH track")
    print()

    stint = StintTelemetry.from_laps(laps)

    print("=== C. Per-lap sectors (n_equal=3) ===")
    secdf = stint.sector_times_per_lap(n_equal=3)
    print(secdf.to_string())
    print()

    print("=== D. Theoretical best lap ===")
    tb = stint.theoretical_best_lap(n_equal=3)
    print(json.dumps(tb, indent=2, default=float))
    print()

    print("=== E. Track map averaged from telemetry (post-line common window) ===")
    tmap = stint.track_map(n_points=200)
    b = tmap.bounds()
    print(f"  n_points={tmap.n_points}  length_m={tmap.length_m:.1f}  "
          f"({tmap.length_m / track_length_m * 100.0:.1f}% of PTH)")
    print(f"  bounds: x=[{b.x_min:.1f},{b.x_max:.1f}] y=[{b.y_min:.1f},{b.y_max:.1f}]\n")

    print("=== F. Verdict ===")
    sum(1 for lap in laps
                if pd.to_numeric(lap.raw["current_lap_dist_m"],
                                 errors="coerce").max() < 0.95 * track_length_m)
    if all(pd.to_numeric(lap.raw["current_lap_dist_m"],
                         errors="coerce").max() >= 0.95 * track_length_m
           for lap in laps):
        # raw d covers full track at least once → good
        pass
    print(f"  Captures cover full track (raw d reaches >= 95% of PTH): "
          f"{all(pd.to_numeric(lap.raw['current_lap_dist_m'], errors='coerce').max() >= 0.95 * track_length_m for lap in laps)}")
    print("  But each CSV is a 75s sliding window containing ONE line crossing,")
    print("  so the 'post-line' window per lap is partial. Sectors and theoretical")
    print("  best are computed only over that post-line slice → results are valid")
    print("  for that slice but NOT directly comparable across laps unless laps")
    print("  are stitched into proper line-to-line slices.\n")

    print("OK")


if __name__ == "__main__":
    main()
