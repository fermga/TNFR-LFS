"""Generate per-track overview plots and CSV profiles for every LFS variant.

Usage:
    python scripts/track_view.py BL1               # one variant
    python scripts/track_view.py --all             # every PTH in C:/LFS/data/smx
    python scripts/track_view.py BL1 --out tracks  # custom output dir

Outputs (under <out_dir>/):
    <NAME>_overview.png      — track map (slope-coloured) + elevation +
                                curvature + width panels
    <NAME>_profile.csv       — per-node geometry table
    _summary.csv             — one row per variant when running --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfs_telemetry.telemetry.track.pth import (
    DEFAULT_SMX_DIR,
    compute_profile,
    list_path_files,
    parse_pth,
)


def profile_to_df(prof) -> pd.DataFrame:
    return pd.DataFrame({
        "s_m": prof.s,
        "x_m": prof.pos[:, 0],
        "y_m": prof.pos[:, 1],
        "z_m": prof.pos[:, 2],
        "dir_x": prof.direction[:, 0],
        "dir_y": prof.direction[:, 1],
        "dir_z": prof.direction[:, 2],
        "heading_rad": prof.heading_rad,
        "slope_pct": prof.slope_pct,
        "curvature_1_per_m": prof.curvature_1_per_m,
        "radius_m": prof.radius_m,
        "width_m": prof.width,
    })


def plot_overview(prof, out_path: Path, title: str | None = None) -> None:
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.4, 1, 1])

    # --- track map coloured by slope ---
    ax_map = fig.add_subplot(gs[0, :])
    sc = ax_map.scatter(prof.pos[:, 0], prof.pos[:, 1],
                        c=prof.slope_pct, cmap="RdBu_r",
                        s=6, vmin=-10, vmax=10)
    plt.colorbar(sc, ax=ax_map, label="slope %  (red = uphill)")
    # Mark start (S) and finish-direction arrow.
    ax_map.plot(prof.pos[0, 0], prof.pos[0, 1], "go", ms=10, label="start")
    ax_map.plot(prof.pos[-1, 0], prof.pos[-1, 1], "rs", ms=8, label="end")
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("X east (m)")
    ax_map.set_ylabel("Y north (m)")
    ax_map.set_title(title or f"{prof.name} — track map")
    ax_map.grid(True, alpha=0.3)
    ax_map.legend(loc="upper right", fontsize=8)

    # --- elevation profile ---
    ax_z = fig.add_subplot(gs[1, 0])
    ax_z.plot(prof.s, prof.pos[:, 2], "k-", lw=1.2)
    ax_z.set_xlabel("distance s (m)")
    ax_z.set_ylabel("elevation z (m)")
    ax_z.set_title("Elevation profile")
    ax_z.grid(True, alpha=0.3)

    # --- slope profile ---
    ax_sl = fig.add_subplot(gs[1, 1])
    ax_sl.plot(prof.s, prof.slope_pct, color="tab:red", lw=1.0)
    ax_sl.axhline(0, color="grey", lw=0.5)
    ax_sl.set_xlabel("distance s (m)")
    ax_sl.set_ylabel("slope %")
    ax_sl.set_title("Slope")
    ax_sl.grid(True, alpha=0.3)

    # --- curvature ---
    ax_k = fig.add_subplot(gs[2, 0])
    # Plot as 1000/R for readability (signed)
    safe_r = np.where(np.abs(prof.radius_m) > 1.0, prof.radius_m, np.inf)
    inv_r = np.sign(prof.curvature_1_per_m) * (1000.0 / safe_r)
    ax_k.plot(prof.s, inv_r, color="tab:purple", lw=1.0)
    ax_k.axhline(0, color="grey", lw=0.5)
    ax_k.set_xlabel("distance s (m)")
    ax_k.set_ylabel("1000/R  (left+ / right-)")
    ax_k.set_title("Curvature")
    ax_k.grid(True, alpha=0.3)

    # --- width ---
    ax_w = fig.add_subplot(gs[2, 1])
    ax_w.plot(prof.s, prof.width, color="tab:green", lw=1.0)
    ax_w.set_xlabel("distance s (m)")
    ax_w.set_ylabel("width (m)")
    ax_w.set_title("Path width (PTH trailing scalar)")
    ax_w.grid(True, alpha=0.3)

    # global header
    L = prof.total_length_m
    zmin, zmax = prof.elevation_range_m
    fig.suptitle(
        f"{prof.name}  —  L≈{L:.0f} m,  Δz={zmax - zmin:.1f} m,  "
        f"Rmin={float(np.min(prof.radius_m)):.0f} m,  "
        f"slope_max={float(np.max(np.abs(prof.slope_pct))):.1f}%",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def process_one(name_or_file: str, out_dir: Path,
                smx_dir: Path = DEFAULT_SMX_DIR) -> dict:
    if Path(name_or_file).suffix.lower() == ".pth":
        pth_file = Path(name_or_file)
    else:
        pth_file = smx_dir / f"{name_or_file.upper()}.pth"
    if not pth_file.exists():
        raise FileNotFoundError(pth_file)

    p = parse_pth(pth_file)
    if p.num_nodes == 0:
        print(f"  {p.name}: empty PTH, skipped")
        return {"variant": p.name, "nodes": 0, "length_m": 0.0}

    prof = compute_profile(p)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{p.name}_profile.csv"
    profile_to_df(prof).to_csv(csv_path, index=False)

    png_path = out_dir / f"{p.name}_overview.png"
    plot_overview(prof, png_path)

    L = prof.total_length_m
    print(f"  {p.name}: nodes={p.num_nodes}  L={L:7.0f} m  "
          f"Rmin={float(np.min(prof.radius_m)):5.0f} m  "
          f"slope±={float(np.max(np.abs(prof.slope_pct))):4.1f}%  "
          f"-> {png_path.name}")
    return {
        "variant": p.name,
        "nodes": p.num_nodes,
        "length_m": L,
        "elev_min_m": prof.elevation_range_m[0],
        "elev_max_m": prof.elevation_range_m[1],
        "min_radius_m": float(np.min(prof.radius_m)),
        "max_slope_pct": float(np.max(np.abs(prof.slope_pct))),
        "mean_width_m": float(np.mean(prof.width)) if prof.width.size else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variant", nargs="?", help="track variant (e.g. BL1) or path to .pth file")
    ap.add_argument("--all", action="store_true", help="process every PTH in --smx-dir")
    ap.add_argument("--smx-dir", default=str(DEFAULT_SMX_DIR),
                    help=f"LFS smx directory (default: {DEFAULT_SMX_DIR})")
    ap.add_argument("--out", default="tracks",
                    help="output directory (default: tracks/)")
    args = ap.parse_args(argv)

    smx_dir = Path(args.smx_dir)
    out_dir = Path(args.out)

    if args.all:
        files = list_path_files(smx_dir)
        print(f"Processing {len(files)} PTH files from {smx_dir} -> {out_dir}/")
        rows = []
        for f in files:
            try:
                rows.append(process_one(str(f), out_dir, smx_dir))
            except Exception as exc:
                print(f"  {f.stem}: ERROR {exc}")
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / "_summary.csv", index=False)
            print(f"\nSummary -> {out_dir / '_summary.csv'}")
        return 0

    if not args.variant:
        ap.error("provide a variant name (e.g. BL1) or use --all")
    process_one(args.variant, out_dir, smx_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
