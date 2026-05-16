"""Generate racing-line + target-speed maps for LFS tracks.

Usage:
    python scripts/racing_line_view.py BL1
    python scripts/racing_line_view.py --all --out racing_lines
    python scripts/racing_line_view.py FE2 --mu-lat 1.4 --mu-long 1.2

Outputs (under <out_dir>/):
    <NAME>_racing.png      — track edges + centerline + heuristic racing
                              line, coloured by target speed (km/h), plus
                              a v(s) sub-plot.
    <NAME>_racing.csv      — per-node table:
                                s_m, x_center_m, y_center_m, z_center_m,
                                x_line_m, y_line_m, offset_m,
                                heading_rad, curvature_1_per_m, radius_m,
                                slope_pct, width_m,
                                segment_id, segment_kind,
                                v_target_ms, v_target_kmh.

NOTE
----
The PTH centerline is the LFS AI driving line (geometric, not optimised).
The "racing line" plotted here is a heuristic outside-apex-outside path
through each turn segment; it is meant as a visual reference, not as a
minimum-curvature optimum.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfs_telemetry.telemetry.track.enrich import segment_track  # noqa: E402
from lfs_telemetry.telemetry.track.pin import (  # noqa: E402
    PinInfo,
    load_all as load_all_pins,
)
from lfs_telemetry.telemetry.track.pth import (  # noqa: E402
    DEFAULT_SMX_DIR,
    compute_profile,
    list_path_files,
    parse_pth,
)
from lfs_telemetry.telemetry.track.knw import (  # noqa: E402
    DEFAULT_KNW_DIR,
    load_for as load_knw_for,
)
from lfs_telemetry.telemetry.track.racing_line import (  # noqa: E402
    compute_edges,
    compute_geometric_line,
    compute_knw_line,
    compute_target_speed,
)


def _coloured_polyline(xy: np.ndarray, values: np.ndarray, *,
                       cmap="viridis", lw: float = 2.5,
                       vmin: float | None = None,
                       vmax: float | None = None) -> LineCollection:
    pts = xy.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, linewidth=lw)
    lc.set_array((values[:-1] + values[1:]) / 2.0)
    if vmin is not None:
        lc.set_clim(vmin=vmin)
    if vmax is not None:
        lc.set_clim(vmax=vmax)
    return lc


def render(profile, segments, line, v_target_ms, name: str,
           out_path: Path) -> None:
    left, right = compute_edges(profile)
    v_kmh = v_target_ms * 3.6

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.18)
    ax_map = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1])

    # --- Track edges + centerline -------------------------------------
    ax_map.plot(left[:, 0], left[:, 1], color="#777", lw=0.7,
                label="track edges")
    ax_map.plot(right[:, 0], right[:, 1], color="#777", lw=0.7)
    ax_map.plot(profile.pos[:, 0], profile.pos[:, 1],
                color="#bbb", lw=0.6, ls=":", label="centerline (PTH)")

    # --- Racing line coloured by target speed --------------------------
    lc = _coloured_polyline(line.line_xy, v_kmh, cmap="viridis", lw=3.0,
                            vmin=float(np.min(v_kmh)),
                            vmax=float(np.max(v_kmh)))
    ax_map.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax_map, fraction=0.035, pad=0.02)
    cbar.set_label("target speed (km/h)")

    # --- Mark apex of every turn segment -------------------------------
    for seg in segments:
        if seg.kind == "straight":
            continue
        mid = (seg.node_start + seg.node_end) // 2
        ax_map.plot(line.line_xy[mid, 0], line.line_xy[mid, 1],
                    "o", color="red", ms=5, mec="black", mew=0.5,
                    zorder=5)

    ax_map.set_aspect("equal", "datalim")
    ax_map.set_title(
        f"{name} — heuristic racing line "
        f"({len(segments)} segments, "
        f"v range {v_kmh.min():.0f}-{v_kmh.max():.0f} km/h)"
    )
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.legend(loc="best", fontsize=8)
    ax_map.grid(alpha=0.3)

    # --- Speed vs distance --------------------------------------------
    ax_v.plot(profile.s, v_kmh, color="#1f77b4", lw=1.4)
    ax_v.fill_between(profile.s, 0, v_kmh, color="#1f77b4", alpha=0.18)
    # mark turn segments as shaded background
    for seg in segments:
        if seg.kind == "straight":
            continue
        colour = "#ffd6d6" if seg.kind == "right" else "#d6e8ff"
        ax_v.axvspan(seg.s_start_m, seg.s_end_m, color=colour, alpha=0.4)
    ax_v.set_xlabel("s along path (m)")
    ax_v.set_ylabel("v_target (km/h)")
    ax_v.set_xlim(profile.s[0], profile.s[-1])
    ax_v.set_ylim(0, v_kmh.max() * 1.05)
    ax_v.grid(alpha=0.3)
    ax_v.set_title("blue band = left turn  •  red band = right turn")

    fig.suptitle(f"{name} target-speed reference (heuristic)",
                 fontsize=13, y=0.995)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def process_one(pth_file: Path, out_dir: Path, *,
                mu_lat: float, mu_long: float, v_cap_ms: float,
                straight_radius: float, min_segment_m: float,
                edge_margin_m: float,
                mu_lat_aero_k: float = 0.0,
                pin: PinInfo | None = None,
                line_source: str = "auto",
                knw_car: str = "FBM",
                knw_dir: Path = DEFAULT_KNW_DIR) -> dict:
    name = pth_file.stem.upper()
    track = parse_pth(pth_file)
    profile = compute_profile(track)
    if profile.s.size < 4:
        return {"variant": name, "status": "empty"}

    segments = segment_track(profile,
                             straight_radius_m=straight_radius,
                             min_segment_m=min_segment_m)

    # Canonical source: .knw AI knowledge file for ``knw_car`` if present;
    # otherwise the heuristic outside-apex-outside line.
    line_used = "heuristic"
    knw_info = None
    if line_source in ("auto", "knw"):
        knw_info = load_knw_for(name, knw_car, knw_dir=knw_dir)
        if knw_info is not None:
            line = compute_knw_line(profile, knw_info,
                                    edge_margin_m=edge_margin_m)
            line_used = f"knw:{knw_car}"
        elif line_source == "knw":
            return {"variant": name, "status": f"no-knw-for-{knw_car}"}
        else:
            line = compute_geometric_line(profile, segments,
                                          edge_margin_m=edge_margin_m)
    else:
        line = compute_geometric_line(profile, segments,
                                      edge_margin_m=edge_margin_m)
    v_target = compute_target_speed(profile,
                                    mu_lat=mu_lat, mu_long=mu_long,
                                    v_cap_ms=v_cap_ms,
                                    mu_lat_aero_k=mu_lat_aero_k)

    # Per-node segment lookup.
    n_nodes = int(profile.s.size)
    seg_id = np.full(n_nodes, -1, dtype=np.int64)
    seg_kind = np.full(n_nodes, "", dtype=object)
    for seg in segments:
        i0, i1 = seg.node_start, min(seg.node_end, n_nodes - 1)
        seg_id[i0:i1 + 1] = seg.index
        seg_kind[i0:i1 + 1] = seg.kind

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}_racing.png"
    csv = out_dir / f"{name}_racing.csv"
    render(profile, segments, line, v_target, name, png)

    df = pd.DataFrame({
        "s_m": profile.s,
        "x_center_m": profile.pos[:, 0],
        "y_center_m": profile.pos[:, 1],
        "z_center_m": profile.pos[:, 2],
        "x_line_m": line.line_xy[:, 0],
        "y_line_m": line.line_xy[:, 1],
        "offset_m": line.offset_m,
        "heading_rad": profile.heading_rad,
        "curvature_1_per_m": profile.curvature_1_per_m,
        "radius_m": profile.radius_m,
        "slope_pct": profile.slope_pct,
        "width_m": profile.width,
        "drive_left_m": profile.drive_left_m,
        "drive_right_m": profile.drive_right_m,
        "limit_left_m": profile.limit_left_m,
        "limit_right_m": profile.limit_right_m,
        "segment_id": seg_id,
        "segment_kind": seg_kind,
        "v_target_ms": v_target,
        "v_target_kmh": v_target * 3.6,
    })
    df.to_csv(csv, index=False)

    z_min, z_max = profile.elevation_range_m
    row = {
        "variant": name,
        "env": name[:2],
        "status": "ok",
        "line_source": line_used,
        "n_nodes": n_nodes,
        "n_segments": len(segments),
        "length_m": float(profile.total_length_m),
        "climb_m": float(profile.total_climb_m),
        "elev_min_m": float(z_min),
        "elev_max_m": float(z_max),
        "min_radius_m": float(np.min(profile.radius_m)),
        "max_slope_pct": float(np.max(np.abs(profile.slope_pct))),
        "mean_width_m": float(np.mean(profile.width)),
        "mean_drive_left_m": float(np.mean(profile.drive_left_m)),
        "mean_drive_right_m": float(np.mean(profile.drive_right_m)),
        "mean_limit_left_m": float(np.mean(profile.limit_left_m)),
        "mean_limit_right_m": float(np.mean(profile.limit_right_m)),
        "v_min_kmh": float(np.min(v_target) * 3.6),
        "v_max_kmh": float(np.max(v_target) * 3.6),
    }
    if pin is not None:
        row.update({
            "env_x_min_m": pin.x_min_m,
            "env_x_max_m": pin.x_max_m,
            "env_y_min_m": pin.y_min_m,
            "env_y_max_m": pin.y_max_m,
            "env_layout_count": pin.layout_count,
        })
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track", nargs="?", default=None,
                    help="LFS variant (e.g. BL1) or .pth file")
    ap.add_argument("--all", action="store_true",
                    help="process every PTH in --smx-dir")
    ap.add_argument("--smx-dir", type=Path, default=DEFAULT_SMX_DIR)
    ap.add_argument("--out", type=Path, default=Path("racing_lines"))
    ap.add_argument("--mu-lat", type=float, default=1.4,
                    help="lateral grip coefficient (default 1.4 ~ formula car)")
    ap.add_argument("--mu-long", type=float, default=1.2,
                    help="longitudinal grip coefficient (default 1.2)")
    ap.add_argument("--v-cap-kmh", type=float, default=288.0,
                    help="hard top-speed cap in km/h (default 288 = 80 m/s)")
    ap.add_argument("--car", default=None,
                    help="LFS car short name or mod id; if given, μ_lat/μ_long "
                         "are taken from the calibration store "
                         "(~/.lfs-telemetry/cars.json) or bundled defaults, "
                         "overriding --mu-lat/--mu-long unless those flags "
                         "are explicitly set.")
    ap.add_argument("--straight-radius", type=float, default=250.0)
    ap.add_argument("--min-segment-m", type=float, default=25.0)
    ap.add_argument("--edge-margin-m", type=float, default=0.4)
    ap.add_argument("--line-source", choices=["auto", "knw", "heuristic"],
                    default="auto",
                    help="auto = .knw if present else heuristic (default); "
                         "knw = require .knw; heuristic = never use .knw.")
    ap.add_argument("--knw-car", default="FBM",
                    help="car name used to look up the .knw AI line "
                         "(default FBM, present in every install layout).")
    ap.add_argument("--knw-dir", type=Path, default=DEFAULT_KNW_DIR)
    args = ap.parse_args(argv)

    if not args.track and not args.all:
        ap.error("provide a track name or --all")

    # If --car is given, override μ from the calibration store / bundled
    # table unless the user *explicitly* passed --mu-lat / --mu-long.
    explicit = {a.split("=")[0] for a in (argv or sys.argv[1:])}
    aero_k = 0.0
    if args.car:
        from lfs_telemetry.telemetry.car_calibration import CarSpecStore
        spec = CarSpecStore().spec_for(args.car)
        if "--mu-lat" not in explicit:
            args.mu_lat = spec.mu_lat
        if "--mu-long" not in explicit:
            args.mu_long = spec.mu_long
        aero_k = spec.mu_lat_aero_k
        aero_note = (f" + {aero_k:.2e}·v²" if aero_k > 0 else "")
        print(f"[racing_line] car={args.car.upper()} -> μ_lat={args.mu_lat:.2f}"
              f"{aero_note} μ_long={args.mu_long:.2f}", file=sys.stderr)

    files: list[Path]
    if args.all:
        files = list_path_files(args.smx_dir)
    else:
        p = Path(args.track)
        if p.suffix.lower() == ".pth" and p.exists():
            files = [p]
        else:
            files = [args.smx_dir / f"{args.track.upper()}.pth"]

    # Pre-load PIN files so each variant can be tagged with its env bbox.
    pins = load_all_pins(args.smx_dir)

    summary_rows = []
    for pth in files:
        if not pth.exists():
            print(f"[skip] {pth} not found", file=sys.stderr)
            continue
        env = pth.stem[:2].upper()
        try:
            row = process_one(
                pth, args.out,
                mu_lat=args.mu_lat, mu_long=args.mu_long,
                v_cap_ms=args.v_cap_kmh / 3.6,
                straight_radius=args.straight_radius,
                min_segment_m=args.min_segment_m,
                edge_margin_m=args.edge_margin_m,
                mu_lat_aero_k=aero_k,
                pin=pins.get(env),
                line_source=args.line_source,
                knw_car=args.knw_car,
                knw_dir=args.knw_dir,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {pth.stem}: {exc}", file=sys.stderr)
            traceback.print_exc()
            continue
        print(f"  {row['variant']:<6} {row.get('status','?'):<6} "
              f"src={row.get('line_source','-'):<10} "
              f"v={row.get('v_min_kmh',0):.0f}-{row.get('v_max_kmh',0):.0f} km/h"
              f"   segs={row.get('n_segments','-')}",
              file=sys.stderr)
        summary_rows.append(row)

    if args.all and summary_rows:
        out_csv = args.out / "_racing_summary.csv"
        pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
        print(f"[summary] wrote {out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
