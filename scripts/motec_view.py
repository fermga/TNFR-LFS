"""MoTeC-style multi-panel telemetry view for a single LFS lap CSV.

Usage:
    python scripts/motec_view.py stint_bl1_fbm_lap01.csv [--save out.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Standard gravity (NIST). Mirrors ``lfs_telemetry.telemetry.constants.GRAVITY``
# so this script stays runnable without installing the package.
GRAVITY = 9.80665

WHEELS = ("FL", "FR", "RL", "RR")
WHEEL_COLORS = {"FL": "#1f77b4", "FR": "#d62728", "RL": "#2ca02c", "RR": "#ff7f0e"}


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["t"] = (df["time_ms"] - df["time_ms"].iloc[0]) / 1000.0
    df["speed_kmh"] = df["speed_ms"] * 3.6
    # planar G-forces (g)
    df["g_lat"] = df["accel_x"] / GRAVITY
    df["g_lon"] = df["accel_y"] / GRAVITY
    # steer angle deg (rad input)
    df["steer_deg"] = np.degrees(df["input_steer"])
    # tyre slip energy proxy = |slip_ratio| + |tan_slip_angle|
    for w in WHEELS:
        df[f"slip_total_{w}"] = (
            df[f"wheel_{w}_slip_ratio"].abs()
            + df[f"wheel_{w}_tan_slip_angle"].abs()
        )
    return df


def plot_lap(df: pd.DataFrame, title: str) -> plt.Figure:
    df = _prepare(df)
    t = df["t"].values

    fig, axes = plt.subplots(
        8, 1, figsize=(14, 16), sharex=True,
        gridspec_kw={"hspace": 0.18,
                     "height_ratios": [2, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4]},
    )
    fig.suptitle(title, fontsize=13, y=0.995)

    # 1. Speed + RPM (twin)
    ax = axes[0]
    ax.plot(t, df["speed_kmh"], color="black", lw=1.2, label="Speed")
    ax.set_ylabel("Speed (km/h)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    axt = ax.twinx()
    axt.plot(t, df["rpm"], color="#888", lw=0.8, alpha=0.7, label="RPM")
    axt.set_ylabel("RPM", color="#666")
    axt.tick_params(axis="y", colors="#666")

    # 2. Throttle / Brake / Clutch
    ax = axes[1]
    ax.fill_between(t, 0, df["throttle"] * 100, color="#2ca02c",
                    alpha=0.55, label="Throttle %")
    ax.fill_between(t, 0, -df["brake"] * 100, color="#d62728",
                    alpha=0.55, label="Brake %")
    ax.plot(t, df["clutch"] * 100, color="#1f77b4", lw=0.8, label="Clutch %")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylabel("Pedals (%)")
    ax.set_ylim(-105, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=3)

    # 3. Steering
    ax = axes[2]
    ax.plot(t, df["steer_deg"], color="#9467bd", lw=1.0)
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylabel("Steer (deg)")
    ax.grid(alpha=0.3)

    # 4. Gear
    ax = axes[3]
    ax.plot(t, df["gear"], drawstyle="steps-post", color="#e377c2", lw=1.0)
    ax.set_ylabel("Gear")
    ax.set_yticks(range(int(df["gear"].min()), int(df["gear"].max()) + 1))
    ax.grid(alpha=0.3)

    # 5. G-G traces over time (lat & lon)
    ax = axes[4]
    ax.plot(t, df["g_lon"], color="#d62728", lw=0.8, label="Long. G (brake/accel)")
    ax.plot(t, df["g_lat"], color="#1f77b4", lw=0.8, label="Lat. G (cornering)")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylabel("G")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    # 6. Suspension deflection per wheel (mm)
    ax = axes[5]
    for w in WHEELS:
        ax.plot(t, df[f"wheel_{w}_susp_deflect_m"] * 1000.0,
                color=WHEEL_COLORS[w], lw=0.7, label=w)
    ax.set_ylabel("Susp. defl. (mm)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=4)

    # 7. Vertical load per wheel (N)
    ax = axes[6]
    for w in WHEELS:
        ax.plot(t, df[f"wheel_{w}_vertical_load_n"],
                color=WHEEL_COLORS[w], lw=0.7, label=w)
    ax.set_ylabel("Vert. load (N)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=4)

    # 8. Tyre slip total (slip_ratio + |tan_slip_angle|)
    ax = axes[7]
    for w in WHEELS:
        ax.plot(t, df[f"slip_total_{w}"],
                color=WHEEL_COLORS[w], lw=0.7, label=w)
    ax.set_ylabel("Slip total\n(|SR|+|tanα|)")
    ax.set_xlabel("Lap time (s)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=4)

    fig.align_ylabels(axes)
    return fig


def plot_gg(df: pd.DataFrame, title: str) -> plt.Figure:
    df = _prepare(df)
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(df["g_lat"], df["g_lon"], c=df["speed_kmh"],
                    cmap="viridis", s=4, alpha=0.7)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Speed (km/h)")
    # 1g reference circle
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in (0.5, 1.0, 1.5):
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="#aaa",
                lw=0.5, ls="--")
        ax.text(r * 0.71, r * 0.71, f"{r}g", fontsize=8, color="#777")
    ax.axhline(0, color="black", lw=0.4)
    ax.axvline(0, color="black", lw=0.4)
    ax.set_xlabel("Lateral G  (←left  right→)")
    ax.set_ylabel("Longitudinal G  (↓brake  accel↑)")
    ax.set_title(title)
    ax.set_aspect("equal")
    lim = max(2.0, np.ceil(max(df["g_lat"].abs().max(),
                               df["g_lon"].abs().max())))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.3)
    return fig


def plot_track_map(df: pd.DataFrame, title: str) -> plt.Figure:
    df = _prepare(df)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(df["pos_x"], df["pos_y"], c=df["speed_kmh"],
                    cmap="plasma", s=2)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Speed (km/h)")
    ax.set_aspect("equal")
    ax.set_xlabel("pos_x (m)")
    ax.set_ylabel("pos_y (m)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--save", type=Path, default=None,
                    help="Save figures next to CSV with this prefix")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    title = f"{args.csv.name}  —  {len(df)} samples  " \
            f"({(df['time_ms'].iloc[-1] - df['time_ms'].iloc[0]) / 1000.0:.3f} s)"

    fig1 = plot_lap(df, title)
    fig2 = plot_gg(df, f"G-G diagram — {args.csv.name}")
    fig3 = plot_track_map(df, f"Track map (speed-coloured) — {args.csv.name}")

    if args.save:
        stem = args.save
        fig1.savefig(stem.with_name(stem.stem + "_lap.png"), dpi=130, bbox_inches="tight")
        fig2.savefig(stem.with_name(stem.stem + "_gg.png"), dpi=130, bbox_inches="tight")
        fig3.savefig(stem.with_name(stem.stem + "_map.png"), dpi=130, bbox_inches="tight")
        print(f"saved 3 PNGs with stem {stem}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
