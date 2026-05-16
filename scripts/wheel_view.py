"""Per-wheel detailed telemetry view for a single LFS lap CSV.

Generates a multi-panel chart focused on tyre/wheel dynamics:
  - vertical load (4 wheels)
  - suspension deflection
  - slip ratio (longitudinal)
  - tan(slip angle) (lateral)
  - tyre forces (Fx, Fy)
  - wheel angular velocity vs vehicle speed (lock detection)
  - tyre air temp
  - lean rel. road (camber under load proxy)

Plus 2 standalone diagnostics:
  - per-wheel friction circle (Fx vs Fy normalised by Fz)
  - load balance front/rear and left/right over time

Usage:
    python scripts/wheel_view.py stint_bl1_fbm_lap01.csv [--save out]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WHEELS = ("FL", "FR", "RL", "RR")
COLORS = {"FL": "#1f77b4", "FR": "#d62728", "RL": "#2ca02c", "RR": "#ff7f0e"}
TYRE_R = 0.30  # rough effective radius (m) for FBM — only for lock-detect overlay


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["t"] = (df["time_ms"] - df["time_ms"].iloc[0]) / 1000.0
    df["speed_kmh"] = df["speed_ms"] * 3.6
    return df


def plot_wheels(df: pd.DataFrame, title: str) -> plt.Figure:
    df = _prep(df)
    t = df["t"].values

    fig, axes = plt.subplots(8, 1, figsize=(14, 18), sharex=True,
                             gridspec_kw={"hspace": 0.20})
    fig.suptitle(title, fontsize=13, y=0.995)

    panels = [
        ("vertical_load_n",        "Vertical load (N)",      1.0),
        ("susp_deflect_m",         "Susp. defl. (mm)",       1000.0),
        ("slip_ratio",             "Slip ratio (long.)",     1.0),
        ("tan_slip_angle",         "tan(slip angle) (lat.)", 1.0),
        ("x_force_n",              "Fx tyre (N)  long.",     1.0),
        ("y_force_n",              "Fy tyre (N)  lat.",      1.0),
        ("ang_vel_rads",           "Wheel ω (rad/s)",        1.0),
        ("air_temp_c",             "Tyre air temp (°C)",     1.0),
    ]
    for ax, (suffix, ylabel, scale) in zip(axes, panels):
        for w in WHEELS:
            ax.plot(t, df[f"wheel_{w}_{suffix}"].values * scale,
                    color=COLORS[w], lw=0.7, label=w)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    # Overlay vehicle speed-derived "expected ω" on the wheel-ω panel.
    expected_w = df["speed_ms"].values / TYRE_R
    axes[6].plot(t, expected_w, color="black", lw=0.6, ls="--",
                 alpha=0.6, label=f"v/R (R={TYRE_R}m)")
    axes[0].legend(loc="upper left", fontsize=8, ncol=4)
    axes[6].legend(loc="upper left", fontsize=8, ncol=5)
    axes[-1].set_xlabel("Lap time (s)")
    fig.align_ylabels(axes)
    return fig


def plot_friction_circles(df: pd.DataFrame, title: str) -> plt.Figure:
    df = _prep(df)
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    fig.suptitle(title, fontsize=13)
    pos = {"FL": (0, 0), "FR": (0, 1), "RL": (1, 0), "RR": (1, 1)}
    for w, (i, j) in pos.items():
        ax = axes[i, j]
        fz = df[f"wheel_{w}_vertical_load_n"].clip(lower=1.0).values
        fx = df[f"wheel_{w}_x_force_n"].values / fz
        fy = df[f"wheel_{w}_y_force_n"].values / fz
        sc = ax.scatter(fy, fx, c=df["t"], cmap="viridis", s=3, alpha=0.7)
        # 1 g friction circle reference
        theta = np.linspace(0, 2 * np.pi, 200)
        for r in (0.5, 1.0, 1.5):
            ax.plot(r * np.cos(theta), r * np.sin(theta),
                    color="#aaa", lw=0.4, ls="--")
        ax.axhline(0, color="black", lw=0.3)
        ax.axvline(0, color="black", lw=0.3)
        ax.set_aspect("equal")
        ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
        ax.set_xlabel("Fy/Fz  (lateral μ)")
        ax.set_ylabel("Fx/Fz  (long μ)")
        ax.set_title(f"{w}", color=COLORS[w], fontweight="bold")
        ax.grid(alpha=0.3)
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("Lap time (s)")
    return fig


def plot_load_balance(df: pd.DataFrame, title: str) -> plt.Figure:
    df = _prep(df)
    t = df["t"].values
    fz_FL = df["wheel_FL_vertical_load_n"]
    fz_FR = df["wheel_FR_vertical_load_n"]
    fz_RL = df["wheel_RL_vertical_load_n"]
    fz_RR = df["wheel_RR_vertical_load_n"]
    total = fz_FL + fz_FR + fz_RL + fz_RR
    front_pct = (fz_FL + fz_FR) / total * 100.0
    left_pct = (fz_FL + fz_RL) / total * 100.0
    diag1 = (fz_FL + fz_RR) / total * 100.0  # FL+RR diagonal

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(title, fontsize=13)

    ax = axes[0]
    ax.plot(t, front_pct, color="#1f77b4", lw=0.9, label="Front load %")
    ax.axhline(50, color="black", lw=0.4, ls="--")
    ax.set_ylabel("Front load (%)")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.plot(t, left_pct, color="#2ca02c", lw=0.9, label="Left load %")
    ax.axhline(50, color="black", lw=0.4, ls="--")
    ax.set_ylabel("Left load (%)")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    ax.plot(t, diag1, color="#d62728", lw=0.9, label="Diagonal FL+RR %")
    ax.axhline(50, color="black", lw=0.4, ls="--")
    ax.set_ylabel("Diagonal (%)")
    ax.set_xlabel("Lap time (s)")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)

    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    title = f"{args.csv.name}  —  wheel telemetry"

    f1 = plot_wheels(df, title)
    f2 = plot_friction_circles(df, f"{args.csv.name}  —  friction circles per wheel")
    f3 = plot_load_balance(df, f"{args.csv.name}  —  load balance over time")

    if args.save:
        s = args.save
        f1.savefig(s.with_name(s.stem + "_wheels.png"), dpi=130, bbox_inches="tight")
        f2.savefig(s.with_name(s.stem + "_friction.png"), dpi=130, bbox_inches="tight")
        f3.savefig(s.with_name(s.stem + "_balance.png"), dpi=130, bbox_inches="tight")
        print(f"saved 3 PNGs with stem {s}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
