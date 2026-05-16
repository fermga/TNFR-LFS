"""Generate a plot pack for a captured lap CSV.

Produces a set of PNGs covering the most informative views of a single
lap, all derived from ``enrich_dataframe``:

* speed / throttle / brake / steer vs distance,
* G-G diagram (a_x vs a_y) coloured by speed,
* per-wheel friction-circle utilization vs distance,
* per-wheel vertical load vs distance,
* load transfer real vs theoretical (long & lat),
* yaw rate measured vs theoretical (understeer/oversteer index),
* tyre work [W] per wheel vs distance,
* track map coloured by speed and by friction utilization.

Usage:
    python tools/full_lap_plots.py stint_bl1_fbm_lap01.csv reports/plots/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lfs_telemetry.telemetry import car_spec_for, enrich_dataframe
from lfs_telemetry.telemetry.protocol.packets import WHEEL_ORDER

WHEEL_COLORS = {"FL": "#1f77b4", "FR": "#d62728",
                "RL": "#2ca02c", "RR": "#ff7f0e"}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  · {path.name}")


def _x_axis(df: pd.DataFrame) -> tuple[pd.Series, str]:
    if "current_lap_dist_m" in df.columns and df["current_lap_dist_m"].max() > 0:
        return df["current_lap_dist_m"], "Lap distance [m]"
    return (df["time_ms"] - df["time_ms"].iloc[0]) / 1000.0, "Time [s]"


def plot_inputs(df: pd.DataFrame, out: Path, name: str) -> None:
    x, xlabel = _x_axis(df)
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(x, df["speed_ms"] * 3.6, color="#222")
    axes[0].set_ylabel("Speed [km/h]")
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, df["throttle"], color="#2ca02c", label="throttle")
    axes[1].plot(x, df["brake"], color="#d62728", label="brake")
    axes[1].set_ylabel("Pedals [0-1]")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)
    if "input_steer" in df.columns:
        axes[2].plot(x, np.degrees(df["input_steer"]), color="#1f77b4")
    axes[2].set_ylabel("Steer [deg]")
    axes[2].grid(alpha=0.3)
    if "rpm" in df.columns:
        axes[2].twinx().plot(x, df["rpm"], color="#888", alpha=0.4, label="rpm")
    if "gear" in df.columns:
        axes[3].step(x, df["gear"], color="#9467bd", where="post")
    axes[3].set_ylabel("Gear")
    axes[3].set_xlabel(xlabel)
    axes[3].grid(alpha=0.3)
    fig.suptitle(f"{name} — driver inputs & speed", fontsize=13)
    _save(fig, out / f"{name}_01_inputs.png")


def plot_gg(df: pd.DataFrame, out: Path, name: str, g: float) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    sc = ax.scatter(df["accel_y"] / g, df["accel_x"] / g,
                    c=df["speed_ms"] * 3.6, s=4, cmap="viridis", alpha=0.7)
    lim = max(abs(df["accel_x"]).max(), abs(df["accel_y"]).max()) / g * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    for r in (0.5, 1.0, 1.5, 2.0):
        ax.add_artist(plt.Circle((0, 0), r, fill=False, color="grey",
                                 ls="--", lw=0.5))
    ax.set_aspect("equal")
    ax.set_xlabel("Lateral g (a_y / g)")
    ax.set_ylabel("Longitudinal g (a_x / g)")
    ax.set_title(f"{name} — G-G diagram (colour = speed km/h)")
    fig.colorbar(sc, ax=ax, label="Speed [km/h]", shrink=0.8)
    ax.grid(alpha=0.3)
    _save(fig, out / f"{name}_02_gg.png")


def plot_friction(df: pd.DataFrame, out: Path, name: str) -> None:
    cols = [f"friction_use_{c}" for c in WHEEL_ORDER
            if f"friction_use_{c}" in df.columns]
    if not cols:
        return
    x, xlabel = _x_axis(df)
    fig, ax = plt.subplots(figsize=(12, 5))
    for c in WHEEL_ORDER:
        col = f"friction_use_{c}"
        if col in df.columns:
            ax.plot(x, df[col].clip(upper=2.0), label=c,
                    color=WHEEL_COLORS[c], lw=0.8, alpha=0.8)
    ax.axhline(1.0, color="k", lw=0.6, ls="--", alpha=0.6,
               label="μ limit")
    ax.set_ylim(0, 2.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Friction circle utilisation")
    ax.set_title(f"{name} — friction-circle use per wheel (clipped at 2.0)")
    ax.legend(loc="upper right", ncol=5)
    ax.grid(alpha=0.3)
    _save(fig, out / f"{name}_03_friction.png")


def plot_loads(df: pd.DataFrame, out: Path, name: str) -> None:
    cols = [f"wheel_{c}_vertical_load_n" for c in WHEEL_ORDER]
    if not all(c in df.columns for c in cols):
        return
    x, xlabel = _x_axis(df)
    fig, ax = plt.subplots(figsize=(12, 5))
    for c in WHEEL_ORDER:
        ax.plot(x, df[f"wheel_{c}_vertical_load_n"], label=c,
                color=WHEEL_COLORS[c], lw=0.8, alpha=0.85)
    ax.set_xlabel(xlabel); ax.set_ylabel("Vertical load [N]")
    ax.set_title(f"{name} — per-wheel vertical load")
    ax.legend(loc="upper right", ncol=4)
    ax.grid(alpha=0.3)
    _save(fig, out / f"{name}_04_loads.png")


def plot_transfer(df: pd.DataFrame, out: Path, name: str) -> None:
    if not {"transfer_long_n_real", "transfer_long_n_theoretical"}.issubset(df.columns):
        return
    x, xlabel = _x_axis(df)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(x, df["transfer_long_n_theoretical"],
                 color="#888", label="theoretical (m·a·h/L)")
    axes[0].plot(x, df["transfer_long_n_real"],
                 color="#1f77b4", label="real (front − rear)", alpha=0.8)
    axes[0].set_ylabel("Long. transfer [N]")
    axes[0].legend(loc="upper right"); axes[0].grid(alpha=0.3)
    axes[1].plot(x, df["transfer_lat_n_theoretical"],
                 color="#888", label="theoretical")
    axes[1].plot(x, df["transfer_lat_n_real"],
                 color="#d62728", label="real (right − left)", alpha=0.8)
    axes[1].set_ylabel("Lat. transfer [N]"); axes[1].set_xlabel(xlabel)
    axes[1].legend(loc="upper right"); axes[1].grid(alpha=0.3)
    fig.suptitle(f"{name} — load transfer: real vs theoretical")
    _save(fig, out / f"{name}_05_load_transfer.png")


def plot_yaw(df: pd.DataFrame, out: Path, name: str) -> None:
    if not {"yaw_rate_rads", "yaw_rate_theoretical_rads"}.issubset(df.columns):
        return
    x, xlabel = _x_axis(df)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(x, np.degrees(df["yaw_rate_rads"]),
                 color="#1f77b4", label="measured ω_z")
    axes[0].plot(x, np.degrees(df["yaw_rate_theoretical_rads"]),
                 color="#888", label="theoretical (v·tan δ / L)")
    axes[0].set_ylabel("Yaw rate [deg/s]")
    axes[0].legend(loc="upper right"); axes[0].grid(alpha=0.3)
    axes[1].plot(x, np.degrees(df["understeer_index"]), color="#9467bd")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_ylabel("Understeer index [deg/s]\n(>0 understeer)")
    axes[1].set_xlabel(xlabel); axes[1].grid(alpha=0.3)
    fig.suptitle(f"{name} — yaw rate measured vs theoretical")
    _save(fig, out / f"{name}_06_yaw.png")


def plot_tyre_work(df: pd.DataFrame, out: Path, name: str) -> None:
    cols = [f"tyre_work_w_{c}" for c in WHEEL_ORDER if f"tyre_work_w_{c}" in df.columns]
    if not cols:
        return
    x, xlabel = _x_axis(df)
    fig, ax = plt.subplots(figsize=(12, 5))
    for c in WHEEL_ORDER:
        col = f"tyre_work_w_{c}"
        if col in df.columns:
            # Smooth a bit for readability.
            ax.plot(x, df[col].rolling(20, min_periods=1).mean(),
                    label=c, color=WHEEL_COLORS[c], lw=0.9, alpha=0.85)
    ax.set_xlabel(xlabel); ax.set_ylabel("Tyre work [W] (20-sample mean)")
    ax.set_title(f"{name} — per-wheel tyre work (energy proxy)")
    ax.legend(loc="upper right", ncol=4); ax.grid(alpha=0.3)
    _save(fig, out / f"{name}_07_tyre_work.png")


def plot_map(df: pd.DataFrame, out: Path, name: str) -> None:
    if not {"pos_x", "pos_y"}.issubset(df.columns):
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    sc1 = axes[0].scatter(df["pos_x"], df["pos_y"], c=df["speed_ms"] * 3.6,
                          s=3, cmap="viridis")
    axes[0].set_aspect("equal"); axes[0].grid(alpha=0.3)
    axes[0].set_title("Track map — speed [km/h]")
    fig.colorbar(sc1, ax=axes[0], shrink=0.8)
    # Friction utilisation = max across 4 wheels.
    fcols = [f"friction_use_{c}" for c in WHEEL_ORDER
             if f"friction_use_{c}" in df.columns]
    if fcols:
        fmax = df[fcols].max(axis=1).clip(upper=1.5)
        sc2 = axes[1].scatter(df["pos_x"], df["pos_y"], c=fmax, s=3,
                              cmap="magma", vmin=0, vmax=1.5)
        axes[1].set_aspect("equal"); axes[1].grid(alpha=0.3)
        axes[1].set_title("Track map — max friction use (any wheel)")
        fig.colorbar(sc2, ax=axes[1], shrink=0.8)
    fig.suptitle(f"{name} — track map")
    _save(fig, out / f"{name}_08_map.png")


def plot_steering_ffb(df: pd.DataFrame, out: Path, name: str) -> None:
    if "steer_torque_nm" not in df.columns:
        return
    x, xlabel = _x_axis(df)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(x, df["steer_torque_nm"], color="#1f77b4", lw=0.8)
    axes[0].axhline(0, color="k", lw=0.4)
    axes[0].set_ylabel("Steer torque [Nm]"); axes[0].grid(alpha=0.3)
    if "ffb_load_pct" in df.columns:
        axes[1].plot(x, df["ffb_load_pct"].clip(upper=1.5),
                     color="#ff7f0e", lw=0.8)
        axes[1].axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.5)
        axes[1].set_ylabel("FFB load (rel.)")
    axes[1].set_xlabel(xlabel); axes[1].grid(alpha=0.3)
    fig.suptitle(f"{name} — steering torque & FFB load")
    _save(fig, out / f"{name}_09_steering_ffb.png")


def main(csv_path: str, out_dir: str) -> int:
    csv = Path(csv_path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv)
    car = str(df["car"].dropna().iloc[0]) if "car" in df.columns else ""
    spec = car_spec_for(car)
    rich = enrich_dataframe(df, spec)
    name = csv.stem
    print(f"Generating plots for {csv.name} → {out}/")
    plot_inputs(rich, out, name)
    plot_gg(rich, out, name, spec.g)
    plot_friction(rich, out, name)
    plot_loads(rich, out, name)
    plot_transfer(rich, out, name)
    plot_yaw(rich, out, name)
    plot_tyre_work(rich, out, name)
    plot_map(rich, out, name)
    plot_steering_ffb(rich, out, name)
    print("Done.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: full_lap_plots.py <input.csv> <out_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
