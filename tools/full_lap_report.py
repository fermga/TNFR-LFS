"""Generate a comprehensive telemetry report for a captured lap CSV.

Loads the canonical replay schema, runs ``enrich_dataframe`` and writes
both a Markdown report and a JSON sidecar with every available signal
summarized (raw + derived). Used to verify all telemetry plumbing on a
real lap recording.

Usage:
    python tools/full_lap_report.py stint_bl1_fbm_lap01.csv reports/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lfs_telemetry.telemetry import car_spec_for, enrich_dataframe
from lfs_telemetry.telemetry.protocol.packets import WHEEL_ORDER


def _stats(s: pd.Series) -> dict:
    if s.dtype == bool:
        s = s.astype(int)
    s = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    if s.empty:
        return {"n": 0}
    return {
        "n": int(s.size),
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "p05": float(s.quantile(0.05)),
        "p50": float(s.quantile(0.50)),
        "p95": float(s.quantile(0.95)),
    }


def _fmt_row(name: str, st: dict) -> str:
    if st["n"] == 0:
        return f"| `{name}` | — | — | — | — | — | — | — |"
    return (f"| `{name}` | {st['n']} | {st['min']:.4g} | {st['max']:.4g} "
            f"| {st['mean']:.4g} | {st['std']:.4g} "
            f"| {st['p05']:.4g} | {st['p95']:.4g} |")


def _section(title: str, df: pd.DataFrame, cols: list[str]) -> tuple[str, dict]:
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    lines = [f"### {title}", ""]
    if missing:
        lines.append(f"_Missing in capture: {', '.join('`'+c+'`' for c in missing)}_")
        lines.append("")
    if present:
        lines.append("| channel | n | min | max | mean | std | p05 | p95 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        stats = {}
        for c in present:
            st = _stats(df[c])
            stats[c] = st
            lines.append(_fmt_row(c, st))
        lines.append("")
        return "\n".join(lines), stats
    lines.append("_(no columns available)_\n")
    return "\n".join(lines), {}


def main(csv_path: str, out_dir: str) -> int:
    csv = Path(csv_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv)
    n_rows = len(df)
    if n_rows == 0:
        print("empty CSV", file=sys.stderr)
        return 1

    # Identify car and time domain.
    car_name = str(df["car"].dropna().iloc[0]) if "car" in df.columns else ""
    spec = car_spec_for(car_name)
    t_ms = df["time_ms"]
    duration_s = float((t_ms.iloc[-1] - t_ms.iloc[0]) / 1000.0)
    dt_med = float(np.median(np.diff(t_ms)) / 1000.0)
    sample_hz = 1.0 / dt_med if dt_med > 0 else float("nan")
    dist_m = (
        float(df["current_lap_dist_m"].max() - df["current_lap_dist_m"].min())
        if "current_lap_dist_m" in df.columns else float("nan")
    )

    # Derived enrichment.
    rich = enrich_dataframe(df, spec)
    n_derived = len(rich.columns) - len(df.columns)

    # Section catalogues.
    chassis = ["time_ms", "speed_ms", "rpm", "gear",
               "ang_vel_x", "ang_vel_y", "ang_vel_z",
               "heading", "pitch", "roll",
               "accel_x", "accel_y", "accel_z",
               "vel_x", "vel_y", "vel_z",
               "pos_x", "pos_y", "pos_z",
               "current_lap_dist_m", "indexed_distance_m"]
    inputs = ["throttle", "brake", "clutch",
              "input_throttle", "input_brake", "input_steer",
              "input_clutch", "input_handbrake",
              "steer_torque_nm", "engine_ang_vel_rads",
              "max_torque_at_vel_nm"]
    engine = ["fuel", "eng_temp_c", "oil_temp_c",
              "oil_pressure_bar", "turbo_bar"]
    dash = ["og_flags", "dash_lights", "show_lights", "og_player_id"]
    wheel_cols = []
    for c in WHEEL_ORDER:
        wheel_cols += [f"wheel_{c}_susp_deflect_m",
                       f"wheel_{c}_vertical_load_n",
                       f"wheel_{c}_slip_ratio",
                       f"wheel_{c}_tan_slip_angle",
                       f"wheel_{c}_x_force_n",
                       f"wheel_{c}_y_force_n",
                       f"wheel_{c}_ang_vel_rads",
                       f"wheel_{c}_lean_rel_road_rad",
                       f"wheel_{c}_air_temp_c",
                       f"wheel_{c}_slip_fraction",
                       f"wheel_{c}_touching",
                       f"wheel_{c}_steer_rad"]
    derived_chassis = ["yaw_rate_rads", "yaw_rate_theoretical_rads",
                       "understeer_index", "beta_rad", "beta_deg"]
    derived_load = ["transfer_long_n_theoretical", "transfer_lat_n_theoretical",
                    "transfer_long_n_real", "transfer_lat_n_real",
                    "load_total_n", "load_front_frac",
                    "load_left_frac", "load_diag_fl_rr_frac"]
    derived_friction = [f"friction_use_{c}" for c in WHEEL_ORDER]
    derived_work = [f"tyre_work_w_{c}" for c in WHEEL_ORDER]
    derived_brake = ["brake_bias_front_real"]
    derived_dash = ["dl_shift_light", "dl_handbrake", "dl_pit_limiter",
                    "dl_tc_active", "dl_oil_warn", "dl_battery_warn",
                    "dl_abs_active"]
    derived_ffb = ["ffb_load_pct"]
    derived_smooth = ["steer_rate_rads", "steer_reversal_rate_hz",
                      "throttle_rate_per_s", "brake_rate_per_s",
                      "overlap_brake_throttle"]
    ctx = [c for c in df.columns if c.startswith("ctx_")]

    # Build sections.
    sections: list[tuple[str, dict]] = []
    sections.append(_section("Chassis & motion (raw)", rich, chassis))
    sections.append(_section("Driver inputs & wheel torques (raw)", rich, inputs))
    sections.append(_section("Engine & fluids (raw)", rich, engine))
    sections.append(_section("Dash lights & OutGauge flags (new)", rich, dash))
    sections.append(_section("Per-wheel telemetry (raw)", rich, wheel_cols))
    sections.append(_section("Derived chassis dynamics", rich, derived_chassis))
    sections.append(_section("Derived load transfer", rich, derived_load))
    sections.append(_section("Derived friction-circle utilization", rich, derived_friction))
    sections.append(_section("Derived tyre work [W]", rich, derived_work))
    sections.append(_section("Derived brake bias", rich, derived_brake))
    sections.append(_section("Derived dash-light booleans", rich, derived_dash))
    sections.append(_section("Derived FFB load", rich, derived_ffb))
    sections.append(_section("Derived control smoothness", rich, derived_smooth))
    sections.append(_section("Race context (snapshot)", rich, ctx))

    # Aggregate stats for JSON sidecar.
    all_stats: dict = {}
    for _, st in sections:
        all_stats.update(st)

    # Special pickups.
    avg_speed = float(df["speed_ms"].mean()) if "speed_ms" in df.columns else float("nan")
    top_speed = float(df["speed_ms"].max()) if "speed_ms" in df.columns else float("nan")
    max_brake = float(df["brake"].max()) if "brake" in df.columns else float("nan")
    max_long_g = (float(df["accel_x"].abs().max()) / spec.g
                  if "accel_x" in df.columns else float("nan"))
    max_lat_g = (float(df["accel_y"].abs().max()) / spec.g
                 if "accel_y" in df.columns else float("nan"))
    max_friction = {
        c: float(rich[f"friction_use_{c}"].max())
        for c in WHEEL_ORDER if f"friction_use_{c}" in rich.columns
    }
    coverage_total = len(df.columns)
    coverage_pct = 100.0 * coverage_total / 95.0  # 95 = max canonical schema width

    # Header.
    header = [
        f"# Full telemetry report — `{csv.name}`",
        "",
        "## Capture summary",
        "",
        "| | |",
        "|---|---|",
        f"| Car | `{car_name}` (spec: mass={spec.mass_kg} kg, WB={spec.wheelbase_m} m, "
        f"μ_lat={spec.mu_lat}, μ_long={spec.mu_long}, drive={spec.driven}) |",
        f"| Rows | {n_rows:,} |",
        f"| Duration | {duration_s:.2f} s |",
        f"| Sample rate | {sample_hz:.1f} Hz (median Δt = {dt_med*1000:.1f} ms) |",
        f"| Lap distance | {dist_m:.1f} m |",
        f"| Avg speed | {avg_speed:.2f} m/s ({avg_speed*3.6:.1f} km/h) |",
        f"| Top speed | {top_speed:.2f} m/s ({top_speed*3.6:.1f} km/h) |",
        f"| Peak |a_x| | {max_long_g:.2f} g |",
        f"| Peak |a_y| | {max_lat_g:.2f} g |",
        f"| Peak brake input | {max_brake:.2f} |",
        f"| Raw columns | {coverage_total} / 95 ({coverage_pct:.0f}%) |",
        f"| Derived columns added | {n_derived} |",
        f"| Total columns after enrich | {len(rich.columns)} |",
        "",
        "### Per-wheel peak friction-circle utilization",
        "",
        "| wheel | peak | mean |",
        "|---|---:|---:|",
    ]
    for c in WHEEL_ORDER:
        col = f"friction_use_{c}"
        if col in rich.columns:
            header.append(
                f"| {c} | {rich[col].max():.3f} | {rich[col].mean():.3f} |")
    header.append("")

    body = "\n".join(header) + "\n## Channels\n\n" + "\n".join(
        s for s, _ in sections)

    md_path = out / (csv.stem + "_full_report.md")
    json_path = out / (csv.stem + "_full_report.json")
    md_path.write_text(body, encoding="utf-8")
    json_path.write_text(
        json.dumps({
            "source_csv": str(csv.name),
            "car": car_name,
            "rows": n_rows,
            "duration_s": duration_s,
            "sample_hz": sample_hz,
            "lap_distance_m": dist_m,
            "avg_speed_ms": avg_speed,
            "top_speed_ms": top_speed,
            "peak_friction_use": max_friction,
            "raw_columns": coverage_total,
            "derived_columns_added": n_derived,
            "channel_stats": all_stats,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Report written: {md_path}")
    print(f"JSON sidecar:   {json_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: full_lap_report.py <input.csv> <out_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
