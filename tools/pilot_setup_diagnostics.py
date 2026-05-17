"""Pilot diagnostics: ¿hay señal física accionable en un stint?

Carga las 5 vueltas sintéticas BL1/FBM, las enriquece con la pipeline
existente y aplica 8 heurísticas vehiculares sin TNFR. Salida: un
veredicto que decide si vale la pena construir el advisor TNFR.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import pandas as pd
from scipy.signal import welch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lfs_telemetry.telemetry.lap import LapTelemetry  # noqa: E402
from lfs_telemetry.telemetry.sectors import lap_sectors  # noqa: E402

CORNERS = ("FL", "FR", "RL", "RR")
LAPS = sorted((REPO / "assets").glob("synthetic_BL1_FBM_lap*.csv"))


def _nz(x: float, default: float = float("nan")) -> float:
    return float(x) if np.isfinite(x) else default


def load_stint() -> list[LapTelemetry]:
    laps = [LapTelemetry.from_csv(p) for p in LAPS]
    if not laps:
        raise SystemExit("No synthetic laps found in assets/")
    return laps


# --------------------------------------------------------------------- #
# 1. Stint consistency
# --------------------------------------------------------------------- #
def check_consistency(laps: list[LapTelemetry]) -> dict:
    times = [l.summary.get("lap_time_s", float("nan")) for l in laps]
    pl_g = [l.summary.get("peak_lat_g", float("nan")) for l in laps]
    valid_times = [t for t in times if np.isfinite(t) and t > 0]
    if len(valid_times) < 2:
        return {"flag": "ERROR", "msg": "lap_time not computable"}
    mu = mean(valid_times)
    sd = pstdev(valid_times)
    cv = sd / mu if mu else float("nan")
    return {
        "lap_times_s": [round(t, 3) for t in times],
        "mean_s": round(mu, 3),
        "std_s": round(sd, 3),
        "cv_pct": round(100 * cv, 2),
        "peak_lat_g_per_lap": [round(_nz(g), 2) for g in pl_g],
        "flag": "OK" if cv < 0.02 else "INCONSISTENT_STINT",
    }


# --------------------------------------------------------------------- #
# 2. Saturation per wheel per sector phase
# --------------------------------------------------------------------- #
def _sector_phases(df: pd.DataFrame, sectors) -> list[tuple[str, np.ndarray]]:
    """Return list of (label, boolean mask) for entry/apex/exit per sector."""
    out = []
    d = pd.to_numeric(df["current_lap_dist_m"], errors="coerce").to_numpy()
    for s in sectors:
        span = s.end_d_m - s.start_d_m
        for phase, (a, b) in (
            ("entry", (0.0, 1 / 3)),
            ("apex", (1 / 3, 2 / 3)),
            ("exit", (2 / 3, 1.0)),
        ):
            lo = s.start_d_m + a * span
            hi = s.start_d_m + b * span
            mask = (d >= lo) & (d < hi)
            out.append((f"S{s.index + 1}.{phase}", mask))
    return out


def check_saturation(laps: list[LapTelemetry]) -> dict:
    flags = []
    borderline = []
    for lap in laps:
        df = lap.enriched
        sectors = lap_sectors(lap, n_equal=3)
        phases = _sector_phases(df, sectors)
        for c in CORNERS:
            col = f"friction_use_{c}"
            if col not in df.columns:
                continue
            vals = df[col].to_numpy()
            for label, mask in phases:
                if mask.sum() < 5:
                    continue
                v = vals[mask]
                v = v[np.isfinite(v)]
                if v.size == 0:
                    continue
                p95 = float(np.percentile(v, 95))
                rec = {
                    "lap": lap.source_path.stem.split("_lap")[-1],
                    "wheel": c,
                    "phase": label,
                    "p95": round(p95, 3),
                }
                if p95 > 0.95:
                    flags.append(rec)
                elif p95 > 0.85:
                    borderline.append(rec)
    return {
        "flag_count": len(flags),
        "borderline_count": len(borderline),
        "flags_top": flags[:10],
        "borderline_top": borderline[:10],
        "flag": "FLAG" if flags else ("borderline" if borderline else "OK"),
    }


# --------------------------------------------------------------------- #
# 3. Understeer / oversteer persistence
# --------------------------------------------------------------------- #
def check_balance_yaw(laps: list[LapTelemetry]) -> dict:
    per_lap = []
    for lap in laps:
        df = lap.enriched
        if "understeer_index" not in df.columns:
            continue
        u = pd.to_numeric(df["understeer_index"], errors="coerce").dropna()
        if u.empty:
            continue
        # mask: only when actually cornering (|lat_g| > 0.5)
        if "accel_y" in df.columns:
            ay = pd.to_numeric(df["accel_y"], errors="coerce").to_numpy()
            mask = np.abs(ay) > 0.5 * lap.car.g
            u_corner = u[mask[: len(u)]] if mask.sum() else u
        else:
            u_corner = u
        per_lap.append(
            {
                "lap": lap.source_path.stem.split("_lap")[-1],
                "u_median": round(_nz(u_corner.median()), 4),
                "u_p90": round(_nz(u_corner.quantile(0.9)), 4),
                "u_p10": round(_nz(u_corner.quantile(0.1)), 4),
            }
        )
    if not per_lap:
        return {"flag": "no_data"}
    medians = [r["u_median"] for r in per_lap]
    mu = mean(medians)
    flag = "OK"
    msg = ""
    if mu > 0.10:
        flag = "FLAG"
        msg = f"persistent understeer (median U = +{mu:.3f})"
    elif mu < -0.10:
        flag = "FLAG"
        msg = f"persistent oversteer (median U = {mu:.3f})"
    return {"per_lap": per_lap, "stint_median": round(mu, 4), "msg": msg, "flag": flag}


# --------------------------------------------------------------------- #
# 4. Lateral load balance steady state
# --------------------------------------------------------------------- #
def check_lateral_balance(laps: list[LapTelemetry]) -> dict:
    vals = []
    for lap in laps:
        df = lap.enriched
        if "load_left_frac" not in df.columns or "accel_y" not in df.columns:
            continue
        ay = pd.to_numeric(df["accel_y"], errors="coerce").to_numpy()
        ll = pd.to_numeric(df["load_left_frac"], errors="coerce").to_numpy()
        # straight-line mask: |ay| < 0.2 g
        mask = np.abs(ay) < 0.2 * lap.car.g
        mask &= np.isfinite(ll)
        if mask.sum() > 20:
            vals.append(float(np.median(ll[mask])))
    if not vals:
        return {"flag": "no_data"}
    mu = mean(vals)
    bias = mu - 0.5
    return {
        "median_load_left_frac_straight": round(mu, 4),
        "bias_vs_05": round(bias, 4),
        "flag": "FLAG" if abs(bias) > 0.02 else ("minor" if abs(bias) > 0.005 else "OK"),
    }


# --------------------------------------------------------------------- #
# 5. Brake bias real vs implicit
# --------------------------------------------------------------------- #
def check_brake_bias(laps: list[LapTelemetry]) -> dict:
    per_lap = []
    for lap in laps:
        df = lap.enriched
        if "brake_bias_front_real" not in df.columns:
            continue
        bb = pd.to_numeric(df["brake_bias_front_real"], errors="coerce")
        bb = bb.dropna()
        if bb.empty:
            continue
        per_lap.append(
            {
                "lap": lap.source_path.stem.split("_lap")[-1],
                "median_pct_F": round(100 * float(bb.median()), 2),
                "p10_pct_F": round(100 * float(bb.quantile(0.1)), 2),
                "p90_pct_F": round(100 * float(bb.quantile(0.9)), 2),
                "samples_braking": int(bb.size),
            }
        )
    if not per_lap:
        return {"flag": "no_data"}
    medians = [r["median_pct_F"] for r in per_lap]
    mu = mean(medians)
    spread = max(medians) - min(medians)
    return {
        "per_lap": per_lap,
        "stint_median_pct_F": round(mu, 2),
        "spread_pct_F": round(spread, 2),
        "flag": "FLAG" if spread > 2.0 else "OK",
        "msg": (
            f"brake bias drifts by {spread:.1f}%F across laps" if spread > 2.0 else ""
        ),
    }


# --------------------------------------------------------------------- #
# 6. Suspension natural mode per wheel
# --------------------------------------------------------------------- #
def check_suspension(laps: list[LapTelemetry]) -> dict:
    per_wheel: dict[str, list[dict]] = {c: [] for c in CORNERS}
    for lap in laps:
        df = lap.enriched
        t = pd.to_numeric(df["time_ms"], errors="coerce").to_numpy() / 1000.0
        if t.size < 200:
            continue
        dt = np.median(np.diff(t))
        if not (np.isfinite(dt) and dt > 0):
            continue
        fs = 1.0 / dt
        nperseg = int(min(len(t), max(64, round(1.0 * fs))))  # 1.0 s window
        for c in CORNERS:
            col = f"wheel_{c}_susp_speed_mps"
            if col not in df.columns:
                continue
            v = pd.to_numeric(df[col], errors="coerce").to_numpy()
            v = v[np.isfinite(v)]
            if v.size < nperseg:
                continue
            try:
                f, p = welch(v, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
            except Exception:
                continue
            band = (f >= 0.5) & (f <= 15.0)
            if not band.any():
                continue
            ix = int(np.argmax(p[band]))
            f_dom = float(f[band][ix])
            rms = float(np.sqrt(np.mean(v**2)))
            per_wheel[c].append({"f_dom_hz": f_dom, "rms_mps": rms})

    summary = {}
    flagged = []
    for c, rows in per_wheel.items():
        if not rows:
            continue
        fs_dom = [r["f_dom_hz"] for r in rows]
        rmss = [r["rms_mps"] for r in rows]
        rec = {
            "f_dom_mean_hz": round(mean(fs_dom), 2),
            "f_dom_std_hz": round(pstdev(fs_dom) if len(fs_dom) > 1 else 0.0, 2),
            "rms_mean_mps": round(mean(rmss), 4),
            "rms_max_mps": round(max(rmss), 4),
        }
        # heurística: si dominante < 4 Hz y RMS > 0.10 m/s sostenido → poco amortiguado
        if rec["f_dom_mean_hz"] < 4.0 and rec["rms_mean_mps"] > 0.10:
            flagged.append(c)
        summary[c] = rec
    return {
        "per_wheel": summary,
        "underdamped_wheels": flagged,
        "flag": "FLAG" if flagged else "OK",
    }


# --------------------------------------------------------------------- #
# 7. Tyre thermal work per axle
# --------------------------------------------------------------------- #
def check_tyre_work(laps: list[LapTelemetry]) -> dict:
    per_lap = []
    for lap in laps:
        df = lap.enriched
        t = pd.to_numeric(df["time_ms"], errors="coerce").to_numpy() / 1000.0
        dt = np.diff(t, prepend=t[0])
        row = {"lap": lap.source_path.stem.split("_lap")[-1]}
        joules = {}
        for c in CORNERS:
            col = f"tyre_work_w_{c}"
            if col not in df.columns:
                continue
            p = pd.to_numeric(df[col], errors="coerce").to_numpy()
            p = np.where(np.isfinite(p), p, 0.0)
            joules[c] = float(np.sum(p * dt))
        if joules:
            row["joules"] = {k: round(v, 0) for k, v in joules.items()}
            front = (joules.get("FL", 0.0) + joules.get("FR", 0.0)) / 2
            rear = (joules.get("RL", 0.0) + joules.get("RR", 0.0)) / 2
            row["front_rear_ratio"] = round(front / rear, 3) if rear else None
            left = (joules.get("FL", 0.0) + joules.get("RL", 0.0)) / 2
            right = (joules.get("FR", 0.0) + joules.get("RR", 0.0)) / 2
            row["left_right_ratio"] = round(left / right, 3) if right else None
            per_lap.append(row)
    if not per_lap:
        return {"flag": "no_data"}
    fr = [r["front_rear_ratio"] for r in per_lap if r.get("front_rear_ratio")]
    lr = [r["left_right_ratio"] for r in per_lap if r.get("left_right_ratio")]
    flag = "OK"
    msgs = []
    if fr and (mean(fr) > 1.5 or mean(fr) < 0.67):
        flag = "FLAG"
        msgs.append(f"front/rear thermal ratio {mean(fr):.2f}")
    if lr and abs(mean(lr) - 1.0) > 0.20:
        flag = "FLAG"
        msgs.append(f"left/right thermal ratio {mean(lr):.2f}")
    return {"per_lap": per_lap, "msg": "; ".join(msgs), "flag": flag}


# --------------------------------------------------------------------- #
# 8. Yaw rate vs theoretical (bicycle model)
# --------------------------------------------------------------------- #
def check_yaw(laps: list[LapTelemetry]) -> dict:
    deltas = []
    for lap in laps:
        df = lap.enriched
        if "yaw_rate_rads" not in df.columns or "yaw_rate_theoretical_rads" not in df.columns:
            continue
        yr = pd.to_numeric(df["yaw_rate_rads"], errors="coerce")
        yt = pd.to_numeric(df["yaw_rate_theoretical_rads"], errors="coerce")
        mask = yr.notna() & yt.notna() & (yt.abs() > 0.05)
        if mask.sum() < 50:
            continue
        diff = (yr - yt)[mask]
        deltas.append(float(diff.median()))
    if not deltas:
        return {"flag": "no_data"}
    mu = mean(deltas)
    return {
        "median_delta_rads": round(mu, 4),
        "median_delta_deg_s": round(np.degrees(mu), 2),
        "per_lap_delta_rads": [round(d, 4) for d in deltas],
        "flag": "FLAG" if abs(mu) > 0.05 else "OK",  # > ~3°/s sistemático
    }


# --------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------- #
def main() -> int:
    laps = load_stint()
    print(f"Loaded {len(laps)} laps from {LAPS[0].parent}\n")

    report = {
        "1_consistency": check_consistency(laps),
        "2_saturation": check_saturation(laps),
        "3_balance_understeer": check_balance_yaw(laps),
        "4_lateral_balance": check_lateral_balance(laps),
        "5_brake_bias": check_brake_bias(laps),
        "6_suspension": check_suspension(laps),
        "7_tyre_thermal": check_tyre_work(laps),
        "8_yaw_vs_theory": check_yaw(laps),
    }

    flags = []
    borderlines = []
    for k, v in report.items():
        f = v.get("flag", "?")
        line = f"  [{f:>14}] {k}"
        if f == "FLAG":
            flags.append(k)
        elif f in ("borderline", "minor", "INCONSISTENT_STINT"):
            borderlines.append(k)
        print(line)
        if "msg" in v and v["msg"]:
            print(f"               · {v['msg']}")

    print("\n--- DETAIL ---")
    print(json.dumps(report, indent=2, default=str))

    print("\n=== VERDICT ===")
    n_flag = len(flags)
    n_border = len(borderlines)
    print(f"  Findings: {n_flag} FLAGS, {n_border} borderline")
    if n_flag >= 3 and "1_consistency" not in borderlines:
        print("  -> GO: 3+ accionables, stint consistente. TNFR engine añadiría "
              "priorización + gramática + delta numérico al setup.")
        return 0
    if n_flag >= 1:
        print("  -> CAUTIOUS GO: pocas señales; las sintéticas dan margen estrecho. "
              "Plantear vueltas reales antes de invertir en Fase 1.")
        return 0
    print("  -> NO-GO: el método físico no separa señal del ruido sintético. "
          "Replantear el alcance o conseguir un stint real.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
