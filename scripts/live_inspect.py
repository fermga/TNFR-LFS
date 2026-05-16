"""Live inspector: pretty-print every field LFS sends us, refreshed at 4 Hz.

Validates that:
  * OutSim, OutSimPack2, OutGauge parsers all decode cleanly
  * Per-wheel data (load, slip, temps, touching) makes physical sense
  * InSim race context (track, weather, players) populates correctly
  * The wheel-order remap RL/RR/FL/FR -> FL/FR/RL/RR is right

Run with LFS already in a session:
    .\.venv\Scripts\python.exe scripts\live_inspect.py
    .\.venv\Scripts\python.exe scripts\live_inspect.py --insim 127.0.0.1
"""
from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time

from lfs_telemetry.telemetry.live import LiveTelemetry
from lfs_telemetry.telemetry.protocol.packets import OSO_ALL, WHEEL_ORDER


def fmt_wheels(pkt2) -> str:
    if pkt2 is None or pkt2.wheels is None:
        return "  (no extended OutSim — wheels unavailable)"
    # Remap to FL/FR/RL/RR for display.
    by_lfs = dict(zip(WHEEL_ORDER, pkt2.wheels))
    lines = ["  wheel        load_N   slipR  slipA(deg)  airT  touch  susp(mm)"]
    for c in ("FL", "FR", "RL", "RR"):
        w = by_lfs[c]
        ang_deg = math.degrees(math.atan(w.tan_slip_angle))
        lines.append(
            f"  {c:>5}    {w.vertical_load_n:8.1f}  {w.slip_ratio:+6.3f}  "
            f"{ang_deg:+8.2f}    {w.air_temp_c:3d}    {w.touching:>2}   "
            f"{w.susp_deflect_m * 1000:+7.2f}"
        )
    return "\n".join(lines)


async def run(host: str | None) -> int:
    print("Connecting to LFS… (Ctrl-C to stop)", file=sys.stderr)
    async with LiveTelemetry(
        outsim_opts=OSO_ALL,
        insim_host=host,
        insim_port=29999,
    ) as live:
        last = time.time()
        n = 0
        async for sample in live.samples():
            n += 1
            now = time.time()
            if now - last < 0.25:  # 4 Hz refresh
                continue
            last = now
            os_p = sample.outsim
            og_p = sample.outgauge
            p2 = sample.outsim2
            ctx = sample.race_context
            ax, ay, az = (os_p.accel if os_p else (0, 0, 0))
            print("\033[2J\033[H", end="")  # clear screen
            print(f"=== LFS-Telemetry live inspector  (samples={n})  "
                  f"t={sample.time_ms} ms ===")
            if og_p:
                print(f"\nCAR  : {og_p.car!r:8}  gear={og_p.gear:>2}  "
                      f"speed={og_p.speed_ms * 3.6:6.1f} km/h  "
                      f"rpm={og_p.rpm:6.0f}  fuel={og_p.fuel * 100:5.1f}%")
                print(f"INPUT: thr={og_p.throttle:.2f}  brk={og_p.brake:.2f}  "
                      f"clu={og_p.clutch:.2f}  "
                      f"engT={og_p.eng_temp_c:5.1f}C  "
                      f"oilT={og_p.oil_temp_c:5.1f}C")
            if os_p:
                print(f"ACCEL: long={ax:+6.2f}  lat={ay:+6.2f}  "
                      f"vert={az:+6.2f}  m/s^2   "
                      f"yaw_rate={os_p.ang_vel[2]:+6.3f} rad/s")
                print(f"POSE : pitch={math.degrees(os_p.pitch):+5.2f}deg  "
                      f"roll={math.degrees(os_p.roll):+5.2f}deg  "
                      f"hdg={math.degrees(os_p.heading):+6.1f}deg")
                print(f"POS  : x={os_p.pos[0]:+8.2f}  y={os_p.pos[1]:+8.2f}  "
                      f"z={os_p.pos[2]:+7.2f} m")
            if p2 is not None:
                if p2.current_lap_dist_m is not None:
                    print(f"DIST : lap={p2.current_lap_dist_m:8.1f} m   "
                          f"steerTorque={p2.steer_torque_nm or 0:+6.2f} Nm   "
                          f"engRPM={(p2.engine_ang_vel_rads or 0) * 60 / (2 * math.pi):6.0f}")
            print("\nWHEELS (remapped FL/FR/RL/RR):")
            print(fmt_wheels(p2))
            if ctx is not None:
                snap = ctx.snapshot()
                print(f"\nINSIM: track={snap.get('track')!r}  "
                      f"weather={snap.get('weather')}  "
                      f"wind={snap.get('wind')}  "
                      f"race={snap.get('race_in_progress')}  "
                      f"players={len(snap.get('players') or {})}  "
                      f"lfs={snap.get('lfs_version')!r}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--insim", default=None,
                    help="enable InSim TCP (e.g. 127.0.0.1)")
    args = ap.parse_args()
    try:
        return asyncio.run(run(args.insim))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
