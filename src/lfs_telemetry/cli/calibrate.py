"""``lfs-telemetry calibrate`` subcommand."""
from __future__ import annotations

import argparse
import asyncio
import sys

from ..telemetry.car_calibration import CarSpecStore, RestCalibrator
from ..telemetry.live import LiveTelemetry


async def _cmd_calibrate(args: argparse.Namespace) -> int:
    store = CarSpecStore(args.store)
    if args.show:
        store.load()
        cars = store.all()
        if not cars:
            print(f"[calibrate] store is empty: {store.path}")
            return 0
        print(f"[calibrate] {store.path}")
        for cid, cal in sorted(cars.items()):
            print(f"  {cid:6s} mass={cal.mass_kg:6.1f} kg  "
                  f"front={cal.weight_dist_front*100:5.1f}%  "
                  f"left={cal.left_fraction*100:5.1f}%  "
                  f"(n={cal.sample_count})")
        return 0

    print(
        "[calibrate] Park the car on a flat track surface, idle gear (or "
        "clutch in), no throttle/brake. The calibration takes ~1 second.",
        file=sys.stderr)
    cal_engine = RestCalibrator()
    deadline = asyncio.get_running_loop().time() + args.timeout
    detected: list[str] = []
    last_diag_t = 0.0
    try:
        async with LiveTelemetry(
            args.outsim_port, args.outgauge_port,
            outsim_opts=args.outsim_opts,
            insim_host=args.insim_host,
            insim_port=args.insim_port,
            insim_admin_password=args.insim_admin,
        ) as live:
            async for sample in live.samples():
                og = sample.outgauge
                car = (og.car if og else None) or "?"
                if car not in detected:
                    detected.append(car)
                    print(f"[calibrate] detected car id: {car}", file=sys.stderr)
                cal = cal_engine.feed(sample)
                if cal is not None:
                    store.put(cal)
                    store.save()
                    print(f"[calibrate] OK  car={cal.car_id}  "
                          f"mass={cal.mass_kg:.1f} kg  "
                          f"front={cal.weight_dist_front*100:.1f}%  "
                          f"left={cal.left_fraction*100:.1f}%  "
                          f"(n={cal.sample_count})  -> {store.path}")
                    return 0
                now = asyncio.get_running_loop().time()
                if now - last_diag_t >= 2.0:
                    last_diag_t = now
                    reason = cal_engine.diagnose()
                    if reason:
                        print(f"[calibrate] waiting: {reason}", file=sys.stderr)
                if now >= deadline:
                    print("[calibrate] timeout — car never reached rest. "
                          "Make sure you are stationary on the track with "
                          "throttle and brake released.", file=sys.stderr)
                    return 1
    except KeyboardInterrupt:
        print("[calibrate] stopped by user", file=sys.stderr)
        return 1
    return 1

