"""Benchmark the structural operator detectors on synthetic telemetry."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from typing import Callable, Iterable, Sequence

from tnfr_core.equations.telemetry import TelemetryRecord
from tnfr_core.operators.operator_detection import (
    detect_en,
    detect_nul,
    detect_ra,
    detect_remesh,
    detect_thol,
    detect_um,
    detect_val,
    detect_zhir,
)


def _base_record() -> TelemetryRecord:
    return TelemetryRecord(
        timestamp=0.0,
        vertical_load=5200.0,
        slip_ratio=0.0,
        lateral_accel=0.0,
        longitudinal_accel=0.0,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        brake_pressure=0.0,
        locking=0.0,
        nfr=100.0,
        si=0.45,
        speed=35.0,
        yaw_rate=0.02,
        slip_angle=0.01,
        steer=0.1,
        throttle=0.4,
        gear=3,
        vertical_load_front=2600.0,
        vertical_load_rear=2600.0,
        mu_eff_front=1.0,
        mu_eff_rear=1.0,
        mu_eff_front_lateral=1.0,
        mu_eff_front_longitudinal=1.0,
        mu_eff_rear_lateral=1.0,
        mu_eff_rear_longitudinal=1.0,
        suspension_travel_front=0.02,
        suspension_travel_rear=0.02,
        suspension_velocity_front=0.0,
        suspension_velocity_rear=0.0,
        wheel_load_fl=600.0,
        wheel_load_fr=600.0,
        wheel_load_rl=600.0,
        wheel_load_rr=600.0,
    )


def _synthetic_series(samples: int) -> list[TelemetryRecord]:
    base = _base_record()
    series: list[TelemetryRecord] = []
    structural = 0.0
    for index in range(samples):
        t = index * 0.1
        oscillation = math.sin(index / 12.0)
        drift = math.cos(index / 20.0)
        nfr = 90.0 + 55.0 * oscillation + 15.0 * drift
        si = 0.4 + 0.18 * math.sin(index / 18.0)
        lateral = 1.2 + 1.1 * abs(math.sin(index / 10.0))
        throttle = 0.4 + 0.3 * max(0.0, math.sin(index / 14.0))
        longitudinal = -2.2 * max(0.0, math.sin((index - 8) / 11.0))
        front_load = 2600.0 + 450.0 * oscillation
        rear_load = 2600.0 - 350.0 * oscillation
        mu_front = 1.05 + 0.2 * math.sin(index / 17.0)
        mu_rear = 0.95 + 0.25 * math.cos(index / 15.0)
        suspension_front = 0.04 * math.sin(index / 9.0)
        suspension_rear = -0.05 * math.cos(index / 9.0)
        steer = 0.12 + 0.02 * math.sin(index / 25.0)
        yaw_rate = 0.05 + 0.4 * math.sin(index / 13.0)
        line_dev = 0.4 * abs(math.sin(index / 8.0)) + 0.6 * max(0.0, math.cos((index - 5) / 16.0))
        structural += 0.08 + 0.15 * max(0.0, math.cos((index - 3) / 19.0))

        record = replace(
            base,
            timestamp=t,
            vertical_load=front_load + rear_load,
            slip_ratio=0.01 * oscillation,
            slip_angle=0.02 * oscillation,
            lateral_accel=lateral,
            longitudinal_accel=longitudinal,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            brake_pressure=0.6 * max(0.0, -longitudinal),
            nfr=nfr,
            si=si,
            speed=40.0 + 6.0 * drift,
            yaw_rate=yaw_rate,
            steer=steer,
            throttle=throttle,
            vertical_load_front=front_load,
            vertical_load_rear=rear_load,
            mu_eff_front=mu_front,
            mu_eff_rear=mu_rear,
            suspension_velocity_front=suspension_front,
            suspension_velocity_rear=suspension_rear,
            wheel_load_fl=front_load * 0.5 + 150.0,
            wheel_load_fr=front_load * 0.5 + 140.0,
            wheel_load_rl=rear_load * 0.5 + 130.0,
            wheel_load_rr=rear_load * 0.5 + 120.0,
            line_deviation=line_dev,
            structural_timestamp=structural,
        )
        series.append(record)
    return series


def _time_detector(
    detector: Callable[..., Sequence[dict]],
    records: Sequence[TelemetryRecord],
    iterations: int,
    **kwargs: float,
) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        detector(records, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed / max(1, iterations)


def _print_results(results: Iterable[tuple[str, float]]) -> None:
    for name, duration in results:
        print(f"{name:<12} {duration * 1e6:8.2f} µs/run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=512, help="Synthetic samples to generate.")
    parser.add_argument("--iterations", type=int, default=25, help="Detector evaluations per scenario.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    series = _synthetic_series(max(1, args.samples))
    iterations = max(1, args.iterations)
    scenarios: list[tuple[str, Callable[..., Sequence[dict]], dict[str, float]]] = [
        ("detect_en", detect_en, dict(psi_threshold=0.9, epi_norm_max=120.0)),
        ("detect_um", detect_um, dict(rho_min=0.6, phase_max=0.15, min_duration=0.3)),
        ("detect_ra", detect_ra, dict(nu_band=(1.0, 3.0), si_min=0.55, delta_nfr_max=18.0, k_min=2)),
        ("detect_val", detect_val, dict(epi_growth_min=0.35, active_nodes_delta_min=2, active_node_load_min=250.0)),
        ("detect_nul", detect_nul, dict(active_nodes_delta_max=-1, epi_concentration_min=0.55, active_node_load_min=250.0)),
        ("detect_thol", detect_thol, dict(epi_accel_min=5.0, stability_window=0.3, stability_tolerance=0.05)),
        ("detect_zhir", detect_zhir, dict(xi_min=0.3, min_persistence=0.3, phase_jump_min=0.2)),
        (
            "detect_remesh",
            detect_remesh,
            dict(
                window=40,
                tau_candidates=(0.6, 1.2, 1.8),
                acf_min=0.7,
                min_repeats=2,
            ),
        ),
    ]

    results = []
    for name, detector, kwargs in scenarios:
        duration = _time_detector(detector, series, iterations, **kwargs)
        results.append((name, duration))

    _print_results(results)


if __name__ == "__main__":
    main()
