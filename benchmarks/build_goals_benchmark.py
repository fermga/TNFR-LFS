"""Benchmark phase averaging logic used by :func:`_build_goals`."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tnfr_lfs.analysis.contextual_delta as _lfs_contextual_delta

_lfs_contextual_delta.ensure_context_loader()

from tnfr_core.equations.contextual_delta import load_context_matrix, resolve_series_context
from tnfr_core.equations.baseline import DeltaCalculator
from tnfr_core.equations.epi import EPIExtractor, TelemetryRecord
from tnfr_core.metrics import segmentation as metrics_segmentation
from tnfr_core.segmentation import segment_microsectors


@dataclass(slots=True)
class PhaseStats:
    """Aggregated ΔNFR metrics for a phase window."""

    avg_delta: float
    avg_si: float
    avg_long: float
    avg_lat: float
    abs_long: float
    abs_lat: float


def _load_records(path: Path, limit: int) -> list[TelemetryRecord]:
    records: list[TelemetryRecord] = []
    with path.open(encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if 0 < limit <= index:
                break
            records.append(
                TelemetryRecord(
                    timestamp=float(row["timestamp"]),
                    vertical_load=float(row["vertical_load"]),
                    slip_ratio=float(row["slip_ratio"]),
                    lateral_accel=float(row["lateral_accel"]),
                    longitudinal_accel=float(row["longitudinal_accel"]),
                    yaw=float(row["yaw"]),
                    pitch=float(row["pitch"]),
                    roll=float(row["roll"]),
                    brake_pressure=float(row["brake_pressure"]),
                    locking=float(row["locking"]),
                    nfr=float(row["nfr"]),
                    si=float(row["si"]),
                    speed=float(row["speed"]),
                    yaw_rate=float(row["yaw_rate"]),
                    slip_angle=float(row["slip_angle"]),
                    steer=float(row["steer"]),
                    throttle=float(row["throttle"]),
                    gear=int(row["gear"]),
                    vertical_load_front=float(row["vertical_load_front"]),
                    vertical_load_rear=float(row["vertical_load_rear"]),
                    mu_eff_front=float(row["mu_eff_front"]),
                    mu_eff_rear=float(row["mu_eff_rear"]),
                    mu_eff_front_lateral=float(row["mu_eff_front_lateral"]),
                    mu_eff_front_longitudinal=float(row["mu_eff_front_longitudinal"]),
                    mu_eff_rear_lateral=float(row["mu_eff_rear_lateral"]),
                    mu_eff_rear_longitudinal=float(row["mu_eff_rear_longitudinal"]),
                    suspension_travel_front=float(row["suspension_travel_front"]),
                    suspension_travel_rear=float(row["suspension_travel_rear"]),
                    suspension_velocity_front=float(row["suspension_velocity_front"]),
                    suspension_velocity_rear=float(row["suspension_velocity_rear"]),
                )
            )
    return records


def _legacy_phase_stats(
    bundles: Sequence, multipliers: Sequence[float], start: int, stop: int
) -> PhaseStats:
    window = range(start, stop)
    if not window:
        return PhaseStats(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    adjusted_delta = [
        bundles[idx].delta_nfr * multipliers[offset]
        for offset, idx in enumerate(window)
        if 0 <= idx < len(bundles)
    ]
    if not adjusted_delta:
        return PhaseStats(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    avg_delta = statistics.fmean(adjusted_delta)
    avg_si = statistics.fmean(bundles[idx].sense_index for idx in window)
    avg_long = statistics.fmean(
        bundles[idx].delta_nfr_proj_longitudinal * multipliers[offset]
        for offset, idx in enumerate(window)
        if 0 <= idx < len(bundles)
    )
    avg_lat = statistics.fmean(
        bundles[idx].delta_nfr_proj_lateral * multipliers[offset]
        for offset, idx in enumerate(window)
        if 0 <= idx < len(bundles)
    )
    abs_long = statistics.fmean(
        abs(bundles[idx].delta_nfr_proj_longitudinal) * multipliers[offset]
        for offset, idx in enumerate(window)
        if 0 <= idx < len(bundles)
    )
    abs_lat = statistics.fmean(
        abs(bundles[idx].delta_nfr_proj_lateral) * multipliers[offset]
        for offset, idx in enumerate(window)
        if 0 <= idx < len(bundles)
    )
    return PhaseStats(avg_delta, avg_si, avg_long, avg_lat, abs_long, abs_lat)


def _optimised_phase_stats(
    bundles: Sequence, multipliers: Sequence[float], start: int, stop: int
) -> PhaseStats:
    sum_delta = 0.0
    sum_si = 0.0
    sum_long = 0.0
    sum_lat = 0.0
    sum_abs_long = 0.0
    sum_abs_lat = 0.0
    count = 0

    for offset, idx in enumerate(range(start, stop)):
        if not (0 <= idx < len(bundles)):
            continue
        bundle = bundles[idx]
        multiplier = multipliers[offset] if offset < len(multipliers) else 1.0
        sum_delta += bundle.delta_nfr * multiplier
        sum_si += bundle.sense_index
        long_component = bundle.delta_nfr_proj_longitudinal
        lat_component = bundle.delta_nfr_proj_lateral
        sum_long += long_component * multiplier
        sum_lat += lat_component * multiplier
        sum_abs_long += abs(long_component) * multiplier
        sum_abs_lat += abs(lat_component) * multiplier
        count += 1

    if not count:
        return PhaseStats(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    inv = 1.0 / count
    return PhaseStats(
        sum_delta * inv,
        sum_si * inv,
        sum_long * inv,
        sum_lat * inv,
        sum_abs_long * inv,
        sum_abs_lat * inv,
    )


def _benchmark(
    label: str,
    func,
    bundles: Sequence,
    windows: Sequence[tuple[int, int, Sequence[float]]],
    repeats: int,
    iterations: int,
) -> list[float]:
    durations: list[float] = []
    for _ in range(max(repeats, 1)):
        start_time = time.perf_counter()
        for _ in range(max(iterations, 1)):
            for start, stop, multipliers in windows:
                func(bundles, multipliers, start, stop)
        durations.append(time.perf_counter() - start_time)
    return durations


def _format(label: str, durations: Sequence[float], calls: int) -> str:
    mean_value = statistics.fmean(durations) if durations else 0.0
    deviation = statistics.pstdev(durations) if len(durations) > 1 else 0.0
    throughput = calls / mean_value if mean_value else math.nan
    return (
        f"{label}: {mean_value:.6f}s ± {deviation:.6f}s "
        f"({throughput:.1f} phase windows/s)"
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_source = Path(__file__).resolve().parents[1] / "tests" / "data" / "synthetic_stint.csv"
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source,
        help="Path to a telemetry CSV file (default: tests/data/synthetic_stint.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=512,
        help="Maximum number of samples to load from the CSV (0 keeps all).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timing repeats to perform (default: 5).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Phase aggregation iterations per repeat (default: 50).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    records = _load_records(args.source, args.limit)
    if not records:
        raise SystemExit("No records available for benchmarking")
    bundles = EPIExtractor().extract(records)
    baseline = DeltaCalculator.derive_baseline(records)
    microsectors = segment_microsectors(records, bundles, baseline=baseline)
    if not microsectors:
        raise SystemExit("Segmentation did not produce any microsectors")

    context_matrix = load_context_matrix()
    resolved_context = resolve_series_context(bundles, matrix=context_matrix)
    full_multipliers = [
        metrics_segmentation._resolve_context_multiplier(
            factors, context_matrix=context_matrix
        )
        for factors in resolved_context
    ]

    reference = next((sector for sector in microsectors if sector.phase_boundaries), None)
    if reference is None:
        raise SystemExit("Reference microsector does not expose phase boundaries")

    windows: list[tuple[int, int, Sequence[float]]] = []
    for start, stop in reference.phase_boundaries.values():
        start = max(0, start)
        stop = max(start, stop)
        window_multipliers = [
            full_multipliers[idx] if 0 <= idx < len(full_multipliers) else 1.0
            for idx in range(start, stop)
        ]
        windows.append((start, stop, window_multipliers))

    # Validate that both implementations produce identical outputs.
    for start, stop, multipliers in windows:
        legacy = _legacy_phase_stats(bundles, multipliers, start, stop)
        optimised = _optimised_phase_stats(bundles, multipliers, start, stop)
        for label, legacy_value, optimised_value in (
            ("avg_delta", legacy.avg_delta, optimised.avg_delta),
            ("avg_si", legacy.avg_si, optimised.avg_si),
            ("avg_long", legacy.avg_long, optimised.avg_long),
            ("avg_lat", legacy.avg_lat, optimised.avg_lat),
            ("abs_long", legacy.abs_long, optimised.abs_long),
            ("abs_lat", legacy.abs_lat, optimised.abs_lat),
        ):
            if not math.isclose(legacy_value, optimised_value, rel_tol=1e-9, abs_tol=1e-12):
                raise SystemExit(
                    f"Mismatch in {label} for window [{start}, {stop}): "
                    f"legacy={legacy_value:.12f} optimised={optimised_value:.12f}"
                )

    calls_per_iteration = sum(max(0, stop - start) for start, stop, _ in windows)
    legacy_durations = _benchmark(
        "legacy",
        _legacy_phase_stats,
        bundles,
        windows,
        args.repeats,
        args.iterations,
    )
    optimised_durations = _benchmark(
        "optimised",
        _optimised_phase_stats,
        bundles,
        windows,
        args.repeats,
        args.iterations,
    )

    print(_format("Legacy", legacy_durations, calls_per_iteration * max(args.iterations, 1)))
    print(
        _format(
            "Optimised",
            optimised_durations,
            calls_per_iteration * max(args.iterations, 1),
        )
    )


if __name__ == "__main__":
    main()
