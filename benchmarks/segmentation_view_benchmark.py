"""Benchmark window views against legacy slicing in segmentation utilities."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tnfr_lfs.analysis.contextual_delta as _lfs_contextual_delta

_lfs_contextual_delta.ensure_context_loader()

from tnfr_core.equations.telemetry import TelemetryRecord
from tnfr_core.metrics.resonance import estimate_excitation_frequency
from tnfr_core.metrics.segmentation import _SequenceSegment
from tnfr_core.metrics.spectrum import estimate_sample_rate, phase_alignment


def _load_records(path: Path, limit: int | None = None) -> list[TelemetryRecord]:
    records: list[TelemetryRecord] = []
    with path.open(encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if limit is not None and 0 <= limit <= index:
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


def _measure_time(func: Callable[[], object], repeats: int, iterations: int) -> list[float]:
    durations: list[float] = []
    for _ in range(max(repeats, 1)):
        start = time.perf_counter()
        for _ in range(max(iterations, 1)):
            func()
        durations.append(time.perf_counter() - start)
    return durations


def _measure_allocations(func: Callable[[], object], iterations: int) -> int:
    gc.collect()
    tracemalloc.start()
    for _ in range(max(iterations, 1)):
        func()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return int(peak)


def _format_series(label: str, values: Iterable[float]) -> str:
    values = list(values)
    if not values:
        return f"{label}: 0.000000s"
    mean_value = statistics.fmean(values)
    deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{label}: {mean_value:.6f}s ± {deviation:.6f}s"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_source = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "data"
        / "synthetic_stint.csv"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source,
        help="Path to a telemetry CSV file (default: tests/data/synthetic_stint.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum number of telemetry samples to load (default: 5000)",
    )
    parser.add_argument(
        "--window-start",
        type=int,
        default=300,
        help="Starting index of the benchmark window (default: 300)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=128,
        help="Number of samples in the benchmark window (default: 128)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=250,
        help="Number of iterations per benchmark repeat (default: 250)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of benchmark repeats for timing statistics (default: 5)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    records = _load_records(args.source, args.limit)
    if len(records) < 2:
        raise SystemExit("benchmark requires at least two telemetry records")

    window_size = max(2, min(len(records), args.window_size))
    max_start = max(0, len(records) - window_size)
    window_start = max(0, min(args.window_start, max_start))
    window_stop = min(len(records), window_start + window_size)

    sample_rate = estimate_sample_rate(records)
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise SystemExit("unable to derive a positive sample rate from telemetry")

    def _alignment_slice() -> tuple[float, float, float]:
        window = records[window_start:window_stop]
        return phase_alignment(window, sample_rate=sample_rate)

    def _alignment_view() -> tuple[float, float, float]:
        window = _SequenceSegment(records, window_start, window_stop - 1)
        return phase_alignment(window, sample_rate=sample_rate)

    def _excitation_slice() -> float:
        window = records[window_start:window_stop]
        return estimate_excitation_frequency(window, sample_rate)

    def _excitation_view() -> float:
        window = _SequenceSegment(records, window_start, window_stop - 1)
        return estimate_excitation_frequency(window, sample_rate)

    alignment_slice = _alignment_slice()
    alignment_view = _alignment_view()
    if not all(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        for a, b in zip(alignment_slice, alignment_view)
    ):
        raise SystemExit("phase alignment outputs diverge between slice and view")

    excitation_slice = _excitation_slice()
    excitation_view = _excitation_view()
    if not math.isclose(excitation_slice, excitation_view, rel_tol=1e-12, abs_tol=1e-12):
        raise SystemExit("excitation frequency diverges between slice and view")

    print(
        f"Benchmark window: start={window_start}, stop={window_stop}, "
        f"size={window_stop - window_start}"
    )
    print(
        f"Phase alignment (Hz, lag, alignment): {alignment_view!r}"
    )
    print(f"Excitation frequency: {excitation_view:.6f} Hz")

    alignment_slice_times = _measure_time(
        _alignment_slice, args.repeats, args.iterations
    )
    alignment_view_times = _measure_time(
        _alignment_view, args.repeats, args.iterations
    )
    excitation_slice_times = _measure_time(
        _excitation_slice, args.repeats, args.iterations
    )
    excitation_view_times = _measure_time(
        _excitation_view, args.repeats, args.iterations
    )

    print(_format_series("Alignment slice", alignment_slice_times))
    print(_format_series("Alignment view", alignment_view_times))
    print(_format_series("Excitation slice", excitation_slice_times))
    print(_format_series("Excitation view", excitation_view_times))

    alignment_slice_alloc = _measure_allocations(
        _alignment_slice, args.iterations
    )
    alignment_view_alloc = _measure_allocations(
        _alignment_view, args.iterations
    )
    excitation_slice_alloc = _measure_allocations(
        _excitation_slice, args.iterations
    )
    excitation_view_alloc = _measure_allocations(
        _excitation_view, args.iterations
    )

    print(
        "Alignment allocations: "
        f"slice={alignment_slice_alloc} B, view={alignment_view_alloc} B"
    )
    print(
        "Excitation allocations: "
        f"slice={excitation_slice_alloc} B, view={excitation_view_alloc} B"
    )


if __name__ == "__main__":
    main()

