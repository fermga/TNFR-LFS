"""Benchmark ΔNFR and ν_f computations with and without caching."""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from tnfr_lfs.cache_settings import CacheOptions
from tnfr_lfs.core.cache import (
    clear_delta_cache,
    clear_dynamic_cache,
    configure_cache,
)
from tnfr_lfs.core.epi import (
    NaturalFrequencyAnalyzer,
    TelemetryRecord,
    delta_nfr_by_node,
    resolve_nu_f_by_node,
)
from tnfr_lfs.io import ReplayCSVBundleReader

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BUNDLE = _REPO_ROOT / "data" / "test1.zip"


@dataclass(slots=True)
class BenchmarkResult:
    """Statistical summary for a benchmark series."""

    label: str
    durations: list[float]
    samples_per_run: int

    @property
    def mean(self) -> float:
        return statistics.fmean(self.durations)

    @property
    def pstdev(self) -> float:
        if len(self.durations) <= 1:
            return 0.0
        return statistics.pstdev(self.durations)

    @property
    def runs(self) -> int:
        return len(self.durations)


def _configure_caches(enable: bool, *, nu_f_cache_size: int) -> None:
    configure_cache(
        enable_delta_cache=enable,
        nu_f_cache_size=nu_f_cache_size if enable else 0,
    )
    clear_delta_cache()
    clear_dynamic_cache()


def _load_records(
    path: Path,
    *,
    telemetry_cache: int,
    sample_limit: int,
) -> list[TelemetryRecord]:
    reader = ReplayCSVBundleReader(path, cache_size=telemetry_cache)
    records = reader.to_records(copy=False)
    if sample_limit > 0:
        return records[:sample_limit]
    return records


def _run_pass(
    records: Sequence[TelemetryRecord],
    *,
    options: CacheOptions,
) -> tuple[float, Mapping[str, float], Mapping[str, float]]:
    analyzer = NaturalFrequencyAnalyzer(cache_options=options)
    start = time.perf_counter()
    last_delta: Mapping[str, float] = {}
    last_nu: Mapping[str, float] = {}
    for record in records:
        last_delta = delta_nfr_by_node(record, cache_options=options)
        snapshot = resolve_nu_f_by_node(
            record,
            analyzer=analyzer,
            cache_options=options,
        )
        last_nu = snapshot.by_node
    elapsed = time.perf_counter() - start
    return elapsed, last_delta, last_nu


def _measure_series(
    records: Sequence[TelemetryRecord],
    *,
    options: CacheOptions,
    repeats: int,
    iterations: int,
    warmup: int,
) -> tuple[list[float], Mapping[str, float], Mapping[str, float]]:
    iterations = max(iterations, 1)
    repeats = max(repeats, 1)
    durations: list[float] = []
    last_delta: Mapping[str, float] = {}
    last_nu: Mapping[str, float] = {}
    for _ in range(repeats):
        for _ in range(max(warmup, 0)):
            _run_pass(records, options=options)
        total = 0.0
        for _ in range(iterations):
            elapsed, last_delta, last_nu = _run_pass(records, options=options)
            total += elapsed
        durations.append(total / iterations)
    return durations, last_delta, last_nu


def _summarise(
    label: str,
    durations: Iterable[float],
    *,
    samples_per_run: int,
) -> BenchmarkResult:
    return BenchmarkResult(label=label, durations=list(durations), samples_per_run=samples_per_run)


def _format_result(result: BenchmarkResult) -> str:
    throughput = result.samples_per_run / result.mean if result.mean else float("nan")
    return (
        f"{result.label}: {result.mean:.6f}s ± {result.pstdev:.6f}s over {result.runs} runs "
        f"({throughput:.1f} samples/s)"
    )


def _print_summary(cached: BenchmarkResult, uncached: BenchmarkResult) -> None:
    print(_format_result(uncached))
    print(_format_result(cached))
    if cached.mean > 0.0:
        speedup = uncached.mean / cached.mean
    else:
        speedup = float("nan")
    delta = uncached.mean - cached.mean
    print(f"Speed-up: ×{speedup:.2f} ({delta:.6f}s saved per run)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=_DEFAULT_BUNDLE,
        help="Path to a Replay Analyzer CSV bundle (directory or .zip).",
    )
    parser.add_argument(
        "--telemetry-cache",
        type=int,
        default=1,
        help="ReplayCSVBundleReader cache size (0 disables caching).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Number of processing passes per measurement run.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of measured runs per scenario.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warm-up passes executed before capturing cached timings.",
    )
    parser.add_argument(
        "--nu-f-cache-size",
        type=int,
        default=256,
        help="Cache size for dynamic ν_f multipliers when caching is enabled.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=128,
        help="Maximum number of telemetry samples to benchmark (0 keeps all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_path = args.bundle.expanduser().resolve()
    if not bundle_path.exists():
        raise SystemExit(f"Bundle not found: {bundle_path}")

    records = _load_records(
        bundle_path,
        telemetry_cache=max(args.telemetry_cache, 0),
        sample_limit=max(args.sample_limit, 0),
    )
    if not records:
        raise SystemExit("Bundle does not contain telemetry samples")
    print(f"Loaded {len(records)} telemetry samples from {bundle_path}")

    iterations = max(args.iterations, 1)
    repeats = max(args.repeats, 1)

    uncached_options = CacheOptions(
        enable_delta_cache=False,
        nu_f_cache_size=0,
        telemetry_cache_size=max(args.telemetry_cache, 0),
    )
    cached_options = CacheOptions(
        enable_delta_cache=True,
        nu_f_cache_size=max(args.nu_f_cache_size, 0),
        telemetry_cache_size=max(args.telemetry_cache, 0),
    )

    _configure_caches(enable=False, nu_f_cache_size=uncached_options.nu_f_cache_size)
    uncached_durations, uncached_delta, uncached_nu = _measure_series(
        records,
        options=uncached_options,
        repeats=repeats,
        iterations=iterations,
        warmup=0,
    )
    uncached_summary = _summarise(
        "Uncached ΔNFR/ν_f",
        uncached_durations,
        samples_per_run=len(records),
    )

    _configure_caches(enable=True, nu_f_cache_size=cached_options.nu_f_cache_size)
    cached_durations, cached_delta, cached_nu = _measure_series(
        records,
        options=cached_options,
        repeats=repeats,
        iterations=iterations,
        warmup=max(args.warmup, 1),
    )
    cached_summary = _summarise(
        "Cached ΔNFR/ν_f",
        cached_durations,
        samples_per_run=len(records),
    )

    _print_summary(cached_summary, uncached_summary)

    uncached_delta_value = float(uncached_delta.get("tyres", math.nan))
    cached_delta_value = float(cached_delta.get("tyres", math.nan))
    uncached_nu_value = float(uncached_nu.get("tyres", math.nan))
    cached_nu_value = float(cached_nu.get("tyres", math.nan))
    if math.isfinite(uncached_delta_value) and math.isfinite(cached_delta_value):
        print(
            "ΔNFR tyres (uncached → cached): "
            f"{uncached_delta_value:.3f} → {cached_delta_value:.3f}"
        )
    if math.isfinite(uncached_nu_value) and math.isfinite(cached_nu_value):
        print(
            "ν_f tyres (uncached → cached): "
            f"{uncached_nu_value:.3f} → {cached_nu_value:.3f}"
        )

    _configure_caches(enable=True, nu_f_cache_size=cached_options.nu_f_cache_size)


if __name__ == "__main__":
    main()
