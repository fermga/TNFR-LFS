"""Benchmark vectorised ν_f multiplier extraction against the legacy approach."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Iterable, Sequence

from tnfr_lfs.resources import data_root
from tnfr_core.runtime.shared import CacheOptions
from tnfr_core.epi import NaturalFrequencyAnalyzer, NaturalFrequencySettings, TelemetryRecord
from tnfr_core.spectrum import cross_spectrum, estimate_sample_rate, power_spectrum
from tnfr_lfs.telemetry.offline import ReplayCSVBundleReader

_DEFAULT_BUNDLE = data_root() / "test1.zip"


def _load_records(path: Path, *, telemetry_cache: int, sample_limit: int) -> list[TelemetryRecord]:
    reader = ReplayCSVBundleReader(path, cache_size=telemetry_cache)
    records = reader.to_records(copy=False)
    if sample_limit > 0:
        return records[:sample_limit]
    return records


def _legacy_dynamic_multipliers(
    history: Sequence[TelemetryRecord],
    settings: NaturalFrequencySettings,
    car_model: str | None,
) -> tuple[dict[str, float], float]:
    history = list(history)
    if len(history) < 2:
        return {}, 0.0

    duration = history[-1].timestamp - history[0].timestamp
    if duration < max(0.0, settings.min_window_seconds - 1e-6):
        return {}, 0.0

    sample_rate = estimate_sample_rate(history)
    if sample_rate <= 0.0:
        return {}, 0.0

    min_samples = max(4, int(settings.min_window_seconds * sample_rate))
    if len(history) < min_samples:
        return {}, 0.0

    steer_series = [float(record.steer) for record in history]
    throttle_series = [float(record.throttle) for record in history]
    brake_series = [float(record.brake_pressure) for record in history]
    suspension_front = [float(record.suspension_velocity_front) for record in history]
    suspension_rear = [float(record.suspension_velocity_rear) for record in history]
    suspension_combined = [
        (front + rear) * 0.5 for front, rear in zip(suspension_front, suspension_rear)
    ]

    low = max(0.0, settings.bandpass_low_hz)
    high = max(low, settings.bandpass_high_hz)

    def _legacy_dominant_frequency(series: Sequence[float]) -> float:
        spectrum = power_spectrum(series, sample_rate)
        band = [entry for entry in spectrum if low <= entry[0] <= high]
        if not band:
            return 0.0
        frequency, energy = max(band, key=lambda entry: entry[1])
        if energy <= 1e-9:
            return 0.0
        return frequency

    def _legacy_dominant_cross(
        x_series: Sequence[float], y_series: Sequence[float]
    ) -> float:
        spectrum = cross_spectrum(x_series, y_series, sample_rate)
        band = [entry for entry in spectrum if low <= entry[0] <= high]
        if not band:
            return 0.0
        frequency, real, imag = max(
            band, key=lambda entry: math.hypot(entry[1], entry[2])
        )
        magnitude = math.hypot(real, imag)
        if magnitude <= 1e-9:
            return 0.0
        return frequency

    steer_freq = _legacy_dominant_frequency(steer_series)
    throttle_freq = _legacy_dominant_frequency(throttle_series)
    brake_freq = _legacy_dominant_frequency(brake_series)
    suspension_freq = _legacy_dominant_frequency(suspension_combined)
    tyre_freq = _legacy_dominant_cross(steer_series, suspension_combined)

    vehicle_frequency = settings.resolve_vehicle_frequency(car_model)

    def _normalise(frequency: float) -> float:
        if frequency <= 0.0:
            return 1.0
        ratio = frequency / vehicle_frequency
        ratio = max(settings.min_multiplier, min(settings.max_multiplier, ratio))
        return ratio

    dominant_frequency = 0.0
    multipliers: dict[str, float] = {}
    if steer_freq > 0.0:
        multipliers["driver"] = _normalise(steer_freq)
        dominant_frequency = steer_freq
    if throttle_freq > 0.0:
        multipliers["transmission"] = _normalise(throttle_freq)
        if throttle_freq > dominant_frequency:
            dominant_frequency = throttle_freq
    if brake_freq > 0.0:
        multipliers["brakes"] = _normalise(brake_freq)
        if brake_freq > dominant_frequency:
            dominant_frequency = brake_freq
    if suspension_freq > 0.0:
        value = _normalise(suspension_freq)
        multipliers["suspension"] = value
        multipliers.setdefault("chassis", value)
        if suspension_freq > dominant_frequency:
            dominant_frequency = suspension_freq
    if tyre_freq > 0.0:
        value = _normalise(tyre_freq)
        multipliers["tyres"] = value
        multipliers["chassis"] = value
        if tyre_freq > dominant_frequency:
            dominant_frequency = tyre_freq
    elif suspension_freq > 0.0 and steer_freq > 0.0:
        blended = (multipliers["suspension"] + multipliers["driver"]) * 0.5
        multipliers["tyres"] = blended

    return multipliers, dominant_frequency


def _select_windows(
    records: Sequence[TelemetryRecord],
    *,
    settings: NaturalFrequencySettings,
    sample_rate: float,
    limit: int,
    window_seconds: float | None,
) -> list[Sequence[TelemetryRecord]]:
    if window_seconds is None:
        window_seconds = settings.max_window_seconds
    window_samples = max(4, int(math.ceil(window_seconds * sample_rate)))
    windows: list[Sequence[TelemetryRecord]] = []
    for index in range(window_samples, len(records) + 1):
        window = records[index - window_samples : index]
        if window[-1].timestamp - window[0].timestamp < settings.min_window_seconds - 1e-6:
            continue
        windows.append(window)
        if 0 < limit <= len(windows):
            break
    return windows


def _measure_vectorised(
    windows: Iterable[Sequence[TelemetryRecord]],
    analyzer: NaturalFrequencyAnalyzer,
    car_model: str | None,
) -> float:
    start = time.perf_counter()
    for window in windows:
        analyzer._compute_dynamic_multipliers_raw(window, car_model)
    return time.perf_counter() - start


def _measure_legacy(
    windows: Iterable[Sequence[TelemetryRecord]],
    analyzer: NaturalFrequencyAnalyzer,
    car_model: str | None,
) -> float:
    start = time.perf_counter()
    for window in windows:
        _legacy_dynamic_multipliers(window, analyzer.settings, car_model)
    return time.perf_counter() - start


def _format(label: str, duration: float, windows: int) -> str:
    throughput = windows / duration if duration else float("nan")
    return f"{label}: {duration:.6f}s ({throughput:.1f} windows/s)"


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
        "--sample-limit",
        type=int,
        default=256,
        help="Maximum number of telemetry samples to load (0 keeps all).",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Window duration in seconds (defaults to analyzer settings).",
    )
    parser.add_argument(
        "--window-limit",
        type=int,
        default=128,
        help="Limit the number of windows evaluated (0 keeps all).",
    )
    parser.add_argument(
        "--car-model",
        type=str,
        default=None,
        help="Optional car model identifier used to resolve ν_f bands.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_path = args.bundle.expanduser().resolve()
    if not bundle_path.exists():
        raise SystemExit(f"Bundle not found: {bundle_path}")

    records = _load_records(
        bundle_path,
        telemetry_cache=args.telemetry_cache,
        sample_limit=args.sample_limit,
    )
    if len(records) < 2:
        raise SystemExit("Not enough telemetry samples to benchmark.")

    analyzer = NaturalFrequencyAnalyzer(cache_options=CacheOptions(nu_f_cache_size=0))
    sample_rate = estimate_sample_rate(records)
    if sample_rate <= 0.0:
        raise SystemExit("Could not resolve a valid telemetry sample rate.")

    windows = _select_windows(
        records,
        settings=analyzer.settings,
        sample_rate=sample_rate,
        limit=args.window_limit,
        window_seconds=args.window_seconds,
    )
    if not windows:
        raise SystemExit("No telemetry windows satisfied the duration constraints.")

    vectorised_duration = _measure_vectorised(windows, analyzer, args.car_model)
    legacy_duration = _measure_legacy(windows, analyzer, args.car_model)

    print(_format("Vectorised", vectorised_duration, len(windows)))
    print(_format("Legacy", legacy_duration, len(windows)))
    if vectorised_duration > 0.0:
        speedup = legacy_duration / vectorised_duration
        delta = legacy_duration - vectorised_duration
        print(f"Speed-up: ×{speedup:.2f} ({delta:.6f}s saved over {len(windows)} windows)")


if __name__ == "__main__":
    main()
