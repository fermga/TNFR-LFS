"""Windowed telemetry metrics used by the HUD and setup planner."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Sequence

from .dissonance import compute_useful_dissonance_stats
from .epi import TelemetryRecord
from .epi_models import EPIBundle
from .spectrum import phase_alignment
from .resonance import estimate_excitation_frequency

__all__ = ["WindowMetrics", "compute_window_metrics"]


@dataclass(frozen=True)
class WindowMetrics:
    """Aggregated metrics derived from a telemetry window."""

    si: float
    d_nfr_couple: float
    d_nfr_res: float
    d_nfr_flat: float
    nu_f: float
    nu_exc: float
    rho: float
    phase_lag: float
    phase_alignment: float
    useful_dissonance_ratio: float
    useful_dissonance_percentage: float


def compute_window_metrics(
    records: Sequence[TelemetryRecord],
    *,
    phase_indices: Sequence[int] | None = None,
    bundles: Sequence[EPIBundle] | None = None,
) -> WindowMetrics:
    """Return averaged plan metrics for a telemetry window.

    Parameters
    ----------
    records:
        Ordered window of :class:`TelemetryRecord` samples.
    bundles:
        Optional precomputed :class:`~tnfr_lfs.core.epi_models.EPIBundle` series
        matching ``records``. When provided the ΔNFR derivative used for the
        Useful Dissonance Ratio (UDR) is computed from the smoothed bundle
        values.
    """

    if not records:
        return WindowMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    si_value = mean(record.si for record in records)

    if phase_indices:
        selected = [
            records[index]
            for index in phase_indices
            if 0 <= index < len(records)
        ]
    else:
        selected = list(records)
    if len(selected) < 4:
        selected = list(records)
    freq, lag, alignment = phase_alignment(selected)
    nu_exc = estimate_excitation_frequency(records)
    rho = nu_exc / freq if freq > 1e-9 else 0.0

    couple, resonance, flatten = _segment_gradients(records, segments=3)

    if bundles:
        timestamps = [bundle.timestamp for bundle in bundles]
        delta_series = [bundle.delta_nfr for bundle in bundles]
        yaw_rates = [bundle.chassis.yaw_rate for bundle in bundles]
    else:
        timestamps = [record.timestamp for record in records]
        delta_series = [getattr(record, "delta_nfr", record.nfr) for record in records]
        yaw_rates = [record.yaw_rate for record in records]
    _useful_samples, _high_yaw_samples, udr = compute_useful_dissonance_stats(
        timestamps,
        delta_series,
        yaw_rates,
    )

    return WindowMetrics(
        si=si_value,
        d_nfr_couple=couple,
        d_nfr_res=resonance,
        d_nfr_flat=flatten,
        nu_f=freq,
        nu_exc=nu_exc,
        rho=rho,
        phase_lag=lag,
        phase_alignment=alignment,
        useful_dissonance_ratio=udr,
        useful_dissonance_percentage=udr * 100.0,
    )


def _segment_gradients(
    records: Sequence[TelemetryRecord], *, segments: int
) -> tuple[float, ...]:
    parts = _split_records(records, segments)
    return tuple(_gradient(part) for part in parts)


def _split_records(
    records: Sequence[TelemetryRecord], segments: int
) -> list[Sequence[TelemetryRecord]]:
    length = len(records)
    if length == 0:
        return [tuple()] * segments

    base, remainder = divmod(length, segments)
    slices: list[Sequence[TelemetryRecord]] = []
    index = 0
    for segment_index in range(segments):
        size = base + (1 if segment_index < remainder else 0)
        if size <= 0:
            slices.append(records[index:index])
            continue
        next_index = index + size
        slices.append(records[index:next_index])
        index = next_index
    return slices


def _gradient(records: Iterable[TelemetryRecord]) -> float:
    iterator = list(records)
    if len(iterator) < 2:
        return 0.0
    start = iterator[0]
    end = iterator[-1]
    dt = end.timestamp - start.timestamp
    if dt <= 0.0:
        return 0.0
    return (end.nfr - start.nfr) / dt

