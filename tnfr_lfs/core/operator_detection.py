"""Detection utilities for on-track operator events.

Each detection routine analyses a windowed sequence of :class:`TelemetryRecord`
objects and yields event descriptors when the observed behaviour exceeds the
configured thresholds.  The detectors are intentionally lightweight so that
they can be executed on every microsector without adding measurable overhead
to the orchestration pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from statistics import mean
from typing import List, Mapping, Sequence, Tuple

from .epi import TelemetryRecord
from .structural_time import compute_structural_timestamps

__all__ = [
    "OperatorEvent",
    "STRUCTURAL_OPERATOR_LABELS",
    "canonical_operator_label",
    "normalize_structural_operator_identifier",
    "silence_event_payloads",
    "detect_al",
    "detect_oz",
    "detect_il",
    "detect_silence",
]


STRUCTURAL_OPERATOR_LABELS: Mapping[str, str] = {
    "AL": "Support",
    "OZ": "Dissonance",
    "IL": "Coherence",
    "SILENCE": "Structural silence",
}

STRUCTURAL_OPERATOR_ALIASES: Mapping[str, str] = {
    "SILENCIO": "SILENCE",
}


def normalize_structural_operator_identifier(identifier: str) -> str:
    """Return the canonical structural identifier for ``identifier``."""

    if not isinstance(identifier, str):
        return str(identifier)
    key = identifier.upper()
    return STRUCTURAL_OPERATOR_ALIASES.get(key, key)


def normalize_structural_operator_identifier(identifier: str) -> str:
    """Return the canonical structural identifier for ``identifier``."""

    if not isinstance(identifier, str):
        return str(identifier)
    return identifier.upper()


def normalize_structural_operator_identifier(identifier: str) -> str:
    """Return the canonical structural identifier for ``identifier``."""

    if not isinstance(identifier, str):
        return str(identifier)
    return identifier.upper()


def canonical_operator_label(identifier: str) -> str:
    """Return the canonical structural label for an operator identifier."""

    if not isinstance(identifier, str):
        return str(identifier)
    key = normalize_structural_operator_identifier(identifier)
    return STRUCTURAL_OPERATOR_LABELS.get(key, identifier)


def silence_event_payloads(
    events: Mapping[str, Sequence[Mapping[str, object]] | None] | None,
) -> Tuple[Mapping[str, object], ...]:
    """Return all silence payloads, accepting case-insensitive identifiers."""

    if not events:
        return ()

    collected: List[Mapping[str, object]] = []
    for name, payload in events.items():
        if normalize_structural_operator_identifier(name) != "SILENCE":
            continue
        if not payload:
            continue
        if isinstance(payload, SequenceABC) and not isinstance(payload, Mapping):
            collected.extend(payload)  # type: ignore[list-item]
        else:
            collected.append(payload)  # type: ignore[arg-type]
    return tuple(collected)


@dataclass(frozen=True)
class OperatorEvent:
    """Summary of a detected operator opportunity."""

    name: str
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    duration: float
    severity: float
    peak_value: float

    def as_mapping(self) -> Mapping[str, float | str | int]:
        return {
            "name": self.name,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "severity": self.severity,
            "peak_value": self.peak_value,
        }


def _window(records: Sequence[TelemetryRecord], start: int, size: int) -> Sequence[TelemetryRecord]:
    lower = max(0, start - size + 1)
    return records[lower : start + 1]


def _finalise_event(
    name: str,
    records: Sequence[TelemetryRecord],
    start_index: int,
    end_index: int,
    peak_value: float,
    threshold: float,
) -> OperatorEvent:
    start_index = max(0, start_index)
    end_index = max(start_index, end_index)
    start_time = float(records[start_index].timestamp)
    end_time = float(records[end_index].timestamp)
    duration = max(0.0, end_time - start_time)
    severity = peak_value / (threshold + 1e-9)
    return OperatorEvent(
        name=name,
        start_index=start_index,
        end_index=end_index,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        severity=max(0.0, severity),
        peak_value=peak_value,
    )


def detect_al(
    records: Sequence[TelemetryRecord],
    *,
    window: int = 5,
    lateral_threshold: float = 1.6,
    load_threshold: float = 250.0,
) -> List[Mapping[str, float | str | int]]:
    """Detect lateral support (AL) opportunities.

    The detector evaluates absolute lateral acceleration within a sliding
    window and confirms that the accompanying load transfer is significant
    before emitting an event.
    """

    if not records:
        return []

    events: List[OperatorEvent] = []
    active_start: int | None = None
    peak_value = 0.0
    for index, record in enumerate(records):
        window_records = _window(records, index, window)
        if len(window_records) < window:
            continue
        lateral_avg = mean(abs(sample.lateral_accel) for sample in window_records)
        load_span = max(sample.vertical_load for sample in window_records) - min(
            sample.vertical_load for sample in window_records
        )
        meets_threshold = lateral_avg >= lateral_threshold and load_span >= load_threshold
        if meets_threshold:
            if active_start is None:
                active_start = index - window + 1
                peak_value = lateral_avg
            else:
                peak_value = max(peak_value, lateral_avg)
        elif active_start is not None:
            events.append(
                _finalise_event(
                    canonical_operator_label("AL"),
                    records,
                    active_start,
                    index - 1,
                    peak_value,
                    lateral_threshold,
                )
            )
            active_start = None
            peak_value = 0.0

    if active_start is not None:
        events.append(
            _finalise_event(
                canonical_operator_label("AL"),
                records,
                active_start,
                len(records) - 1,
                peak_value,
                lateral_threshold,
            )
        )

    return [event.as_mapping() for event in events]


def detect_oz(
    records: Sequence[TelemetryRecord],
    *,
    window: int = 5,
    slip_threshold: float = 0.12,
    yaw_threshold: float = 0.25,
) -> List[Mapping[str, float | str | int]]:
    """Detect oversteer (OZ) excursions."""

    if not records:
        return []

    events: List[OperatorEvent] = []
    active_start: int | None = None
    peak_value = 0.0
    for index, record in enumerate(records):
        window_records = _window(records, index, window)
        if len(window_records) < window:
            continue
        slip_avg = mean(abs(sample.slip_angle) for sample in window_records)
        yaw_avg = mean(abs(sample.yaw_rate) for sample in window_records)
        meets_threshold = slip_avg >= slip_threshold and yaw_avg >= yaw_threshold
        if meets_threshold:
            if active_start is None:
                active_start = index - window + 1
                peak_value = max(slip_avg, yaw_avg)
            else:
                peak_value = max(peak_value, slip_avg, yaw_avg)
        elif active_start is not None:
            events.append(
                _finalise_event(
                    canonical_operator_label("OZ"),
                    records,
                    active_start,
                    index - 1,
                    peak_value,
                    max(slip_threshold, yaw_threshold),
                )
            )
            active_start = None
            peak_value = 0.0

    if active_start is not None:
        events.append(
            _finalise_event(
                canonical_operator_label("OZ"),
                records,
                active_start,
                len(records) - 1,
                peak_value,
                max(slip_threshold, yaw_threshold),
            )
        )

    return [event.as_mapping() for event in events]


def detect_il(
    records: Sequence[TelemetryRecord],
    *,
    window: int = 5,
    base_threshold: float = 0.35,
    speed_gain: float = 0.012,
) -> List[Mapping[str, float | str | int]]:
    """Detect ideal-line (IL) deviations with a speed dependent threshold."""

    if not records:
        return []

    events: List[OperatorEvent] = []
    active_start: int | None = None
    peak_value = 0.0
    for index, record in enumerate(records):
        window_records = _window(records, index, window)
        if len(window_records) < window:
            continue
        speed_samples = [abs(sample.speed) for sample in window_records if math.isfinite(sample.speed)]
        mean_speed = mean(speed_samples) if speed_samples else 0.0
        threshold = base_threshold + (speed_gain * mean_speed)
        deviation_samples = [
            abs(sample.line_deviation)
            for sample in window_records
            if math.isfinite(sample.line_deviation)
        ]
        deviation_peak = max(deviation_samples) if deviation_samples else 0.0
        meets_threshold = deviation_peak >= threshold
        if meets_threshold:
            if active_start is None:
                active_start = index - window + 1
                peak_value = deviation_peak
            else:
                peak_value = max(peak_value, deviation_peak)
        elif active_start is not None:
            events.append(
                _finalise_event(
                    canonical_operator_label("IL"),
                    records,
                    active_start,
                    index - 1,
                    peak_value,
                    threshold,
                )
            )
            active_start = None
            peak_value = 0.0

    if active_start is not None:
        final_records = _window(records, len(records) - 1, window)
        final_speed_samples = [
            abs(sample.speed) for sample in final_records if math.isfinite(sample.speed)
        ]
        mean_speed = mean(final_speed_samples) if final_speed_samples else 0.0
        threshold = base_threshold + (speed_gain * mean_speed)
        events.append(
            _finalise_event(
                canonical_operator_label("IL"),
                records,
                active_start,
                len(records) - 1,
                peak_value,
                threshold,
            )
        )

    return [event.as_mapping() for event in events]


def detect_silence(
    records: Sequence[TelemetryRecord],
    *,
    window: int = 15,
    load_threshold: float = 400.0,
    accel_threshold: float = 0.8,
    delta_nfr_threshold: float = 45.0,
    structural_window: int = 11,
    structural_density_threshold: float = 0.2,
    min_duration: float = 0.8,
) -> List[Mapping[str, float | str | int]]:
    """Detect structural silence intervals with low dynamic activation."""

    if not records:
        return []

    window = max(2, int(window))
    structural_window = max(3, int(structural_window))
    structural_axis = compute_structural_timestamps(
        records,
        window_size=structural_window,
        base_timestamp=records[0].timestamp,
    )

    densities: List[float] = [0.0]
    for index in range(1, len(records)):
        current = float(records[index].timestamp)
        previous = float(records[index - 1].timestamp)
        chronological_dt = max(0.0, current - previous)
        if chronological_dt <= 1e-9:
            densities.append(0.0)
            continue
        structural_dt = max(0.0, structural_axis[index] - structural_axis[index - 1])
        density = max(0.0, (structural_dt / (chronological_dt + 1e-9)) - 1.0)
        densities.append(density)

    def _slack(value: float, threshold: float) -> float:
        if threshold <= 1e-9:
            return 0.0
        return max(0.0, (threshold - value) / threshold)

    def _event_metrics(start_index: int, end_index: int) -> tuple[dict[str, float], float]:
        samples = records[start_index : end_index + 1]
        load_values = [float(sample.vertical_load) for sample in samples]
        load_span = max(load_values) - min(load_values) if load_values else 0.0
        lat_mean = mean(abs(sample.lateral_accel) for sample in samples)
        long_mean = mean(abs(sample.longitudinal_accel) for sample in samples)
        accel_mean = 0.5 * (lat_mean + long_mean)
        nfr_values = [float(getattr(sample, "nfr", 0.0)) for sample in samples]
        delta_nfr_span = max(nfr_values) - min(nfr_values) if nfr_values else 0.0
        density_slice = densities[start_index : end_index + 1]
        structural_density_mean = mean(density_slice) if density_slice else 0.0
        slack_components = (
            _slack(load_span, load_threshold),
            _slack(accel_mean, accel_threshold),
            _slack(delta_nfr_span, delta_nfr_threshold),
            _slack(structural_density_mean, structural_density_threshold),
        )
        slack = min(slack_components)
        return (
            {
                "load_span": float(load_span),
                "accel_mean": float(accel_mean),
                "delta_nfr_span": float(delta_nfr_span),
                "structural_density_mean": float(structural_density_mean),
            },
            float(slack),
        )

    def _append_event(start_index: int, end_index: int) -> None:
        nonlocal events
        start_index = max(0, start_index)
        end_index = max(start_index, end_index)
        metrics, slack = _event_metrics(start_index, end_index)
        if slack <= 0.0:
            return
        event = _finalise_event(
            canonical_operator_label("SILENCE"),
            records,
            start_index,
            end_index,
            slack,
            1.0,
        )
        if event.duration < float(min_duration):
            return
        payload = event.as_mapping()
        payload.update(metrics)
        structural_start = float(structural_axis[start_index])
        structural_end = float(structural_axis[end_index])
        payload.update(
            {
                "structural_start": structural_start,
                "structural_end": structural_end,
                "structural_duration": max(0.0, structural_end - structural_start),
                "load_threshold": float(load_threshold),
                "accel_threshold": float(accel_threshold),
                "delta_nfr_threshold": float(delta_nfr_threshold),
                "structural_density_threshold": float(structural_density_threshold),
                "structural_window": int(structural_window),
                "slack": float(slack),
            }
        )
        events.append(payload)

    events: List[dict[str, float | str | int]] = []
    active_start: int | None = None

    for index in range(len(records)):
        window_records = _window(records, index, window)
        if len(window_records) < window:
            continue
        load_values = [float(sample.vertical_load) for sample in window_records]
        load_span = max(load_values) - min(load_values) if load_values else 0.0
        lat_mean = mean(abs(sample.lateral_accel) for sample in window_records)
        long_mean = mean(abs(sample.longitudinal_accel) for sample in window_records)
        accel_mean = 0.5 * (lat_mean + long_mean)
        nfr_values = [float(getattr(sample, "nfr", 0.0)) for sample in window_records]
        delta_nfr_span = max(nfr_values) - min(nfr_values) if nfr_values else 0.0
        density_window = densities[max(0, index - window + 1) : index + 1]
        structural_density_mean = mean(density_window) if density_window else 0.0

        meets_thresholds = (
            load_span <= load_threshold
            and accel_mean <= accel_threshold
            and delta_nfr_span <= delta_nfr_threshold
            and structural_density_mean <= structural_density_threshold
        )

        if meets_thresholds:
            if active_start is None:
                active_start = index - window + 1
        elif active_start is not None:
            _append_event(active_start, index - 1)
            active_start = None

    if active_start is not None:
        _append_event(active_start, len(records) - 1)

    return events
