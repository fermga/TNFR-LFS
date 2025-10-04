"""Detection utilities for on-track operator events.

Each detection routine analyses a windowed sequence of :class:`TelemetryRecord`
objects and yields event descriptors when the observed behaviour exceeds the
configured thresholds.  The detectors are intentionally lightweight so that
they can be executed on every microsector without adding measurable overhead
to the orchestration pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import List, Mapping, Sequence

from .epi import TelemetryRecord

__all__ = ["OperatorEvent", "detect_al", "detect_oz", "detect_il"]


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
                _finalise_event("AL", records, active_start, index - 1, peak_value, lateral_threshold)
            )
            active_start = None
            peak_value = 0.0

    if active_start is not None:
        events.append(
            _finalise_event("AL", records, active_start, len(records) - 1, peak_value, lateral_threshold)
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
                _finalise_event("OZ", records, active_start, index - 1, peak_value, max(slip_threshold, yaw_threshold))
            )
            active_start = None
            peak_value = 0.0

    if active_start is not None:
        events.append(
            _finalise_event(
                "OZ",
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
        mean_speed = mean(abs(sample.speed) for sample in window_records)
        threshold = base_threshold + (speed_gain * mean_speed)
        deviation_peak = max(abs(sample.line_deviation) for sample in window_records)
        meets_threshold = deviation_peak >= threshold
        if meets_threshold:
            if active_start is None:
                active_start = index - window + 1
                peak_value = deviation_peak
            else:
                peak_value = max(peak_value, deviation_peak)
        elif active_start is not None:
            events.append(_finalise_event("IL", records, active_start, index - 1, peak_value, threshold))
            active_start = None
            peak_value = 0.0

    if active_start is not None:
        final_records = _window(records, len(records) - 1, window)
        mean_speed = mean(abs(sample.speed) for sample in final_records)
        threshold = base_threshold + (speed_gain * mean_speed)
        events.append(
            _finalise_event(
                "IL", records, active_start, len(records) - 1, peak_value, threshold
            )
        )

    return [event.as_mapping() for event in events]
