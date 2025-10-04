"""Shared helpers for ΔNFR dissonance diagnostics."""

from __future__ import annotations

from math import isfinite
from typing import Sequence, Tuple

__all__ = [
    "YAW_ACCELERATION_THRESHOLD",
    "compute_useful_dissonance_stats",
]


# Empirical yaw acceleration threshold (rad/s²) beyond which chassis rotation
# is considered a high-energy event.  It was derived from telemetry reviews of
# GT3/GTE cars where corrective actions are typically required when yaw rate
# ramps exceed ~0.5 rad/s².
YAW_ACCELERATION_THRESHOLD = 0.5


def compute_useful_dissonance_stats(
    timestamps: Sequence[float],
    delta_series: Sequence[float],
    yaw_rate_series: Sequence[float],
    *,
    yaw_acc_threshold: float = YAW_ACCELERATION_THRESHOLD,
) -> Tuple[int, int, float]:
    """Return useful/high yaw samples and their ratio.

    Parameters
    ----------
    timestamps:
        Chronologically ordered timestamps matching ``delta_series``.
    delta_series:
        ΔNFR samples used to estimate the derivative.
    yaw_rate_series:
        Chassis yaw rate samples used to infer yaw acceleration.
    yaw_acc_threshold:
        Absolute yaw acceleration threshold that classifies a sample as a
        "high yaw energy" event.
    """

    useful_samples = 0
    high_energy_samples = 0

    length = min(len(timestamps), len(delta_series), len(yaw_rate_series))
    if length < 2:
        return useful_samples, high_energy_samples, 0.0

    for index in range(1, length):
        prev_time = timestamps[index - 1]
        current_time = timestamps[index]
        dt = current_time - prev_time
        if dt <= 0.0 or not isfinite(dt):
            continue

        delta_derivative = (delta_series[index] - delta_series[index - 1]) / dt
        yaw_acceleration = (yaw_rate_series[index] - yaw_rate_series[index - 1]) / dt

        if not (isfinite(delta_derivative) and isfinite(yaw_acceleration)):
            continue

        if abs(yaw_acceleration) < yaw_acc_threshold:
            continue

        high_energy_samples += 1
        if delta_derivative < 0.0:
            useful_samples += 1

    ratio = useful_samples / high_energy_samples if high_energy_samples else 0.0
    return useful_samples, high_energy_samples, ratio

