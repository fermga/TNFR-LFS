"""Shared helpers for ΔNFR dissonance diagnostics."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised when JAX is unavailable
    import jax.numpy as jnp  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - exercised when JAX is available
    jnp = None  # type: ignore[assignment]

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

    xp = jnp if jnp is not None else np

    timestamps_arr = xp.asarray(timestamps, dtype=xp.float64)
    delta_arr = xp.asarray(delta_series, dtype=xp.float64)
    yaw_rate_arr = xp.asarray(yaw_rate_series, dtype=xp.float64)

    length = int(
        min(timestamps_arr.shape[0], delta_arr.shape[0], yaw_rate_arr.shape[0])
    )
    if length < 2:
        return 0, 0, 0.0

    timestamps_arr = timestamps_arr[:length]
    delta_arr = delta_arr[:length]
    yaw_rate_arr = yaw_rate_arr[:length]

    dt = xp.diff(timestamps_arr)
    if dt.size == 0:
        return 0, 0, 0.0

    delta_derivative = xp.diff(delta_arr) / dt
    yaw_acceleration = xp.diff(yaw_rate_arr) / dt

    valid_mask = xp.isfinite(dt) & (dt > 0.0)
    valid_mask &= xp.isfinite(delta_derivative) & xp.isfinite(yaw_acceleration)

    high_energy_mask = valid_mask & (xp.abs(yaw_acceleration) >= yaw_acc_threshold)
    useful_mask = high_energy_mask & (delta_derivative < 0.0)

    high_energy_samples = int(xp.sum(high_energy_mask).item())
    useful_samples = int(xp.sum(useful_mask).item())

    ratio = float(useful_samples / high_energy_samples) if high_energy_samples else 0.0
    return useful_samples, high_energy_samples, ratio

