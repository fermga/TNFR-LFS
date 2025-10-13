# `tnfr_core.dissonance` module
Shared helpers for ΔNFR dissonance diagnostics.

## Functions
- `compute_useful_dissonance_stats(timestamps: Sequence[float], delta_series: Sequence[float], yaw_rate_series: Sequence[float], *, yaw_acc_threshold: float = YAW_ACCELERATION_THRESHOLD) -> Tuple[int, int, float]`
  - Return useful/high yaw samples and their ratio.

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

## Attributes
- `YAW_ACCELERATION_THRESHOLD = 0.5`

