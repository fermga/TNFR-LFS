# `tnfr_core.delta_utils` module
Common helpers for ΔNFR distribution logic.

## Functions
- `distribute_weighted_delta(delta: float, signals: Mapping[str, float], *, min_total: float = 1e-09, min_signal: float = 0.0) -> Dict[str, float]`
  - Distribute ``delta`` proportionally to the magnitude of ``signals``.

Parameters
----------
delta:
    Total ΔNFR magnitude to distribute across the provided signals.
signals:
    Mapping of signal identifiers to their associated strength.  The
    absolute value of each entry is used when computing the proportional
    distribution.  Non-finite or non-numeric values are ignored.
min_total:
    Minimum aggregate signal magnitude required before falling back to a
    uniform distribution.  Any aggregate below this threshold, or a
    non-finite total, triggers an even split across the remaining keys.
min_signal:
    Signals with an absolute magnitude at or below this threshold are
    treated as zero-strength contributions.  They still participate in the
    uniform fallback should the aggregate strength be insufficient.

