"""Physical constants shared across the telemetry pipeline.

Centralising these avoids subtle drift (e.g. ``9.81`` vs the NIST
standard ``9.80665``) that would otherwise creep in across calibrate,
observables, derived metrics, racing-line and Studio renderers.
"""

from __future__ import annotations

# Standard gravity (NIST, exact). All g-force, weight transfer, and
# tyre-load computations should multiply / divide by this single value.
GRAVITY: float = 9.80665

__all__ = ["GRAVITY"]
