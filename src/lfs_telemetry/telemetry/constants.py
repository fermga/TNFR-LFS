"""Physical constants shared across the telemetry pipeline.

Centralising these avoids subtle drift (e.g. ``9.81`` vs the NIST
standard ``9.80665``) that would otherwise creep in across calibrate,
observables, derived metrics, racing-line and Studio renderers.
"""

from __future__ import annotations

# Standard gravity (NIST, exact). All g-force, weight transfer, and
# tyre-load computations should multiply / divide by this single value.
GRAVITY: float = 9.80665

# Default LFS networking ports. Canonical defaults documented in
# ``docs/InSim.txt`` and the LFS ``cfg.txt`` template:
#   * InSim   = 29999 (TCP, runtime-enabled via ``/insim 29999``)
#   * OutSim  = 30000 (UDP, configured in ``cfg.txt`` ``OutSim Port``)
#   * OutGauge = 30001 (UDP, configured in ``cfg.txt`` ``OutGauge Port``)
# All CLI / GUI / library defaults import these so there is exactly one
# source of truth.
INSIM_DEFAULT_PORT: int = 29999
OUTSIM_DEFAULT_PORT: int = 30000
OUTGAUGE_DEFAULT_PORT: int = 30001

__all__ = [
    "GRAVITY",
    "INSIM_DEFAULT_PORT",
    "OUTSIM_DEFAULT_PORT",
    "OUTGAUGE_DEFAULT_PORT",
]
