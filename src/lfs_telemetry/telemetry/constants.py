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

# ---------------------------------------------------------------------------
# Unit conversions.
# Use these instead of inlining magic factors (``* 3.6``, ``- 273.15`` …).
# ---------------------------------------------------------------------------
# Speed.
SPEED_MS_TO_KMH: float = 3.6
SPEED_KMH_TO_MS: float = 1.0 / 3.6
SPEED_MS_TO_MPH: float = 2.2369362920544
# Temperature (the conversions are additive, not multiplicative).
TEMP_K_TO_C_OFFSET: float = 273.15
# Pressure.
PRESSURE_PA_TO_BAR: float = 1e-5
PRESSURE_PA_TO_PSI: float = 1.4503773773e-4
# Torque.
TORQUE_NM_TO_LBFT: float = 0.7375621493

__all__ = [
    "GRAVITY",
    "INSIM_DEFAULT_PORT",
    "OUTGAUGE_DEFAULT_PORT",
    "OUTSIM_DEFAULT_PORT",
    "PRESSURE_PA_TO_BAR",
    "PRESSURE_PA_TO_PSI",
    "SPEED_KMH_TO_MS",
    "SPEED_MS_TO_KMH",
    "SPEED_MS_TO_MPH",
    "TEMP_K_TO_C_OFFSET",
    "TORQUE_NM_TO_LBFT",
]
