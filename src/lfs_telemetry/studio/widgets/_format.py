"""Shared formatting helpers for studio widgets.

Centralises the small ``_fmt*`` functions that used to live (duplicated)
in setup_tab, sectors_tab, stint_tab, race_dashboard_dock and
live_modules. Each helper returns the LFS-style display string used by
the in-game UI (em-dash ``—`` or ``--:--.---`` for "no data", signed
floats for deltas, etc.).

Functions are tiny and side-effect free; importing the module is cheap
and safe for all widget code paths.
"""

from __future__ import annotations

import math

__all__ = [
    "EMDASH",
    "format_clock_ms",
    "format_finite",
    "format_gap_meters",
    "format_gap_seconds",
    "format_lap_time_ms",
    "format_lap_time_s",
    "format_signed_delta_ms",
    "format_signed_delta_s",
    "format_signed_finite",
]


EMDASH = "—"


def _is_finite(v: float | int | None) -> bool:
    if v is None:
        return False
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def format_finite(
    v: float | None,
    digits: int = 2,
    suffix: str = "",
    fallback: str = EMDASH,
) -> str:
    """Plain non-signed float with N decimals; ``fallback`` if not finite."""
    if not _is_finite(v):
        return fallback
    return f"{float(v):.{digits}f}{suffix}"


def format_signed_finite(
    v: float | None,
    digits: int = 1,
    suffix: str = "",
    fallback: str = EMDASH,
) -> str:
    """LFS-style signed value (e.g. camber/toe): ``+1.5°`` / ``-2.3°``."""
    if not _is_finite(v):
        return fallback
    return f"{float(v):+.{digits}f}{suffix}"


def format_lap_time_s(s: float | None, fallback: str = EMDASH) -> str:
    """``M:SS.mmm`` from seconds, ``SS.mmm`` for sub-minute laps."""
    if not _is_finite(s):
        return fallback
    m, r = divmod(float(s), 60.0)
    return f"{int(m):d}:{r:06.3f}" if m else f"{r:.3f}"


def format_lap_time_ms(ms: int | None, fallback: str = EMDASH) -> str:
    """Same as :func:`format_lap_time_s` but the input is milliseconds."""
    if ms is None or ms <= 0:
        return fallback
    return format_lap_time_s(float(ms) / 1000.0, fallback=fallback)


def format_clock_ms(ms: int | None) -> str:
    """Always-padded ``M:SS.mmm`` clock; ``--:--.---`` when missing."""
    if ms is None or ms < 0:
        return "--:--.---"
    minutes, rem = divmod(int(ms), 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{minutes:01d}:{seconds:02d}.{millis:03d}"


def format_signed_delta_s(s: float | None, fallback: str = EMDASH) -> str:
    """Signed seconds delta, 3-decimal: ``+0.123`` / ``-1.456``."""
    if not _is_finite(s):
        return fallback
    return f"{float(s):+.3f}"


def format_signed_delta_ms(ms: int | None, fallback: str = "--.---") -> str:
    """Signed seconds-from-ms delta: ``+0.123`` / ``-1.456``."""
    if ms is None:
        return fallback
    sign = "+" if ms >= 0 else "-"
    abs_ms = abs(int(ms))
    seconds, millis = divmod(abs_ms, 1000)
    return f"{sign}{seconds:01d}.{millis:03d}"


def format_gap_seconds(s: float | None, fallback: str = "--.---") -> str:
    """Right-aligned seconds gap with ``s`` suffix: ``  1.23s``."""
    if s is None:
        return fallback
    return f"{s:5.2f}s"


def format_gap_meters(m: float | None, fallback: str = EMDASH) -> str:
    """Meters below 1 km, km above. ``123.4 m`` / ``1.23 km``."""
    if m is None:
        return fallback
    if abs(m) >= 1000.0:
        return f"{m / 1000.0:.2f} km"
    return f"{m:.1f} m"
