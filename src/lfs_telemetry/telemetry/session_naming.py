"""Centralised filename-safe naming helpers for capture sessions.

Two call sites previously rolled their own sanitiser:

* :mod:`lfs_telemetry.cli` builds ``YYYYMMDD-HHMMSS_CAR_TRACK`` tags for
  per-lap CSV filenames *after* samples arrive (so it can infer the
  car/track from telemetry).
* :mod:`lfs_telemetry.app.capture_runner` builds the session-folder
  name ``{stem}_{YYYYMMDD-HHMMSS}`` *before* the capture starts (so it
  can hand the path to the CLI subprocess).

Both want the same primitive — "take a free-form string and turn it
into something safe to use as a path component" — but with slightly
different alphabets (cli used strict alnum; the runner additionally
allowed ``-`` and ``_`` from user-typed stems). This module exposes one
:func:`safe_token` helper that takes the allowed-extra-chars as an
argument, plus the high-level :func:`session_tag` that derives the
``ts_car_track`` tag from a buffer of :class:`TelemetrySample` items.

Keeping the helper here (instead of in :mod:`cli`) means the test
suite, the CLI and the Studio runner all converge on the same
sanitisation rules.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .live import TelemetrySample


def safe_token(value: str | None, *, extra_allowed: str = "") -> str:
    """Return ``value`` with everything outside ``[A-Za-z0-9]`` (plus
    any character listed in ``extra_allowed``) replaced by ``_``.

    Empty/blank input yields ``"unknown"``. The result has leading and
    trailing underscores stripped so consecutive substitutions don't
    leave dangling separators.
    """
    if not value:
        return "unknown"
    allowed = set(extra_allowed)
    cleaned = "".join(
        c if (c.isalnum() or c in allowed) else "_" for c in str(value)
    )
    return cleaned.strip("_") or "unknown"


def timestamp_tag(now: datetime | None = None) -> str:
    """``YYYYMMDD-HHMMSS`` for the current (or supplied) wall-clock."""
    return (now or datetime.now()).strftime("%Y%m%d-%H%M%S")


def session_tag(samples: "list[TelemetrySample]",
                *, now: datetime | None = None) -> str:
    """Build ``YYYYMMDD-HHMMSS_CAR_TRACK`` from the first sample that
    carries the relevant attribute. Missing values become ``unknown``.
    """
    car: str | None = None
    track: str | None = None
    for s in samples:
        if car is None and s.outgauge and s.outgauge.car:
            car = s.outgauge.car
        if track is None and s.race_context and s.race_context.track:
            track = s.race_context.track
        if car and track:
            break
    return f"{timestamp_tag(now)}_{safe_token(car)}_{safe_token(track)}"
