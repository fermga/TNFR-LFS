"""Lap filtering for the TNFR Setup Advisor.

The advisor requires **N consecutive laps from the same stint** so the
network statistics aggregate over a stable car+track+setup state. A stint
is defined as a contiguous run with:

* identical car short name (``car`` channel),
* identical track code (``ctx_track``),
* no pit stop between consecutive laps,
* monotonically increasing ``time_ms`` boundary between files,
* no ``is_race_start`` mid-stint (race-start laps have a stopped grid
  segment that contaminates aggregates and must sit at lap 0).

The function is pure and returns the **longest** contiguous valid window
of length ``>= min_count``. If no such window exists, the returned
:class:`StintFilterResult` has ``laps=()`` and a non-``"ok"`` reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from lfs_telemetry.telemetry.lap import LapTelemetry


@dataclass(frozen=True)
class StintFilterResult:
    """Outcome of :func:`filter_consecutive_laps`."""

    laps: tuple[LapTelemetry, ...]
    reason: str  # "ok" or short failure tag (e.g. "car_mismatch")
    rejected: tuple[tuple[int, str], ...] = ()  # (lap_index, reason) pairs

    @property
    def ok(self) -> bool:
        return self.reason == "ok"


def filter_consecutive_laps(
    laps: Sequence[LapTelemetry],
    *,
    min_count: int = 5,
) -> StintFilterResult:
    """Return the longest contiguous stint of ``>= min_count`` laps.

    Lap order is taken as given (the caller is expected to sort by
    ``time_ms`` or filename). Validation runs lap by lap:

    * lap[i].summary must report the same ``car`` and ``track`` as lap[0]
    * ``pit_in_lap`` must be False for the *previous* lap (the in-lap
      before a pit ends the stint immediately after it)
    * ``is_race_start`` only allowed at index 0 of the window
    * boundary ``time_ms[i].first >= time_ms[i-1].last`` (no regression)
    """
    if not laps:
        return StintFilterResult(laps=(), reason="empty_input")
    if len(laps) < min_count:
        return StintFilterResult(
            laps=(),
            reason=f"need_{min_count}_got_{len(laps)}",
        )

    s0 = laps[0].summary
    ref_car = s0.get("car")
    ref_track = s0.get("track")

    runs: list[list[int]] = [[0]]
    rejected: list[tuple[int, str]] = []

    for i in range(1, len(laps)):
        prev_lap = laps[i - 1]
        cur_lap = laps[i]
        prev_s = prev_lap.summary
        cur_s = cur_lap.summary
        reason: str | None = None

        if cur_s.get("car") != ref_car:
            reason = "car_mismatch"
        elif cur_s.get("track") != ref_track:
            reason = "track_mismatch"
        elif prev_s.get("pit_in_lap", False):
            reason = "pit_break"
        elif cur_lap.is_race_start:
            reason = "race_start_mid_stint"
        else:
            prev_t_end = float(prev_lap.raw["time_ms"].iloc[-1])
            cur_t_start = float(cur_lap.raw["time_ms"].iloc[0])
            if cur_t_start < prev_t_end:
                reason = "time_regression"

        if reason is None:
            runs[-1].append(i)
        else:
            rejected.append((i, reason))
            runs.append([i])

    longest = max(runs, key=len)
    if len(longest) < min_count:
        return StintFilterResult(
            laps=(),
            reason=f"longest_run_{len(longest)}_lt_{min_count}",
            rejected=tuple(rejected),
        )
    return StintFilterResult(
        laps=tuple(laps[i] for i in longest),
        reason="ok",
        rejected=tuple(rejected),
    )
