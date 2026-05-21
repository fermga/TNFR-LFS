"""Multi-mode average lap time helper (Detect&Monitor-style).

Three averages are produced from a per-driver list of completed lap
times in milliseconds plus the set of laps on which the driver entered
the pit lane:

* ``total``  — arithmetic mean of every completed lap.
* ``clean``  — arithmetic mean of every lap within 103 % of the best
               lap (the same threshold D&M uses, also a common
               convention for "race pace" in endurance analysis).
* ``stint``  — arithmetic mean excluding the lap 1 (initial out-lap
               or grid-start lap), the in-lap of every pit stop, and
               the immediately following out-lap. Approximates the
               clean stint pace D&M reports.

All values are pure ``int`` ms or ``None`` when not enough data is
available. The helper is pure logic (no I/O, no Qt) and exists in its
own module so it can be unit-tested in isolation and reused by both
the live publisher and any future report.
"""

from __future__ import annotations

from collections.abc import Iterable

CLEAN_THRESHOLD = 1.03


def compute_lap_averages(
    lap_times_ms: Iterable[int],
    pit_in_laps: Iterable[int] = (),
) -> dict[str, int | None]:
    """Return ``{"stint", "clean", "total"}`` averages in ms.

    ``lap_times_ms`` is the chronological list of completed lap times
    (lap 1 first). ``pit_in_laps`` are the 1-based lap indices on
    which the driver entered the pit lane (i.e. the in-laps); the
    out-lap (``in_lap + 1``) is also excluded from ``stint``.
    """
    laps = [int(t) for t in lap_times_ms if int(t) > 0]
    if not laps:
        return {"stint": None, "clean": None, "total": None}

    total_avg = round(sum(laps) / len(laps))

    best = min(laps)
    threshold = best * CLEAN_THRESHOLD
    clean_laps = [t for t in laps if t <= threshold]
    clean_avg: int | None = (
        round(sum(clean_laps) / len(clean_laps))
        if clean_laps else None
    )

    skip: set[int] = {1}  # lap 1 is the out-lap from grid / pit-start
    for in_lap in pit_in_laps:
        in_lap = int(in_lap)
        skip.add(in_lap)
        skip.add(in_lap + 1)
    stint_laps = [
        t for i, t in enumerate(laps, start=1) if i not in skip
    ]
    stint_avg: int | None = (
        round(sum(stint_laps) / len(stint_laps))
        if stint_laps else None
    )

    return {"stint": stint_avg, "clean": clean_avg, "total": total_avg}


__all__ = ["CLEAN_THRESHOLD", "compute_lap_averages"]
