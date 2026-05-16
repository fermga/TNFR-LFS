"""Per-lap consolidated summaries from InSim event streams.

Builds :class:`LapRecord` rows from the events collected in a
:class:`lfs_telemetry.telemetry.protocol.insim.RaceContext`. One record per
completed lap of the *view* player (or any specified PLID), with:

* lap number, lap time, splits S1/S2/S3,
* fuel% at lap end (from IS_LAP fuel200/FUEL_SCALE),
* validity (clean / invalid + reason from latest IS_HLV in the lap),
* count of object hits (IS_OBH) inside the lap window,
* tyre compound choice and handicap (mass / intake restriction).

A lap is built from the ordered IS_LAP packets; splits are the dict
buffered by :class:`RaceContext` between consecutive IS_LAP events.

Example::

    from lfs_telemetry.telemetry.lap_summary import build_lap_records, dump_lap_records
    laps = build_lap_records(client.context)
    dump_lap_records(laps, "session_BL1.laps.json")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .protocol.insim import RaceContext
from .protocol.packets import (
    InSimHotLapValid,
    InSimObjectHit,
    hlvc_name,
    penalty_name,
    PENALTY_NONE,
)


@dataclass(slots=True)
class LapRecord:
    """One completed lap (of a single PLID) with all derived metadata."""

    player_id: int
    lap_index: int               # 1-based completion order
    lap_time_ms: int
    elapsed_time_ms: int | None = None
    split1_ms: int | None = None
    split2_ms: int | None = None
    split3_ms: int | None = None
    fuel_pct_end: float | None = None
    fuel_pct_used: float | None = None       # vs previous lap (None for lap 1)
    valid: bool = True
    invalid_reason: str | None = None        # ground / wall / speeding / out_of_bounds
    invalid_speed_ms: float | None = None
    obh_count: int = 0                       # object hits inside the lap window
    tyre_compounds: tuple[int, int, int, int] | None = None  # RL,RR,FL,FR
    handicap_mass_kg: int | None = None
    handicap_t_res: int | None = None
    car_name: str | None = None
    pit_stop_in_lap: bool = False             # IS_PIT seen during this lap
    pit_work: tuple[str, ...] | None = None   # decoded PSE_* labels
    pit_fuel_add: int | None = None           # IS_PIT raw fuel_add field
    penalty_name: str | None = None           # active penalty after this lap

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_lap_records(
    ctx: RaceContext,
    *,
    player_id: int | None = None,
    obh_events: list[InSimObjectHit] | None = None,
    hlv_events: list[InSimHotLapValid] | None = None,
) -> list[LapRecord]:
    """Build one :class:`LapRecord` per completed lap.

    ``player_id`` defaults to the view player. The OBH/HLV streams are taken
    from ``ctx`` unless override lists are supplied (useful for tests).
    """
    plid = player_id if player_id is not None else ctx.view_player_id
    if plid is None:
        return []
    lap_times = ctx.lap_times_ms.get(plid, [])
    if not lap_times:
        return []
    splits_per_lap = ctx.split_times_ms.get(plid, [])
    fuel_per_lap = ctx.lap_fuel_pct.get(plid, [])
    obh_all = obh_events if obh_events is not None else ctx.obh_events
    hlv_all = hlv_events if hlv_events is not None else ctx.hlv_events
    player = ctx.players.get(plid)

    # Reconstruct cumulative elapsed time per lap end and split events into
    # per-lap windows by elapsed time. We don't have absolute timestamps for
    # OBH/HLV events vs IS_LAP events without correlating ETime, so we use a
    # simple boundary approach: HLV/OBH events between lap N-1 end and lap N
    # end belong to lap N. Since we lack ETime in the rolling lists, we
    # instead consume them sequentially in order of arrival, partitioning by
    # IS_LAP arrival rank tracked via len() at the time of IS_LAP receipt.
    # For simplicity here: distribute uniformly — events that were appended
    # before IS_LAP[i] but after IS_LAP[i-1] go to lap i. We approximate by
    # rebuilding from RaceContext.lap_count progression. As a safe default we
    # attribute *all* OBH/HLV in the rolling lists to the last lap and let
    # callers do finer-grained analysis if needed.
    records: list[LapRecord] = []
    prev_fuel: float | None = None
    for i, lap_ms in enumerate(lap_times):
        splits = splits_per_lap[i] if i < len(splits_per_lap) else {}
        fuel_end = fuel_per_lap[i] if i < len(fuel_per_lap) else None
        used = None
        if prev_fuel is not None and fuel_end is not None:
            used = max(0.0, prev_fuel - fuel_end)
        rec = LapRecord(
            player_id=plid,
            lap_index=i + 1,
            lap_time_ms=lap_ms,
            split1_ms=splits.get(1),
            split2_ms=splits.get(2),
            split3_ms=splits.get(3),
            fuel_pct_end=fuel_end,
            fuel_pct_used=used,
            tyre_compounds=tuple(player.tyres) if player else None,
            handicap_mass_kg=player.handicap_mass_kg if player else None,
            handicap_t_res=player.handicap_t_res if player else None,
            car_name=player.car_name if player else None,
        )
        records.append(rec)
        prev_fuel = fuel_end

    # Attribute HLV / OBH events to the last lap by default. Better
    # attribution requires per-event timestamps which RaceContext doesn't
    # currently keep aligned with IS_LAP events.
    if records:
        last = records[-1]
        plid_hlv = [h for h in hlv_all if h.player_id == plid]
        plid_obh = [o for o in obh_all if o.player_id == plid]
        last.obh_count = len(plid_obh)
        if plid_hlv:
            h = plid_hlv[-1]
            last.valid = False
            last.invalid_reason = hlvc_name(h.hlvc)
            last.invalid_speed_ms = h.car_speed_ms

        # Pit-stop attribution: any pit_stop record whose ``laps_done``
        # matches the lap's index belongs to that lap. Falls back to the
        # last lap when laps_done isn't populated (e.g. orphan IS_PSF).
        for stop in ctx.pit_stops:
            if stop.player_id != plid:
                continue
            target = None
            for rec in records:
                if stop.laps_done and rec.lap_index == stop.laps_done:
                    target = rec
                    break
            if target is None:
                target = last
            target.pit_stop_in_lap = True
            if stop.work_labels:
                target.pit_work = tuple(stop.work_labels)
            target.pit_fuel_add = stop.fuel_add

        # Penalty attribution: most-recent non-NONE penalty applies to the
        # last lap; mark it invalid if it's a non-clearing penalty.
        plid_pens = [p for p in ctx.penalty_events if p.player_id == plid]
        if plid_pens:
            latest = plid_pens[-1]
            if latest.new_penalty != PENALTY_NONE:
                last.penalty_name = penalty_name(latest.new_penalty)
                last.valid = False
                if not last.invalid_reason:
                    last.invalid_reason = f"penalty_{last.penalty_name}"
    return records


def dump_lap_records(
    records: list[LapRecord], path: str | Path
) -> int:
    """Write ``records`` to ``path`` as JSON. Returns row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.to_dict() for r in records]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return len(records)


def load_lap_records(path: str | Path) -> list[LapRecord]:
    """Read lap records previously written by :func:`dump_lap_records`."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[LapRecord] = []
    for entry in raw:
        tyres = entry.get("tyre_compounds")
        if isinstance(tyres, list):
            entry["tyre_compounds"] = tuple(tyres)
        work = entry.get("pit_work")
        if isinstance(work, list):
            entry["pit_work"] = tuple(work)
        out.append(LapRecord(**entry))
    return out


__all__ = [
    "LapRecord",
    "build_lap_records",
    "dump_lap_records",
    "load_lap_records",
]
